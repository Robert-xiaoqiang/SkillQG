import torch
import pytorch_lightning as pl
import re
import os
import json
import glob

from evalpackage import Evaluator, Bleu, Rouge, Meteor


class TrainerModuleEvalMixin:
    bleu_scorer = Bleu(4)
    rouge_scorer = Rouge()

    def write_rank_prediction(self, hyp_questions, gold_standards, qids, subdir):
        prediction_dict = { }
        sum_dict = { }
        for hyp_question, gold_standard, qid in zip(hyp_questions, gold_standards, qids):
            prediction_dict[qid] = {
                'hyp': hyp_question,
                'ref': gold_standard
            }
            scores_dict = Evaluator.compute_individual_metrics(gold_standard, hyp_question)
            prediction_dict[qid].update(scores_dict)

            for k, v in scores_dict.items():
                sum_dict[k] = sum_dict.get(k, 0.0) + v
        
        average_dict = { }
        for k, v in sum_dict.items():
            average_dict[k] = v / float(len(hyp_questions))

        prediction_dirname = os.path.join(self.prediction_path, subdir)
        os.makedirs(prediction_dirname, exist_ok = True)
        prediction_filename = os.path.join(prediction_dirname, 'prediction_{}.json'.format(torch.distributed.get_rank()))
        average_filename = os.path.join(prediction_dirname, 'average_{}.json'.format(torch.distributed.get_rank()))
        with open(prediction_filename, 'w') as f:
            json.dump(prediction_dict, f,  ensure_ascii = True, indent = 4)
        with open(average_filename, 'w') as f:
            json.dump(average_dict, f, indent = 4)

        return average_dict

    def gather_rank_prediction(self, subdir):
        prediction_dirname = os.path.join(self.prediction_path, subdir)

        target_prediction = { }
        for source_prediction_filename in glob.glob(os.path.join(prediction_dirname, 'prediction_[0-9]*.json')):
            with open(source_prediction_filename) as f:
                source_prediction = json.load(f)
            target_prediction.update(source_prediction)
        
        target_average = { }
        for source_average_filename in glob.glob(os.path.join(prediction_dirname, 'average_[0-9]*.json')):
            with open(source_average_filename) as f:
                source_average = json.load(f)
            for k, v in source_average.items():
                l = target_average.setdefault(k, [ ])
                l.append(v)
        for k, l in target_average.items():
            if len(l):
                target_average[k] = sum(l) / len(l)

        target_prediction_filename = os.path.join(prediction_dirname, 'prediction_gather.json')
        target_average_filename = os.path.join(prediction_dirname, 'average_gather.json')

        with open(target_prediction_filename, 'w') as f:
            json.dump(target_prediction, f, ensure_ascii = True, indent = 4)
        with open(target_average_filename, 'w') as f:
            json.dump(target_average, f, indent = 4)

    def evaluate_training_batch(self, hyp_questions, gold_standards):
        hyp_input = { }
        gold_input = { }

        score, scores = TrainerModuleEvalMixin.bleu_scorer.compute_score_flatten(gold_standards, hyp_questions)

        return score[3]
    
    def save_huggingface_model(self, snapshot_key):
        # the typical snapshot_keys consist of best, last, epoch_#, iteration_# ...
        model_dirname = os.path.join(self.snapshot_path, snapshot_key)
        os.makedirs(model_dirname, exist_ok = True)

        self.model.save_pretrained(model_dirname)
        self.tokenizer.save_pretrained(model_dirname)


class TrainerModuleCLMMixin:
    '''
        decoding process of Causal Decoder-only LM (CLM)
    '''
    def decode(self, input_ids, sample_outputs):
        input_ids_len = input_ids.shape[-1] # have been padded as the same length of DOC_STRIDE (yet MAX_INPUT_LENGTH in training and validation step)
        
        # B*num_return_sequences x max_seq_len
        hyp_questions = self.tokenizer.batch_decode(sample_outputs[:, input_ids_len:], skip_special_tokens = True)

        return hyp_questions


class TrainerModuleSeq2SeqLMMixin:
    '''
        decoding process of Sequence-to-Sequence LM (Seq2SeqLM)
    '''
    def decode(self, input_ids, sample_outputs):
        # The output of Seq2SeqLM is just what we want, and it is unnecessary to filter out the previous part
        # This is just conditional generation and the model does not watch other things like CausalLM, such as old words, input words and previous words
        # B*num_return_sequences x max_query_length
        hyp_questions = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens = True)

        return hyp_questions


class TrainerModuleRLMixin:
    def get_rl_ratio(self, epoch):
        ret = 0.0
        
        epoch_diff = epoch - self.config.RL.START_EPOCH
        
        if epoch_diff >= 0:
            ret = max(self.config.RL.MAX_RL_RATIO, self.config.RL.RL_RATIO_BASE * (epoch_diff + 1))

        return ret

    def training_step_feeding(self, batch, batch_idx):
        lm_loss = super().training_step_feeding(batch, batch_idx)

        train_input, test_input, gold_questions = batch
        model_input = test_input
        input_ids = model_input['input_ids']
        input_ids_len = input_ids.size(-1)

        pg_loss = 0.0
        rl_ratio = self.get_rl_ratio(self.current_epoch)

        if rl_ratio > 0.001:
            sample_logits = self.model(**model_input)['logits']        
            # B x max_input_len, max_input_len-input_ids.shape[-1] shaped tuple of (B*numreturnsequences*beamsize x vocabsize)
            sample_ids, sample_logits = self.generate(input_ids, num_return_sequences = 1, do_sample = True, output_scores = True)
            sample_questions = self.decode(input_ids, sample_ids)

            sample_reward = self.evaluate_training_batch(sample_questions, list(gold_questions))
            sample_logits = torch.stack(sample_logits, dim = 0).permute(1, 0, 2) # B*numreturnsequences*beamsize x max_output_len x vocabsize
            
            sample_logits = sample_logits.view(-1, self.config.GENERATE.BEAM_SIZE, *sample_logits.shape[1:])[:, 0]
            sample_labels = train_input['labels'][:, input_ids_len:]
            sample_loss = self.criterion(sample_logits.reshape(-1, sample_logits.size(-1)), sample_labels.reshape(-1))

            with torch.no_grad():
                baseline_ids = self.generate(input_ids, num_return_sequences = 1, do_sample = False, output_scores = False)
            baseline_questions = self.decode(input_ids, baseline_ids)
            greedy_reward = self.evaluate_training_batch(baseline_questions, list(gold_questions))

            # SCST policy gradient
            # reduction == mean
            pg_loss = (sample_reward - greedy_reward) * sample_loss

        ret_loss = rl_ratio * pg_loss + (1 - rl_ratio) * lm_loss
        return [ ret_loss, lm_loss, pg_loss, rl_ratio, sample_questions, baseline_questions ]

    def training_step_logging(self, batch, batch_idx, loss):
        ret_loss, lm_loss, pg_loss, rl_ratio, sample_questions, baseline_questions = loss

        self.loss_avg_meter.update(ret_loss.item())

        self.log('train/loss_cur', ret_loss.item(), on_step = True, prog_bar = True, logger = False)
        self.log('train/loss_avg', self.loss_avg_meter.average(), on_step = True, prog_bar = True, logger = False)
        self.log('train/loss_lm', lm_loss.item(), on_step = True, prog_bar = True, logger = False)
        self.log('train/loss_rl', pg_loss.item(), on_step = True, prog_bar = True, logger = False)
        
        iteration = self.current_epoch * self.num_train_batch_per_epoch + batch_idx + 1
        self.logger.experiment.add_scalar('train/loss_cur', ret_loss.item(), iteration)
        self.logger.experiment.add_scalar('train/loss_avg', self.loss_avg_meter.average(), iteration)
        self.logger.experiment.add_scalar('train/loss_lm', lm_loss.item(), iteration)
        self.logger.experiment.add_scalar('train/loss_rl', pg_loss.item(), iteration)

        if not iteration % self.config.TRAIN.TB_FREQ:
            self.logger.experiment.add_text('train/sample_questions', sample_questions[0], iteration)
            self.logger.experiment.add_text('train/baseline_questions', baseline_questions[0], iteration)
        
        return ret_loss