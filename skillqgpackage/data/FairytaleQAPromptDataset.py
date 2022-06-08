import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

import os
import json
import csv

from .DataCommon import CLMMixin, Seq2SeqLMMixin


class FairytaleQAPromptDatasetMixin(Dataset):
    '''
        Prerequisite object/instance-related attributes: self.config, self.split_set, self.tokenizer
    '''
    def parse_and_build(self):   
        assert self.split_set in { 'train', 'val', 'test' }, 'FairytaleQA splits parsing error'

        cxt_token = self.config.MODEL.SPECIAL_TOKENS.CXT_TOKEN
        ans_token = self.config.MODEL.SPECIAL_TOKENS.ANS_TOKEN
        rsk_token = self.config.MODEL.SPECIAL_TOKENS.RSK_TOKEN
        pik_token = self.config.MODEL.SPECIAL_TOKENS.PIK_TOKEN

        # convert
        self.contexts = [ ]
        self.questions = [ ]
        self.answers = [ ]
        self.reasoning_skills = [ ]
        self.additional_inputs = [ ]
        input_contexts = [ ]
        self.qids = [ ]

        if self.split_set == 'train':
            with open(self.config.TRAIN.DATASET_FILENAME) as f_train:
                train_set = json.load(f_train)
                self.data_file = train_set
        elif self.split_set == 'val':
            with open(self.config.VAL.DATASET_FILENAME) as f_val:
                val_set = json.load(f_val)
                self.data_file = val_set
        elif self.split_set == 'test':
            with open(self.config.TEST.DATASET_FILENAME) as f_test:
                test_set = json.load(f_test)
                self.data_file = test_set
        else:
            self.data_file = None

        talking_pairs = [ ]

        for qid, augmented_sample_entry in self.data_file.items():
            context, question, answer, reasoning_skill = augmented_sample_entry.pop('context'), augmented_sample_entry.pop('question'), augmented_sample_entry.pop('answer'), augmented_sample_entry.pop('reasoning_skill')
            for prompt_key, prompt_talking in augmented_sample_entry.items():
                for talking_key, talking_entry in prompt_talking.items():
                    talking_question = talking_entry['question']
                    talking_answers = talking_entry['answers']
                    for talking_answer in talking_answers:
                        talking_pair = talking_question + ' ' + talking_answer
                        talking_pairs.append(talking_pair)
            additional_input = ' '.join(talking_pairs)

            self.contexts.append(context)
            self.questions.append(question)
            self.answers.append(answer)
            self.reasoning_skills.append(reasoning_skill)
            self.additional_inputs.append(additional_input)
            input_context = cxt_token + ' ' + context + ' ' + \
                            ans_token + ' ' + answer + ' ' + \
                            rsk_token + ' ' + reasoning_skill + ' ' + \
                            pik_token + ' ' + additional_input
            
            input_contexts.append(input_context)
            self.qids.append(qid)

        '''
        convert the whole of dataset into torch.*Tensor (tensor {tensor from constant to scalar}), cache them in the CPU RAM, and feed a mini-batch of samples into GPU memory when necessary
        '''
        if self.split_set in { 'train', 'val' }:
            train_inputs, test_inputs = self.prepare_input(input_contexts, self.questions)
        else:
            train_inputs, test_inputs = self.prepare_input(input_contexts)

        self.train_inputs, self.test_inputs = train_inputs, test_inputs

    def __getitem__(self, index):
        train_input = { }
        test_input = { }
        question_text = self.questions[index]
        qid = self.qids[index]

        if self.split_set in { 'train', 'val' }:
            for key in self.train_inputs.keys():
                train_input[key] = self.train_inputs[key][index]
            for key in self.test_inputs.keys():
                test_input[key] = self.test_inputs[key][index]

            if self.split_set == 'train':
                return train_input, test_input, question_text
            elif self.split_set == 'val':
                return train_input, test_input, question_text, qid
        else:
            for key in self.test_inputs.keys():
                test_input[key] = self.test_inputs[key][index]

            # fake test in our experiments, because they contain the ground truth sequence
            return test_input, question_text, qid

    def __len__(self):
        return len(self.qids)


class FairytaleQAPromptCLMDataset(FairytaleQAPromptDatasetMixin, CLMMixin):
    def __init__(self, config, split_set, tokenizer):
        super().__init__()
        self.config = config
        self.split_set = split_set
        self.tokenizer = tokenizer

        self.parse_and_build()


class FairytaleQAPromptSeq2SeqLMDataset(FairytaleQAPromptDatasetMixin, Seq2SeqLMMixin):
    def __init__(self, config, split_set, tokenizer):
        super().__init__()
        self.config = config
        self.split_set = split_set
        self.tokenizer = tokenizer

        self.parse_and_build()