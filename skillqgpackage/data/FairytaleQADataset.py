import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

import os
import json
import csv

from .DataCommon import CLMMixin, Seq2SeqLMMixin


class FairytaleQADatasetMixin(Dataset):
    '''
        Prerequisite object/instance-related attributes: self.config, self.split_set, self.tokenizer
        
        Note:
        self.config.TRAIN/VAL/TEST.DATASET_FILENAME is the root directory for the three splits
    '''
    def parse_and_build(self):
        assert self.config.TRAIN.DATASET_FILENAME == self.config.VAL.DATASET_FILENAME == self.config.TEST.DATASET_FILENAME, 'FairytaleQA directory parsing error'
        root_directory = self.config.TRAIN.DATASET_FILENAME
        context_root_directory = os.path.join(root_directory, 'section-stories')
        question_root_directory = os.path.join(root_directory, 'questions')
        
        assert self.split_set in { 'train', 'val', 'test' }, 'FairytaleQA splits parsing error'

        context_directory = os.path.join(context_root_directory, self.split_set)
        question_directory = os.path.join(question_root_directory, self.split_set)

        cxt_token = self.config.MODEL.SPECIAL_TOKENS.CXT_TOKEN
        ans_token = self.config.MODEL.SPECIAL_TOKENS.ANS_TOKEN
        rsk_token = self.config.MODEL.SPECIAL_TOKENS.RSK_TOKEN

        # # convert
        self.contexts = [ ]
        self.questions = [ ]
        self.answers = [ ]
        self.reasoning_skills = [ ]
        input_contexts = [ ]
        self.qids = [ ]

        def section2context(section_id, context_list):
            '''
                section_id: str or list[str], count from 1
                context_list: list of <section_id(str), its text(str)> pairs, removing its original header, count from 0 (its index + 1 == section_id)
            '''
            section_id_list = [ int(section_id) ] if section_id.find(',') == -1 else list(map(lambda s: int(s), section_id.split(',')))
            section_text_list = [ ]
            for sid in section_id_list:
                section_id_str, section_text = context_list[sid - 1] # list of strs
                section_text_list.append(section_text.strip())

            ret = ' '.join(section_text_list)
            
            return ret

        for question_file in os.listdir(question_directory):
            question_main_filename = os.path.splitext(question_file)[0]
            document_id_index = question_main_filename.find('-questions')
            assert document_id_index != -1, 'FairytaleQA filenames parsing error'
            document_id = question_main_filename[:document_id_index]

            context_file = document_id + '-story.csv'

            question_filename = os.path.join(question_directory, question_file)
            context_filename = os.path.join(context_directory, context_file)
 
            with open(question_filename) as qf, open(context_filename) as cf:
                qcsv, ccsv = csv.reader(qf), csv.reader(cf)
                qheader, cheader = next(qcsv), next(ccsv)
                
                clist = list(ccsv) # list of <section_id(str), its text(str)> pairs
                
                for qentry in qcsv:
                    '''
                    entry fields:
                    [ (0) 'question_id', (1) 'local-or-sum', (2) 'cor_section', (3) 'attribute1', (4) 'attribute2', (5) 'question', (6) 'ex-or-im1', (7) 'answer1', (8) 'answer2', (9) 'answer3', (10) 'ex-or-im2', (11) 'answer4', (12) 'answer5', (13) 'answer6']
                    '''
                    entry_id, reasoning_skill, question, answer = qentry[0], qentry[3], qentry[5], qentry[7]
                    section_id_in_context = qentry[2]
                    context = section2context(section_id_in_context, clist)
                    qid = document_id + '-' + entry_id

                    self.contexts.append(context)
                    self.questions.append(question)
                    self.answers.append(answer)
                    input_context = cxt_token + ' ' + context + ' ' + \
                                    ans_token + ' ' + answer + ' ' + \
                                    rsk_token + ' ' + reasoning_skill
                    
                    input_contexts.append(input_context)
                    self.qids.append(qid)
        # print(len(self.questions))
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


class FairytaleQACLMDataset(FairytaleQADatasetMixin, CLMMixin):
    def __init__(self, config, split_set, tokenizer):
        super().__init__()
        self.config = config
        self.split_set = split_set
        self.tokenizer = tokenizer

        self.parse_and_build()


class FairytaleQASeq2SeqLMDataset(FairytaleQADatasetMixin, Seq2SeqLMMixin):
    def __init__(self, config, split_set, tokenizer):
        super().__init__()
        self.config = config
        self.split_set = split_set
        self.tokenizer = tokenizer

        self.parse_and_build()