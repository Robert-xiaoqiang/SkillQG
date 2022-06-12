import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

import os
import json
import csv

from .DataCommon import MLMMixin


class FairytaleQAMLCLSDataset(Dataset):
    def __init__(self, config, split_set, tokenizer):
        super().__init__()
        self.config = config
        self.split_set = split_set
        self.tokenizer = tokenizer

    def parse_and_build(self):
        assert self.config.TRAIN.DATASET_FILENAME == self.config.VAL.DATASET_FILENAME == self.config.TEST.DATASET_FILENAME, 'FairytaleQA directory parsing error'
        root_directory = self.config.TRAIN.DATASET_FILENAME
        context_root_directory = os.path.join(root_directory, 'section-stories')
        question_root_directory = os.path.join(root_directory, 'questions')
        
        assert self.split_set in { 'train', 'val', 'test' }, 'FairytaleQA splits parsing error'

        context_directory = os.path.join(context_root_directory, self.split_set)
        question_directory = os.path.join(question_root_directory, self.split_set)

        # # convert
        self.whole_stories = [ ]
        self.reasoning_skills_lists = [ ]
        self.labels_lists = [ ]

        input_contexts = [ ]
        self.sample_ids = [ ]

        '''
            0: remember
            1: understand
            2: apply
            3: analyze
            4: create
            5: evaluate
        '''
        # a 5-dimensional vector
        self.skill2label = {
            'character': 0,
            'setting': 0,
            'feeling': 1,
            'action': 2,
            'causal relationship': 3,
            'outcome resolution': 3,
            'prediction': 4
        }
        num_labels = len(set(self.skill2label.values()))

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
                
                # ccsv: iterable (object) of <section_id(str), its text(str)> pairs
                whole_story = ' '.join(map(lambda l: l[1], ccsv))
     
                reasoning_skills_set = set()
                labels_list = [ 0 ] * num_labels
                
                entry_id_index, reasoning_skill_index = qheader.index('question_id'), qheader.index('attribute1')

                for qentry in qcsv:
                    '''
                    entry fields:
                    [ (0) 'question_id', (1) 'local-or-sum', (2) 'cor_section', (3) 'attribute1', (4) 'attribute2', (5) 'question', (6) 'ex-or-im1', (7) 'answer1', (8) 'answer2', (9) 'answer3', (10) 'ex-or-im2', (11) 'answer4', (12) 'answer5', (13) 'answer6']
                    '''
                    entry_id, reasoning_skill = qentry[entry_id_index], qentry[reasoning_skill_index]

                    reasoning_skills_set.add(reasoning_skill)

                    label_index = self.skill2label[reasoning_skill]
                    labels_list[label_index] = 1

                self.whole_stories.append(whole_story)
                self.reasoning_skills_lists.append(list(reasoning_skills_set))
                self.labels_lists.append(labels_list)
                self.sample_ids.append(document_id)

        '''
        convert the whole of dataset into torch.*Tensor (tensor {tensor from constant to scalar}), cache them in the CPU RAM, and feed a mini-batch of samples into GPU memory when necessary
        '''
        if self.split_set in { 'train', 'val' }:
            train_inputs, test_inputs = self.prepare_input(self.whole_stories, self.labels_lists)
        else:
            train_inputs, test_inputs = self.prepare_input(self.whole_stories)

        self.train_inputs, self.test_inputs = train_inputs, test_inputs

    def __getitem__(self, index):
        train_input = { }
        test_input = { }
        labels_list = self.labels_lists[index]
        sample_ids = self.sample_ids[index]

        if self.split_set in { 'train', 'val' }:
            for key in self.train_inputs.keys():
                train_input[key] = self.train_inputs[key][index]
            for key in self.test_inputs.keys():
                test_input[key] = self.test_inputs[key][index]

            if self.split_set == 'train':
                return train_input, test_input, labels_list
            elif self.split_set == 'val':
                return train_input, test_input, labels_list, sample_ids
        else:
            for key in self.test_inputs.keys():
                test_input[key] = self.test_inputs[key][index]

            # fake test in our experiments, because they contain the ground truth sequence
            return test_input, labels_list, sample_ids

    def __len__(self):
        return len(self.sample_ids)


class FairytaleQAMLCLSMLMDataset(FairytaleQAMLCLSDataset, MLMMixin):
    def __init__(self, config, split_set, tokenizer):
        super().__init__()
        self.config = config
        self.split_set = split_set
        self.tokenizer = tokenizer

        # this in-memory construction on CPU is infeasible for longer sequence
        self.parse_and_build()