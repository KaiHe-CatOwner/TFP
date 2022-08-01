import os
import time
import json
import torch
import random
import numpy as np
from copy import copy
import torch.utils.data as data
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Union
from .fewshotsampler import FewshotSampler, SupportFewshotSampler, FewshotSampleBase
from numpy import argsort

ignore_index = 0 

def batch_convert_ids_to_tensors(batch_token_ids: List[List]) -> torch.Tensor:
    bz = len(batch_token_ids)
    batch_tensors = [torch.LongTensor(batch_token_ids[i]).squeeze(0) for i in range(bz)]
    batch_tensors = pad_sequence(batch_tensors, True, padding_value=ignore_index).long()
    return batch_tensors

def get_class_name(rawtag, word_map=None):
    # get (finegrained) class name
    if rawtag.startswith('B-') or rawtag.startswith('I-'):
        if word_map:
            return word_map[rawtag[2:]][-1]
        else:
            return rawtag[2:]
    else:
        if word_map:
            return word_map[rawtag][-1]
        else:
            return rawtag


class Sample(FewshotSampleBase):
    def __init__(self, filelines, word_map=None, other_token_list=[]):
        filelines = [line.split('\t') for line in filelines]
        self.words, self.tags = zip(*filelines)
        self.words = [word.lower() for word in self.words]

        # strip B-, I-
        self.normalized_tags = [get_class_name(tag, word_map) for tag in self.tags]
        # self.normalized_tags = list(map(get_class_name, (self.tags, word_map)))
        self.class_count = {}
        self.other_token_list = other_token_list
        
    def __count_entities__(self):
        current_tag = self.normalized_tags[0]
        for tag in self.normalized_tags[1:]:
            if tag == current_tag:
                continue
            else:
                if (current_tag not in self.other_token_list) and current_tag != 'O':
                    if current_tag in self.class_count:
                        self.class_count[current_tag] += 1
                    else:
                        self.class_count[current_tag] = 1
                current_tag = tag
        if (current_tag not in self.other_token_list) and current_tag != 'O':
            if current_tag in self.class_count:
                self.class_count[current_tag] += 1
            else:
                self.class_count[current_tag] = 1

    def get_class_count(self):
        if self.class_count:
            return self.class_count
        else:
            self.__count_entities__()
            return self.class_count

    def get_tag_class(self):
        # strip 'B' 'I' 
        tag_class = list(set(self.normalized_tags))
        for item in self.other_token_list:
            if item in tag_class:
                tag_class.remove(item)
        
        if 'O' in tag_class:
            tag_class.remove('O')
            
        return tag_class

    def valid(self, target_classes):
        return (set(self.get_class_count().keys()).intersection(set(target_classes))) and not (set(self.get_class_count().keys()).difference(set(target_classes)))

    def __str__(self):
        newlines = zip(self.words, self.tags)
        return '\n'.join(['\t'.join(line) for line in newlines])


class FewShotNERDatasetwithRandomSample(data.Dataset):
    """
        Fewshot NER Dataset
        """
    def __init__(self, filepath, tokenizer, embedding_layer, N, K, Q, max_length, word_map, ignore_label_id=-1, args=None, test=False):
        if not os.path.exists(filepath):
            raise Exception("[ERROR] Data file does not exist!")
            
        self.prompt_orderd = args.prompt_orderd
        self.embedding_layer = embedding_layer
        
        self.class2sampleid = {}
        self.test = test
        self.N = N
        self.K = K
        self.Q = Q
        self.word_map = word_map
        self.word2type = OrderedDict()
        self.type2index = OrderedDict()
        for index, (key, value) in enumerate(self.word_map.items()):
            self.type2index[self.word_map[key][0]] = index
            self.word2type[value[-1]] = key

        self.tokenizer = tokenizer
        self.BOS = tokenizer.cls_token
        self.EOS = tokenizer.sep_token

        self.max_length = max_length
        self.ignore_label_id = ignore_label_id
        self.other_token_list = args.other_token_list
        
        self.samples, self.classes = self.__load_data_from_file__(filepath)
        self.sampler = FewshotSampler(N, K, Q, self.samples, classes=self.classes)

        self.prompt = args.prompt
        self.pseudo_token = args.pseudo_token
        self.tokenizer.add_special_tokens({'additional_special_tokens': [args.pseudo_token]})

        if self.prompt in [0, 1]:
            # assert len(args.template) >= (self.N + 2), 'template is too short'
            self.template = args.template

    def __insert_sample__(self, index, sample_classes):
        for item in sample_classes:
            if item in self.class2sampleid:
                self.class2sampleid[item].append(index)
            else:
                self.class2sampleid[item] = [index]
    
    def __load_data_from_file__(self, filepath):
        samples = []
        classes = []
        with open(filepath, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        samplelines = []
        index = 0
        for line in lines:
            line = line.strip()
            if len(line.split('\t'))>1:
                samplelines.append(line)
            else:
                sample = Sample(samplelines, self.word_map, self.other_token_list)
                samples.append(sample)
                sample_classes = sample.get_tag_class()
                self.__insert_sample__(index, sample_classes)
                classes += sample_classes
                samplelines = []
                index += 1
        classes = list(set(classes))
        return samples, classes

    def __get_token_label_list__(self, words, tags):
        tokens = []
        word_labels = []
        labels = []
        for word, tag in zip(words, tags):
            word_token = self.tokenizer.tokenize(word, is_split_into_words=True)
            if word_token:
                tokens.extend(word_token)
                # tokenize the label to token id and make it the same number of tokens as the original words
                word_label = [tag] * len(word_token)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label = [self.tag2label[tag]] + [self.ignore_label_id] * (len(word_token) - 1)
                word_labels.extend(word_label)
                labels.extend(label)
        return tokens, word_labels, labels

    def _get_prompt_input(self, classes, encoder_inputs):
        prompt_pos  = []
        raw_len = len(encoder_inputs)
        if self.prompt in [0, 1]:
            # prompt = [self.pseudo_token]
            prompt = []
            for i in range(len(classes)):
                prompt += [classes[i]] 
                prompt_pos.append(raw_len+len(prompt)-1)
                prompt += [self.pseudo_token]
            prompt = prompt[:-1]
            
        # prompt 2
        elif self.prompt == 2:
            # prompt = [classes[0]]
            prompt = []
            for i in range(1, len(classes)):
                prompt += [classes[i]] 
                prompt_pos.append(raw_len+len(prompt)-1)
                prompt += [self.EOS]
            prompt = prompt[:-1]
            
        # prompt 3
        elif self.prompt == 3:
            prompt = [iclass for iclass in classes]
            prompt_pos= list(range(raw_len, raw_len+len(prompt)-1))

        return prompt, prompt_pos

    def _convert_tagging_label_to_index_label(self, tagging_label_ids):
        classes_token_ids = [self.tokenizer.get_vocab()[tclass] for tclass in self.tag2label.keys()]

        index_label_ids = []
        for ids in tagging_label_ids:
            index_label_ids.append(classes_token_ids.index(ids))
        return index_label_ids

    def __getraw__(self, tokens, word_labels, labels, prompt_tags, raw_prompt_indexs):
        # get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well), tags
        
        # split into chunks of length (max_length-2)
        # 2 is for special tokens [CLS] and [SEP] or <s> and </s>

        tokens_list = []
        word_labels_list = []
        olabels_list = []

        while len(tokens) > self.max_length - 2:
            tokens_list.append(tokens[:self.max_length-2])
            tokens = tokens[self.max_length-2:]
            
            word_labels_list.append(word_labels[:self.max_length-2])
            word_labels = word_labels[self.max_length-2:]
            
            olabels_list.append(labels[:self.max_length-2])
            labels = labels[self.max_length-2:]
            
        if len(tokens) > 1:
            tokens_list.append(tokens)
            word_labels_list.append(word_labels)
            olabels_list.append(labels)

        # add special tokens and get masks
        encoder_tokens_list = []
        tagging_labels_list = []
        index_labels_list = []
        labels_list = []
        prompt_tags_list = []
        prompt_pos_list = []
        for i, tokens in enumerate(tokens_list):
            encoder_inputs = [self.BOS] + tokens + [self.EOS]
            prompt, prompt_pos = self._get_prompt_input(prompt_tags, encoder_inputs)
            encoder_inputs +=prompt + [self.EOS]
            encoder_ids = self.tokenizer.convert_tokens_to_ids(encoder_inputs)
            tagging_labels = [self.other_token_list[0]] + word_labels_list[i] + [self.other_token_list[0]]
            tagging_label_ids = self.tokenizer.convert_tokens_to_ids(tagging_labels)
            index_label_ids = self._convert_tagging_label_to_index_label(tagging_label_ids)
            
            none_index =  prompt_tags.index(self.other_token_list[0])
            
            encoder_tokens_list.append(encoder_ids)
            tagging_labels_list.append(tagging_label_ids)
            index_labels_list.append(index_label_ids)
            labels_list.append([0] + olabels_list[i] + [0])
            
            prompt_tags_list.append(raw_prompt_indexs)
            prompt_pos_list.append(prompt_pos)
            
        return encoder_tokens_list, tagging_labels_list, index_labels_list, labels_list, prompt_tags_list, prompt_pos_list

    def __additem__(self, d, inputs, tagging_labels, index_labels, labels, prompt_tags_index, prompt_pos_list):
        d['inputs'] += inputs
        d['tagging_labels'] += tagging_labels
        d['index_labels'] += index_labels
        d['labels'] += labels
        d['prompt_tags_index'] += prompt_tags_index
        d['prompt_pos'] += prompt_pos_list
        
    def __populate__(self, idx_list, distinct_tags, savelabeldic=False):
        '''
            populate samples into data dict
            set savelabeldic=True if you want to save label2tag dict
            'index': sample_index
            'word': tokenized word ids
            'mask': attention mask in BERT
            'label': NER labels (index of N-way)
            'index_labels': used vocabulary index to map label, due to shuffle type words,
            'sentence_num': number of sentences in this set (a batch contains multiple sets)
            'text_mask': 0 for special tokens and paddings, 1 for real text
            '''
            
        dataset = {'inputs': [], 'tagging_labels':[], 'index_labels':[], 'labels':[], "prompt_tags_index":[], "prompt_pos":[]}
        
        for idx in idx_list:
            tokens, word_labels, labels = self.__get_token_label_list__(self.samples[idx].words, self.samples[idx].normalized_tags)
            if len(tokens) > self.max_length - 2:
                continue
            
            prompt_tags = self.__get_prompt_tags__(distinct_tags, tokens)
            raw_prompt_indexs = [self.type2index[i] for i in prompt_tags]
            encoder_inputs, tagging_labels, index_labels, ilabels, prompt_tags_index, prompt_pos_list \
                = self.__getraw__(tokens, word_labels, labels, prompt_tags, raw_prompt_indexs)
            self.__additem__(dataset, encoder_inputs, tagging_labels, index_labels, ilabels, prompt_tags_index, prompt_pos_list)
            
        dataset['sentence_num'] = [len(dataset['inputs'])]
        dataset['target_classes'] = distinct_tags
        
        if savelabeldic:
            dataset['label2tag'] = [self.label2tag]
        return dataset

    def __get_prompt_tags__(self, distinct_tags, tokens=None):
        prompt_tags = distinct_tags.copy()   
        if self.prompt_orderd == "shuffle":
            random.shuffle(prompt_tags)
        
        elif self.prompt_orderd == "fix":
            pass
        elif self.prompt_orderd =="similar":
            prompt_tags = self.__sim_prompt__(prompt_tags, tokens)
        else:
            raise Exception("prompt_orderd error !")
        return prompt_tags
    
    def __sim_prompt__(self, prompt_tags, tokens):
        raise Exception("no finish !")
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        prompt_ids = self.tokenizer.convert_tokens_to_ids(prompt_tags)
        with torch.no_grad():
            tokens_embedding = torch.mean(self.embedding_layer(torch.tensor(tokens_ids)), dim=0)
            prompt_embedding = self.embedding_layer(torch.tensor(prompt_ids))

            index = argsort((-(torch.pow(tokens_embedding - prompt_embedding, 2)).sum(-1)).tolist())
            index = index[::-1]
            prompt_tags= [prompt_tags[index[i]] for i in index]
        
        return prompt_tags
    
    def __getitem__(self, index):
        target_classes, support_idx, query_idx = self.sampler.__next__()
        distinct_tags = self.other_token_list + target_classes
        self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}  # use for convert label index in N-way
        self.label2tag = {idx:self.word2type[tag] for idx, tag in enumerate(distinct_tags)}
        
        support_set = self.__populate__(support_idx, distinct_tags)
        query_set = self.__populate__(query_idx, distinct_tags, savelabeldic=True)
        return support_set, query_set
    
    def __len__(self):
        return 1000000


class FewShotNERDatasetwithEpisodeSample(FewShotNERDatasetwithRandomSample):
    def __init__(self, filepath, tokenizer, N, K, Q, max_length, word_map, ignore_label_id=-1, args=None):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
            
        self.prompt_orderd = prompt_orderd
        self.embedding_layer = embedding_layer
        
        self.class2sampleid = {}
        self.N = N
        self.K = K
        self.Q = Q
        self.word_map = word_map
        self.word2type = OrderedDict()
        for key, value in self.word_map.items():
            self.word2type[value[-1]] = key

        self.tokenizer = tokenizer
        self.BOS = '[CLS]'
        self.EOS = '[SEP]'

        self.samples = self.__load_data_from_file__(filepath)
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id
        self.other_token_list = args.other_token_list
        
        self.pseudo_token = args.pseudo_token
        self.tokenizer.add_special_tokens({'additional_special_tokens': [args.pseudo_token]})

        self.prompt = args.prompt

        if self.prompt == 0:
            assert len(args.template) >= (self.N + 2), \
                f'template is too short: template length = {len(args.template)}, N = {self.N}.'
            self.template = args.template

    def __load_data_from_file__(self, filepath):
        with open(filepath)as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = json.loads(lines[i].strip())
        return lines

    def __populate__(self, data, target_classes, prompt_tags, savelabeldic=False):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'inputs': [], 'tagging_labels':[], 'index_labels':[], 'labels':[]}
        for i in range(len(data['word'])):
            normalized_tags = [get_class_name(tag, self.word_map) for tag in data['label'][i]]
            tokens, word_labels, labels = self.__get_token_label_list__(data['word'][i], normalized_tags)
            encoder_inputs, tagging_labels, index_labels, ilabels = self.__getraw__(tokens, word_labels, labels, prompt_tags)
            self.__additem__(dataset, encoder_inputs, tagging_labels, index_labels, ilabels)
        dataset['sentence_num'] = [len(dataset['inputs'])]
        dataset['target_classes'] = target_classes
        if savelabeldic:
            dataset['label2tag'] = [self.label2tag]
        return dataset

    def __getitem__(self, index):
        sample = self.samples[index]
        target_classes = [get_class_name(type, self.word_map) for type in sample['types']]
        support = sample['support']
        query = sample['query']
        # add 'O' and make sure 'O' is labeled 0
        distinct_tags = self.other_token_list + target_classes
        prompt_tags = distinct_tags.copy()
        random.shuffle(prompt_tags)
        self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx:self.word2type[tag] for idx, tag in enumerate(distinct_tags)}
        query_set = self.__populate__(query, distinct_tags, prompt_tags, savelabeldic=True)
        support_set = self.__populate__(support, distinct_tags, prompt_tags)
        return support_set, query_set

    def __len__(self):
        return len(self.samples)


class FewShotNERDatasetwithFixedSupportSet(FewShotNERDatasetwithRandomSample):

    def __init__(self, filepath, support_file_path,tokenizer, N, K, Q, max_length, word_map, ignore_label_id=-1, args=None) -> None:
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        
        if not os.path.exists(support_file_path):
            print("[ERROR] Support Data file does not exist!")
            assert(0)

        self.prompt_orderd = args.prompt_orderd
        
        self.class2sampleid = {}
        self.N = N
        self.K = K
        self.Q = Q
        self.word_map = word_map
        self.word2type = OrderedDict()
        self.type2index = OrderedDict()
        for index, (key, value) in enumerate(self.word_map.items()):
            self.type2index[self.word_map[key][0]] = index
            self.word2type[value[-1]] = key

        self.tokenizer = tokenizer
        self.BOS = tokenizer.cls_token
        self.EOS = tokenizer.sep_token

        self.data_size = N
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id
        self.other_token_list = args.other_token_list

        self.pseudo_token = args.pseudo_token
        self.tokenizer.add_special_tokens({'additional_special_tokens': [args.pseudo_token]})

        self.prompt = args.prompt

        if self.prompt in [0, 1]:
            assert len(args.template) >= (self.N + 2), 'template is too short'
            self.template = args.template

        self.support_samples, target_classes = self.__load_data_from_file__(support_file_path)

        # target_types = ['location-GPE', 'location-bodiesofwater', 'location-island', 'location-mountain', 'location-park', 'location-road/railway/highway/transit', 'location-other']
        # target_classes = ['nation', 'water', 'island', 'mountain', 'parks', 'road', 'location']
        
        target_classes = [type_words[0] for type_words in self.word_map.values() if type_words[0] not in self.other_token_list]
        self.distinct_tags = self.other_token_list + target_classes 
        self.samples, self.classes = self.__load_data_from_file__(filepath)

    def __generate_support_set__(self, target_classes):
        '''
            populate samples into data dict
            'index': sample_index
            'word': tokenized word ids
            'mask': attention mask in BERT
            'label': NER labels
            'sentence_num': number of sentences in this set (a batch contains multiple sets)
            'text_mask': 0 for special tokens and paddings, 1 for real text
            '''
        prompt_tags = self.__get_prompt_tags__(target_classes)
        self.tag2label = {tag:idx for idx, tag in enumerate(self.distinct_tags)}
        self.label2tag = {idx:self.word2type[tag] for idx, tag in enumerate(self.distinct_tags)}
        dataset = {'inputs': [], 'tagging_labels':[], 'index_labels':[], 'labels':[], "prompt_tags_index":[], "prompt_pos":[]}
        for sample in self.support_samples:
            tokens, word_labels, labels = self.__get_token_label_list__(sample.words, sample.normalized_tags)
            raw_prompt_indexs = [self.type2index[i] for i in prompt_tags]
            encoder_inputs, tagging_labels, index_labels, labels, prompt_indexs, prompt_pos_list = self.__getraw__(tokens, word_labels, labels, prompt_tags, raw_prompt_indexs)
            self.__additem__(dataset, encoder_inputs, tagging_labels, index_labels, labels, prompt_indexs, prompt_pos_list)
        dataset['sentence_num'] = [len(dataset['inputs'])]
        dataset['target_classes'] = self.distinct_tags

        return dataset

    def __getitem__(self, index):
        support_set = self.__generate_support_set__(self.distinct_tags)
        query_idx = range(index * self.data_size, (index + 1) * self.data_size)
        query_set = self.__populate__(query_idx, self.distinct_tags, savelabeldic=True)
        return support_set, query_set
    
    def __len__(self):
        return len(self.samples) // self.data_size


class FewShotNERDatasetforCaseStudy(FewShotNERDatasetwithRandomSample):

    def __init__(self, filepath, 
                        tokenizer, 
                        N, K, Q, 
                        max_length, 
                        word_map, 
                        ignore_label_id=-1, 
                        args=None) -> None:
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)

        self.ignore_label_id = ignore_label_id
        self.class2sampleid = {}
        self.N = N
        self.K = K
        self.Q = Q
        self.word_map = word_map
        self.word2type = OrderedDict()
        for key, value in self.word_map.items():
            self.word2type[value[-1]] = key

        self.tokenizer = tokenizer
        self.BOS = '[CLS]'
        self.EOS = '[SEP]'

        self.data_size = N
        self.max_length = max_length

        self.pseudo_token = args.pseudo_token
        self.tokenizer.add_special_tokens({'additional_special_tokens': [args.pseudo_token]})

        self.prompt = args.prompt

        if self.prompt in [0, 1]:
            assert len(args.template) >= (self.N + 2), 'template is too short'
            self.template = args.template

        target_types = ['location-GPE', 'location-bodiesofwater', 'location-island', 'location-mountain', 'location-park', 'location-road/railway/highway/transit', 'location-other']
        target_classes = ['nation', 'water', 'island', 'mountain', 'parks', 'road', 'location']
        self.distinct_tags = self.other_token_list + target_classes
        # self.support_set = self.__generate_support_set__()
        self.samples, self.classes = self.__load_data_from_file__(filepath)

    def __getitem__(self, index):
        prompt_tags = self.distinct_tags.copy()
        random.shuffle(prompt_tags)

        self.tag2label = {tag:idx for idx, tag in enumerate(self.distinct_tags)}
        self.label2tag = {idx:self.word2type[tag] for idx, tag in enumerate(self.distinct_tags)}

        support_set = self.__generate_support_set__(prompt_tags)
        query_idx = range(index * self.data_size, (index + 1) * self.data_size)
        support_set = query_set = self.__populate__(query_idx, self.distinct_tags, prompt_tags, savelabeldic=True)
        return support_set, query_set
    
    def __len__(self):
        return len(self.samples) // self.data_size


class FewShotNERDatasetforInLabelSpaceTrain(FewShotNERDatasetwithRandomSample):

    def __init__(self, filepath, tokenizer, N, K, Q, max_length, word_map, ignore_label_id=-1, args=None):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        
        self.class2sampleid = {}
        self.N = N
        self.K = K
        
        self.other_token_list = args.other_token_list
        self.prompt_orderd = args.prompt_orderd
        self.word_map = word_map
        self.word2type = OrderedDict()
        self.type2index = OrderedDict()
        for index, (key, value) in enumerate(self.word_map.items()):
            self.type2index[self.word_map[key][0]] = index
            self.word2type[value[-1]] = key

        self.tokenizer = tokenizer
        self.BOS = tokenizer.cls_token
        self.EOS = tokenizer.sep_token

        self.max_length = max_length
        self.ignore_label_id = ignore_label_id

        self.samples, self.classes = self.__load_data_from_file__(filepath)
        self.sampler = SupportFewshotSampler(self.N, self.K, self.samples, classes=self.classes)

        if self.tokenizer:
            self.prompt = args.prompt
            self.pseudo_token = args.pseudo_token
            self.tokenizer.add_special_tokens({'additional_special_tokens': [args.pseudo_token]})

            if self.prompt in [0, 1]:
                assert len(args.template) >= (self.N + 2), 'template is too short'
                self.template = args.template

    def __getitem__(self, index):
        target_classes, support_idx = self.sampler.__next__()
        # add 'none' and make sure 'none' is labeled 0
        distinct_tags = self.other_token_list + target_classes
        self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx:self.word2type[tag] for idx, tag in enumerate(distinct_tags)}
        support_set = self.__populate__(support_idx, distinct_tags, savelabeldic=True)
        
        return support_set, 0
    
    def __len__(self):
        return 50000


class FewShotNERDatasetforInLabelSpaceTest(FewShotNERDatasetwithRandomSample):
    
    def __init__(self, filepath, 
                        tokenizer, 
                        N, K, Q, 
                        max_length, 
                        word_map, 
                        ignore_label_id=-1, 
                        args=None) -> None:
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)

        self.ignore_label_id = ignore_label_id
        self.class2sampleid = {}
        self.N = N
        self.K = K
        
        self.other_token_list = args.other_token_list
        self.prompt_orderd = args.prompt_orderd
        self.word_map = word_map
        self.word2type = OrderedDict()
        self.type2index = OrderedDict()
        for index, (key, value) in enumerate(self.word_map.items()):
            self.type2index[self.word_map[key][0]] = index
            self.word2type[value[-1]] = key

        self.tokenizer = tokenizer
        self.BOS = tokenizer.cls_token
        self.EOS = tokenizer.sep_token

        self.data_size = N
        self.max_length = max_length
        self.other_token_list = args.other_token_list
        self.samples, self.classes = self.__load_data_from_file__(filepath)
        self.distinct_tags = self.other_token_list + self.classes

        if self.tokenizer:
            self.pseudo_token = args.pseudo_token
            self.tokenizer.add_special_tokens({'additional_special_tokens': [args.pseudo_token]})

            self.prompt = args.prompt

            if self.prompt in [0, 1]:
                assert len(args.template) >= (self.N + 2), 'template is too short'
                self.template = args.template

    def __getitem__(self, index):
        
        query_idx = range(index * self.data_size, (index + 1) * self.data_size)
        
        prompt_tags = self.distinct_tags.copy()
        random.shuffle(prompt_tags)
        self.tag2label = {tag:idx for idx, tag in enumerate(self.distinct_tags)}
        self.label2tag = {idx:self.word2type[tag] for idx, tag in enumerate(self.distinct_tags)}
        query_set = self.__populate__(query_idx, self.distinct_tags, savelabeldic=True)
        return 0, query_set
    
    def __len__(self):
        return len(self.samples) // self.data_size



def collate_fn(data):
    batch_support = {'inputs': [], 'tagging_labels':[],'index_labels': [], 'labels':[], 
                     'sentence_num': [], 'target_classes':[], 'prompt_tags_index':[], "prompt_pos":[]}
    
    batch_query = {'inputs': [], 'tagging_labels':[],  'index_labels':[], 'labels':[], 
                   'sentence_num': [], 'target_classes':[], 'label2tag': [], 'prompt_tags_index':[] , "prompt_pos":[]}
    
    support_sets, query_sets = zip(*data)
    for i in range(len(support_sets)):
        if support_sets[i]!=0:
            for k in batch_support:
                if k == 'target_classes' :
                    batch_support[k].append(support_sets[i][k])
                else:
                    batch_support[k] += support_sets[i][k]
        
        if query_sets[i]!=0:
            for k in batch_query:
                if k == 'target_classes' :
                    batch_query[k].append(query_sets[i][k])
                else:
                    batch_query[k] += query_sets[i][k]
                    
    if support_sets[0]!=0:                 
        for k in batch_support:
            if k in ['inputs', 'tagging_labels', 'index_labels', 'prompt_tags_index', "prompt_pos"]:
                batch_support[k] = batch_convert_ids_to_tensors(batch_support[k])  
                
            elif k == 'labels' :
                batch_support[k] = [torch.LongTensor(item) for item in batch_support[k]]
    
    if query_sets[0]!=0:        
        for k in batch_query:
            if k in ['inputs', 'tagging_labels', 'index_labels', 'prompt_tags_index', "prompt_pos"]:
                batch_query[k] = batch_convert_ids_to_tensors(batch_query[k])
            elif k == 'labels':
                batch_query[k] = [torch.LongTensor(item) for item in batch_query[k]]
            
    return batch_support, batch_query


def get_loader(filepath, tokenizer, N, K, Q, batch_size, max_length, model, word_map, 
               ignore_index=-1, args=None, num_workers=32, use_sampled_data=False, 
                support_file_path=None, use_fixed_support_set=False, train=False):
    assert tokenizer.pad_token_id==0
    
    if args.task == 'case_study':
        dataset = FewShotNERDatasetwithFixedSupportSet(filepath, support_file_path, 
                                                        tokenizer, N, K, Q, max_length, 
                                                        ignore_label_id=ignore_index, 
                                                        args=args, word_map=word_map)
    elif args.task == 'in-label-space':
        
        if train:
            dataset = FewShotNERDatasetforInLabelSpaceTrain(filepath, tokenizer, N, K, Q, max_length, 
                                                            ignore_label_id=ignore_index, 
                                                            args=args, word_map=word_map)
        else:
            dataset = FewShotNERDatasetforInLabelSpaceTest(filepath, tokenizer, N, K, Q, max_length, 
                                                            ignore_label_id=ignore_index, 
                                                            args=args, word_map=word_map)
    elif use_fixed_support_set:
        assert support_file_path is not None, "Support datset must be set."
        dataset = FewShotNERDatasetwithFixedSupportSet(filepath, support_file_path, 
                                                        tokenizer, N, K, Q, max_length, 
                                                        ignore_label_id=ignore_index, 
                                                        args=args, word_map=word_map)
        return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    elif not use_sampled_data:
        dataset = FewShotNERDatasetwithRandomSample(filepath, tokenizer, model, N, K, Q, max_length, 
                                                        ignore_label_id=ignore_index, 
                                                        args=args, word_map=word_map)
    else:
        dataset = FewShotNERDatasetwithEpisodeSample(filepath, tokenizer, N, K, Q, max_length, 
                                                        ignore_label_id=ignore_index,
                                                        args=args, word_map=word_map)
    
    data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn,
                                    )
    return data_loader
    

