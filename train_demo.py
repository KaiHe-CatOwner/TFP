import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--ID', default='0', help='run ID')
parser.add_argument('--gpu', default='3', help='gpu numbers')
parser.add_argument('--pretrain_ckpt', default='bert-base-uncased', help='bert / bart pre-trained checkpoint')

parser.add_argument('--if_test_my_code', default=False, type=bool, help='if_test_my_code: True, False')
parser.add_argument('--only_test',  default=False, type=bool, help='only test: True, False')
parser.add_argument('--load_ckpt', default=None, help='load ckpt')

parser.add_argument('--mode', default='conll', help='choice=[inter, intra, supervised / A, B, C / conll(4), wnut(6), i2b2(18) / conll(4), mit-movie(12), ontonotes(18)]')
parser.add_argument('--task', default='in-label-space', help='training mode, must be in [few-nerd, domain-transfer, in-label-space, tagset-extension, case_study]')
parser.add_argument('--support_num', default=1, type=int, help='the id number of support set [X]/[0,1,2,3]/[1,2,3]/[X]')

parser.add_argument('--averaged_times', default=1, type=int, help='averaged_times')
parser.add_argument('--trainN', default=None, type=int, help='N in train')
parser.add_argument('--N', default=5, type=int, help='N way')
parser.add_argument('--K', default=5, type=int, help='K shot')
parser.add_argument('--Q', default=1, type=int, help='Num of query per class')

parser.add_argument('--lr', default=1e-5, type=float,  help='learning rate of Training')
parser.add_argument('--drop_ratio', default=0.5, type=float,  help='drop_ratio')
parser.add_argument('--batch_size', default=1, type=int, help='batch size 16')
parser.add_argument('--test_bz', default=1, type=int, help='test or val batch size 1/1/16/X')
parser.add_argument('--train_iter', default=10000, type=int, help='num of iters in training 10000/10000/10000/XX')
parser.add_argument('--val_iter', default=100, type=int, help='num of iters in validation 100/100/100/100')
parser.add_argument('--val_interval', default=100, type=int, help='val after how many training iters:200/200/50/XX')
parser.add_argument('--test_iter', default=5000, type=int, help='num of iters in testing 5000')

parser.add_argument('--adapt_step', default=10, type=int, help='adapting how many iters in validing or testing:5')
parser.add_argument('--struct', default=True, type=bool, help='StructShot parameter to re-normalizes the transition probabilities')
parser.add_argument('--adapt_auto', default=True, type=bool, help='True, False')
parser.add_argument('--threshold_beta', default=3, type=float, help='loss threshold for early stopping')

parser.add_argument('--prompt_semantic', default='description', help='prompt_semantic=["description", "synonyms", "random", "only_word", "mismatch", "continual", "out_desc", "out_word"]')
parser.add_argument('--prompt_orderd', default='shuffle', help='prompt_orderd=["shuffle", "fix", "similar"]')
parser.add_argument('--dis_method', default='euclidean', help='euclidean, dot, cosine, KL')
parser.add_argument('--dynamic_Other',  default=True, type=bool, help='only test: True, False')
parser.add_argument('--zero_shot', action='store_true', help='no adapting: True, False')

parser.add_argument('--other_token_list', default=["none"], type=list, help='none, if change this word, also need mofidy word_mapping.py, the first need be other  "entity"')
parser.add_argument('--t2c_tau', default=0.05, type=float, help='InforNCE temperature')
parser.add_argument('--c2c_tau', default=1, type=float, help='InforNCE temperature')
parser.add_argument('--t2t_tau', default=0, type=float, help='InforNCE temperature')

parser.add_argument('--span_weight', default=1, type=float, help='span_weight')
parser.add_argument('--type_weight', default=1, type=float, help='type_weight')

parser.add_argument('--if_projector', default=False, type=bool, help='True, False')
parser.add_argument('--dev_project',  default=False, type=bool, help='True, False')
parser.add_argument('--projector0_dim', default=512, type=int, help='projector0_dim')
parser.add_argument('--projector1_dim', default=384, type=int, help='projector1_dim')

parser.add_argument('--contrastive_task', default=True, type=bool, help='whether to run contrastive task or not')
parser.add_argument('--tagging_task', default=False, type=bool, help='whether to run tagging task or not')
parser.add_argument('--contrastive_weight', default=1.0, type=float, help='the rate of steps to begin to add Contrastive Task')
parser.add_argument('--prompt', default=1, type=int, help='choice in [0,1,2,3]: 0: Continue Prompt 1: Partition Prompt 2: Segment Prompt 3: Queue Prompt')
parser.add_argument('--label_word', type=int, default=0, help='the four different label words search method: 0: manual 1: auto 2: LMSearch 3: DataSearch')
parser.add_argument('--pseudo_token', default='<P>', type=str, help='pseudo_token')
parser.add_argument('--max_length', default=64, type=int, help='max length')
parser.add_argument('--adapt_lr', default=None, type=float, help='learning rate of Adapting')
parser.add_argument('--grad_iter', default=1, type=int, help='accumulate gradient every x iterations')
parser.add_argument('--fp16', action='store_true', default=False, help='use nvidia apex fp16')
parser.add_argument('--ckpt_name', type=str, default='', help='checkpoint name.')
parser.add_argument('--seed', type=int, default=42,help='random seed')
parser.add_argument('--ignore_index', type=int, default=-1, help='label index to ignore when calculating loss and metrics')
parser.add_argument('--use_sampled_data', action='store_true', default=False, help='use released sampled data, the data should be stored at "data/episode-data/" ')
parser.add_argument('--use_sgd_for_lm', action='store_true', default=False, help='use SGD instead of AdamW for BERT.')
opt = parser.parse_args()

import os
import sys
import numpy as np
import json
import torch
import random
import copy
import torch
from collections import OrderedDict
from transformers import BertTokenizer, BertConfig
from model.word_encoder import BERTWordEncoder
from util.framework import FewShotNERFramework
from model.copner import COPNER
from torch.nn.utils.rnn import pad_sequence
from util.data_loader import get_loader
from util.utils import print_execute_time
from util.word_synonyms_mapping import WORD_MAP_SYN,  ONTONOTES_WORD_MAP_SYN, CONLL_WORD_MAP_SYN, WNUT_WORD_MAP_SYN, I2B2_WORD_MAP_SYN, MOVIES_WORD_MAP_SYN


from util.word_descript_mapping import WORD_MAP_DES,  ONTONOTES_WORD_MAP_DES, CONLL_WORD_MAP_DES, WNUT_WORD_MAP_DES, I2B2_WORD_MAP_DES, MOVIES_WORD_MAP_DES
if opt.prompt_semantic == "mismatch":
    from util.word_mapping_mismatch import WORD_MAP,      ONTONOTES_WORD_MAP,     CONLL_WORD_MAP,     WNUT_WORD_MAP,     I2B2_WORD_MAP,     MOVIES_WORD_MAP
elif opt.prompt_semantic == "random":
    from util.word_mapping_unused import   WORD_MAP,      ONTONOTES_WORD_MAP,     CONLL_WORD_MAP,     WNUT_WORD_MAP,     I2B2_WORD_MAP,     MOVIES_WORD_MAP
else :
    from util.word_mapping import          WORD_MAP,      ONTONOTES_WORD_MAP,     CONLL_WORD_MAP,     WNUT_WORD_MAP,     I2B2_WORD_MAP,     MOVIES_WORD_MAP
    
    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.backends.cudnn.deterministic = True
    
if opt.task == 'few-nerd':
    if (opt.N != 5) and (opt.N != 10):
        raise Exception("few-nerd N-way = 5 or 10")
    if (opt.K != 1) and (opt.K != 5):
        raise Exception("few-nerd k-shot = 1 or 5")
    

if opt.task == 'domain-transfer' :
    if (opt.K != 1) and (opt.K != 5):
        raise Exception("few-nerd k-shot = 1 or 5")
    if opt.mode =="conll":
        opt.N=4
    if opt.mode =="wnut":
        opt.N=6
    if opt.mode =="i2b2":
        opt.N=18
        

if opt.task == 'in-label-space':
    opt.hight_source_flag = False
    if opt.mode =="conll":
        opt.N=4
    if opt.mode =="mit-movie":
        opt.N=12
    if opt.mode =="ontonotes":
        opt.N=18
    if opt.K==1:
        opt.K=5
else:
    opt.hight_source_flag = True
       
if opt.if_test_my_code:
    opt.train_iter = 3
    opt.val_iter = 1
    opt.test_iter = 1
    opt.val_interval = 1
    opt.adapt_step = 1

if "entity" not in opt.other_token_list:
    del WORD_MAP["E"]
    del ONTONOTES_WORD_MAP["E"]
    del CONLL_WORD_MAP["E"]
    del WNUT_WORD_MAP["E"]
    del I2B2_WORD_MAP["E"]
    del MOVIES_WORD_MAP["E"]
    
    del WORD_MAP_DES["E"]
    del ONTONOTES_WORD_MAP_DES["E"]
    del CONLL_WORD_MAP_DES["E"]
    del WNUT_WORD_MAP_DES["E"]
    del I2B2_WORD_MAP_DES["E"]
    del MOVIES_WORD_MAP_DES["E"]
    
    del WORD_MAP_SYN["E"]
    del ONTONOTES_WORD_MAP_SYN["E"]
    del CONLL_WORD_MAP_SYN["E"]
    del WNUT_WORD_MAP_SYN["E"]
    del I2B2_WORD_MAP_SYN["E"]
    del MOVIES_WORD_MAP_SYN["E"]
              
def get_description(WORD_MAP, tokenizer):
    token_list = []
    for k ,v in WORD_MAP.items():
        tmp = [tokenizer.cls_token] + tokenizer.tokenize(v[0]) + [tokenizer.sep_token]
        token_list.append(torch.tensor(tokenizer.convert_tokens_to_ids(tmp)))
    return pad_sequence(token_list, True, padding_value=tokenizer.pad_token_id).long()

def get_syn(WORD_MAP, tokenizer):
    syn_dic = {}
    for k, v in WORD_MAP.items():
        syn_dic[k] = tokenizer.convert_tokens_to_ids(v)
    return syn_dic

def get_data(opt, trainN, N, K, Q, max_length, tokenizer, model):
    print('loading data... \n')
    
    if opt.task == 'case_study':
        opt.data = f'data/few-nerd/intra/Figure_data/target_200_no_other.txt'
        opt.support = f'data/few-nerd/intra/Figure_data/support/'+str(opt.support_num)+'.txt'
        train_data_loader = val_data_loader = test_data_loader = get_loader(opt.data, tokenizer, word_map = opt.test_word_map,
                N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length,  model=model,
                ignore_index=opt.ignore_index, args=opt, support_file_path=opt.support,
                use_fixed_support_set=True)

    elif opt.task == 'few-nerd':
        if not opt.use_sampled_data:
            opt.train = f'data/few-nerd/{opt.mode}/train.txt'
            opt.dev = f'data/few-nerd/{opt.mode}/test.txt'
            opt.test = f'data/few-nerd/{opt.mode}/test.txt'
            if not (os.path.exists(opt.train) and os.path.exists(opt.dev) and os.path.exists(opt.test)):
                os.system(f'bash data/few-nerd/download.sh {opt.mode}')
        else:
            opt.train = f'data/few-nerd/episode-data/{opt.mode}/train_{opt.N}_{opt.K}.jsonl'
            opt.dev = f'data/few-nerd/episode-data/{opt.mode}/test_{opt.N}_{opt.K}.jsonl'
            opt.test = f'data/few-nerd/episode-data/{opt.mode}/test_{opt.N}_{opt.K}.jsonl'
            if not (os.path.exists(opt.train) and os.path.exists(opt.dev) and os.path.exists(opt.test)):
                os.system(f'bash data/few-nerd/download.sh episode-data')
                os.system('unzip -d data/few-nerd/ data/few-nerd/episode-data.zip')

        print(f'loading few-nerd train data: {opt.train}')
        train_data_loader = get_loader(opt.train, tokenizer, word_map = opt.train_word_map, 
                N=trainN, K=K, Q=Q, batch_size=opt.batch_size, max_length=max_length, model=model,
                ignore_index=opt.ignore_index, args=opt, use_sampled_data=opt.use_sampled_data)
        print(f'loading few-nerd eval data: {opt.dev}')
        val_data_loader = get_loader(opt.dev, tokenizer, word_map = opt.dev_word_map, 
                N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length, model=model,
                ignore_index=opt.ignore_index, args=opt, use_sampled_data=opt.use_sampled_data)
        print(f'loading few-nerd test data: {opt.test}')
        test_data_loader = get_loader(opt.test, tokenizer, word_map = opt.test_word_map, 
                N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length, model=model,
                ignore_index=opt.ignore_index, args=opt, use_sampled_data=opt.use_sampled_data)

    elif opt.task == 'tagset-extension':
        opt.train = f'data/ontonotes/{opt.mode}/train_all.txt'
        opt.support = f'data/ontonotes/{opt.mode}/support-{K}shot/{opt.support_num}.txt'
        opt.test = f'data/ontonotes/{opt.mode}/test_all.txt'

        print(f'loading tagset-extension train data: {opt.train}')
        train_data_loader = get_loader(opt.train, tokenizer, word_map = opt.train_word_map,
                N=trainN, K=K, Q=Q, batch_size=opt.batch_size, max_length=max_length, model=model,
                ignore_index=opt.ignore_index, args=opt)
        
        print(f'loading tagset-extension val data: {opt.test}')
        val_data_loader = get_loader(opt.test, tokenizer, word_map = opt.test_word_map,
                N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length, model=model,
                ignore_index=opt.ignore_index, args=opt, support_file_path=opt.support,
                use_fixed_support_set=True)
        
        print(f'loading tagset-extension test data: {opt.test}')
        test_data_loader = get_loader(opt.test, tokenizer, word_map = opt.test_word_map,
                N=N, K=K, Q=Q, batch_size=opt.batch_size, max_length=max_length, model=model,
                ignore_index=opt.ignore_index, args=opt, support_file_path=opt.support,
                use_fixed_support_set=False)

    elif opt.task == 'domain-transfer':
        opt.train = 'data/ontonotes/train_new.txt'
        opt.support = f'data/domain/{opt.mode}/support-{K}shot/{opt.support_num}.txt'
        opt.test = f'data/domain/{opt.mode}/test.txt'
        
        print(f'loading domain-transfer train data: {opt.train}')
        train_data_loader = get_loader(opt.train, tokenizer, word_map = opt.train_word_map, model=model,
                N=trainN, K=K, Q=Q, batch_size=opt.batch_size, max_length=max_length, 
                ignore_index=opt.ignore_index, args=opt)
        
        print(f'loading domain-transfer val data: {opt.test}')
        val_data_loader = get_loader(opt.test, tokenizer, word_map = opt.dev_word_map, model=model,
                N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length, 
                ignore_index=opt.ignore_index, args=opt, support_file_path=opt.support,
                use_fixed_support_set=True)
        
        print(f'loading domain-transfer test data: {opt.test}')
        test_data_loader = get_loader(opt.test, tokenizer, word_map = opt.test_word_map, model=model,
                N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length, 
                ignore_index=opt.ignore_index, args=opt, support_file_path=opt.support,
                use_fixed_support_set=True)

    elif opt.task == 'in-label-space':
        opt.train = f'data/size-influence/{opt.mode}/{K}shot/{opt.support_num}.txt'
        opt.test = f'data/size-influence/{opt.mode}/test.txt'

        print(f'loading in-label-space support data: {opt.train}')
        train_data_loader = get_loader(
                        opt.train, tokenizer, word_map = opt.test_word_map,
                        N=N, K=K, Q=Q, batch_size=opt.batch_size, max_length=max_length, model=model,
                        ignore_index=opt.ignore_index, args=opt, train=True)
        
        print(f'loading in-label-space test data: {opt.test}')
        val_data_loader = test_data_loader = get_loader(
                        opt.test, tokenizer, word_map = opt.test_word_map,
                        N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length, model=model,
                        ignore_index=opt.ignore_index, args=opt)
    
    else:
        raise Exception("not Implement !!!")
    
    return opt, train_data_loader, val_data_loader, test_data_loader

def add_word_map(opt,  tokenizer):
        
    if opt.task == 'case_study':
        opt.train_word_map = opt.dev_word_map = opt.test_word_map = WORD_MAP
        opt.train_desc_input = opt.dev_desc_input = opt.test_desc_input = get_description(WORD_MAP_DES, tokenizer)
        
    elif opt.task == 'few-nerd':
        opt.train_word_map = opt.dev_word_map = opt.test_word_map = WORD_MAP
        opt.train_syn = opt.dev_syn = opt.test_syn = get_syn(WORD_MAP_SYN, tokenizer)
        opt.train_desc_input = opt.dev_desc_input = opt.test_desc_input = get_description(WORD_MAP_DES, tokenizer)

    elif opt.task == 'domain-transfer' :
        opt.train_word_map = ONTONOTES_WORD_MAP
        opt.train_syn = get_syn(ONTONOTES_WORD_MAP_SYN, tokenizer)   
        opt.train_desc_input = get_description(ONTONOTES_WORD_MAP_DES, tokenizer)
        if opt.mode == 'conll':
            opt.dev_word_map = opt.test_word_map = CONLL_WORD_MAP
            opt.dev_syn = opt.test_syn = get_syn(CONLL_WORD_MAP_SYN, tokenizer)   
            opt.dev_desc_input = opt.test_desc_input = get_description(CONLL_WORD_MAP_DES, tokenizer)
        elif opt.mode == 'wnut':
            opt.dev_word_map = opt.test_word_map = WNUT_WORD_MAP
            opt.dev_syn = opt.test_syn = get_syn(WNUT_WORD_MAP_SYN, tokenizer)
            opt.dev_desc_input = opt.test_desc_input = get_description(WNUT_WORD_MAP_DES, tokenizer)
        elif opt.mode == 'i2b2':
            opt.dev_word_map = opt.test_word_map = I2B2_WORD_MAP
            opt.dev_syn = opt.test_syn = get_syn(I2B2_WORD_MAP_SYN, tokenizer)
            opt.dev_desc_input = opt.test_desc_input = get_description(I2B2_WORD_MAP_DES, tokenizer)
            
    elif opt.task == 'in-label-space':
        if opt.mode == 'conll':
            opt.train_word_map = opt.dev_word_map = opt.test_word_map = CONLL_WORD_MAP
            opt.train_syn = opt.dev_syn = opt.test_syn = get_syn(CONLL_WORD_MAP_SYN, tokenizer)
            opt.train_desc_input = opt.dev_desc_input = opt.test_desc_input = get_description(CONLL_WORD_MAP_DES, tokenizer)
        if opt.mode == 'ontonotes':
            opt.train_word_map = opt.dev_word_map = opt.test_word_map = ONTONOTES_WORD_MAP
            opt.train_syn = opt.dev_syn = opt.test_syn = get_syn(ONTONOTES_WORD_MAP_SYN, tokenizer)
            opt.train_desc_input = opt.dev_desc_input = opt.test_desc_input = get_description(ONTONOTES_WORD_MAP_DES, tokenizer)
        elif opt.mode == 'mit-movie':
            opt.train_word_map = opt.dev_word_map = opt.test_word_map = MOVIES_WORD_MAP
            opt.train_syn = opt.dev_syn = opt.test_syn = get_syn(MOVIES_WORD_MAP_SYN, tokenizer)
            opt.train_desc_input = opt.dev_desc_input = opt.test_desc_input = get_description(MOVIES_WORD_MAP_DES, tokenizer)

    else:
        raise Exception("not Implement !!!")
    
    return opt


@print_execute_time
def main(train_index, opt):
    
    if opt.load_ckpt is not None:
        opt.averaged_times = 1
    
    trainN = opt.trainN if opt.trainN is not None else opt.N
    N = opt.N
    K = opt.K
    Q = opt.Q
    max_length = opt.max_length
    
    if opt.prompt in [0, 1]:
        if N == 5:
            opt.template = [1] * (N + 2)
        else:
            opt.template = [1] * (N + 2)
    
    if opt.adapt_lr is None and opt.lr:
        opt.adapt_lr = opt.lr

    print('if_test_my_code:', opt.if_test_my_code)
    print('opt.train_iter:', opt.train_iter)
    print('ID: {}'.format(opt.ID))
    print('GPU: {}'.format(opt.gpu))
    print("{}-way-{}-shot Few-Shot NER".format(N, K))
    print('task: {}'.format(opt.task))
    print('mode: {}'.format(opt.mode))
    print('label word: {}'.format(opt.label_word))
    print("support: {}".format(opt.support_num))
    print("Contrastive Task: {}".format(opt.contrastive_task))
    print("Tagging Task: {}".format(opt.tagging_task))
    print("max_length: {}".format(max_length))
    print("batch_size: {}".format(opt.test_bz if opt.only_test else opt.batch_size))
    print('loading model and tokenizer...')
    pretrain_ckpt = opt.pretrain_ckpt or 'facebook/bart-base'
    config = BertConfig.from_pretrained(pretrain_ckpt)
    
    tokenizer = BertTokenizer.from_pretrained(pretrain_ckpt)
    add_dic = {'additional_special_tokens': [opt.pseudo_token]}
    tokenizer.add_special_tokens(add_dic)
    pseudo_token_id = tokenizer.get_vocab()[opt.pseudo_token]
    opt.tokenizer = tokenizer
    
    sentence_encoder = BERTWordEncoder.from_pretrained(pretrain_ckpt, config=config, args=opt)
    sentence_encoder.bert.resize_token_embeddings(len(tokenizer))
    opt = add_word_map(opt, tokenizer)
    model = COPNER(sentence_encoder, opt, opt.train_word_map if not opt.only_test else opt.test_word_map)
    fix_embedding = copy.deepcopy(model.sentence_encoder.bert.embeddings.word_embeddings)
    opt, train_data_loader, val_data_loader, test_data_loader = get_data(opt, trainN, N, K, Q, max_length, tokenizer, fix_embedding)
    framework = FewShotNERFramework(train_index, opt, train_data_loader, val_data_loader, test_data_loader, tokenizer)
    
    prefix = '-'.join([opt.task, opt.mode, str(N), str(K), 'seed'+str(opt.seed)])
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
        
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}-{}.pt'.format(opt.ID, prefix)

    i = 1
    while os.path.exists(ckpt):
        ckpt = 'checkpoint/{}-{}-{}.pt'.format(opt.ID, prefix, i)
        i += 1

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        print('model-save-path:', ckpt)

        best_precision, best_recall, best_f1 = framework.train(model, prefix, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                                    val_interval=opt.val_interval, fp16=opt.fp16, train_iter=opt.train_iter, 
                                    warmup_step=int(opt.train_iter * 0.0),  adapt_steps=opt.adapt_step, 
                                    val_iter=opt.val_iter,  learning_rate=opt.lr, 
                                    use_sgd_for_lm=opt.use_sgd_for_lm)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

        print("only test ......")
        best_precision, best_recall, best_f1, fp_error, fn_error, within_error, outer_error, loss, correct_cnt, span_correct_ratio \
                        = framework.eval(model, opt.test_iter, adapt_steps=opt.adapt_step, ckpt=ckpt, word_map=opt.test_word_map)

    return best_precision, best_recall, best_f1, ckpt


if __name__ == "__main__":
    
    total_P = []
    total_R = []
    total_F = []
    total_ckpt = []
    for train_index in range(opt.averaged_times):
        best_precision, best_recall, best_f1, ckpt = main(train_index, opt)
        total_P.append(best_precision)
        total_R.append(best_recall)
        total_F.append(best_f1)
        total_ckpt.append(ckpt)
    
    print("total_P", total_P)
    print("total_R", total_R)
    print("total_F", total_F)
    
    with open('result.txt', 'a', encoding='utf-8') as fp:
        print('writing result to result.txt...')
        fp.write( "ID: {:<2s}, total_F: {:<5s}  total_F_std: {:<5s}  total_P: {:<5s}  total_R: {:<5s}      N: {:<2s}  K: {:<2s} LR: {:<4s}  batch_size: {:<2s}  mode: {:<5s}  task: {:<8s} support_num {:<2s} pretrain_ckpt: {:<17s}  total_ckpt: {:<20s}".format(
                    str(opt.ID), 
                    str(round(np.mean(total_F), 2)), 
                    str(round(np.std(total_F), 2)),
                    str(round(np.mean(total_P), 2)), 
                    str(round(np.mean(total_R), 2)),  
                    str(opt.N), str(opt.K), str(opt.lr), str(opt.batch_size), 
                    str(opt.mode), str(opt.task), str(opt.support_num), 
                    str(opt.pretrain_ckpt), str(total_ckpt)) )
        
        fp.write('\n')
        

            
