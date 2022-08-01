import os
import sklearn.metrics
from tqdm import tqdm
import numpy as np
import sys
import time
import copy
import json
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam
from transformers import AdamW, get_constant_schedule_with_warmup
from transformers import BertConfig
from model.word_encoder import BERTWordEncoder
from model.copner import COPNER
from model.viterbi import ViterbiDecoder
import util.data_loader as data_loader
import shutil
from apex import amp

def get_abstract_transitions(train_fname, args, tokenizer):
    """
    Compute abstract transitions on the training dataset for StructShot
    """
    samples = data_loader.FewShotNERDatasetforInLabelSpaceTrain(train_fname, tokenizer, 1, 1, 1, 1, word_map=args.train_word_map, args=args).samples
    
    tag_lists = [sample.tags for sample in samples]

    s_o, s_i = 0., 0.
    o_o, o_i = 0., 0.
    i_o, i_i, x_y = 0., 0., 0.
    for tags in tag_lists:
        if tags[0] == 'O': s_o += 1
        else: s_i += 1
        for i in range(len(tags)-1):
            p, n = tags[i], tags[i+1]
            if p == 'O':
                if n == 'O': o_o += 1
                else: o_i += 1
            else:
                if n == 'O':
                    i_o += 1
                elif p != n:
                    x_y += 1
                else:
                    i_i += 1

    trans = []
    trans.append(s_o / (s_o + s_i))
    trans.append(s_i / (s_o + s_i))
    trans.append(o_o / (o_o + o_i))
    trans.append(o_i / (o_o + o_i))
    trans.append(i_o / (i_o + i_i + x_y))
    trans.append(i_i / (i_o + i_i + x_y))
    trans.append(x_y / (i_o + i_i + x_y))
    return trans


class FewShotNERFramework:

    def __init__(self, train_index, args, train_data_loader, val_data_loader, test_data_loader, tokenizer):
        self.args = args
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.train_index = train_index
        self.tokenizer = tokenizer
        sys_path = str(sys.argv[1:])[1:-1].replace("'", "").replace("--", "").replace(",", "_").replace(" ", "") if len(str(sys.argv[1:]))>2 else str(sys.argv[1:])
        self.res_path ="./results/output/"+sys_path[:80].replace("/","_")
        self.record_path = self.res_path+"/record.txt"
        self.viterbi = args.struct
        
        if self.viterbi:
            abstract_transitions = get_abstract_transitions(args.train, args, tokenizer)
            self.viterbi_decoder = ViterbiDecoder(args.N+2, abstract_transitions, args.t2c_tau)
        
        # if ("description" == args.prompt_semantic) or ("mismatch" == args.prompt_semantic) or ("out_desc" == args.prompt_semantic):
        if ("description" == args.prompt_semantic) or ("out_desc" == args.prompt_semantic):
            self.train_desc_input = self.args.train_desc_input.cuda()
            self.dev_desc_input = self.args.dev_desc_input.cuda()
            self.test_desc_input = self.args.test_desc_input.cuda()
            
        if "synonyms" == args.prompt_semantic:    
            self.train_syn = self.args.train_syn
            self.dev_syn = self.args.dev_syn
            self.test_syn = self.args.test_syn
        
        if ("only_word" == args.prompt_semantic) or ("random" == args.prompt_semantic) or ("out_word" == args.prompt_semantic) or ("mismatch" == args.prompt_semantic) :    
            self.train_word_map = self.args.train_word_map
            self.dev_word_map = self.args.dev_word_map 
            self.test_word_map = self.args.test_word_map
            
        if self.train_index=="temp":
            self.res_path ="./results/output/temp"
        elif self.train_index==0:
            if os.path.exists(self.res_path):
                shutil.rmtree(self.res_path)
            os.mkdir(self.res_path)
            with open(os.path.join(self.res_path, "config.json"), "w") as f:
                tmp_dic = copy.deepcopy(vars(args))
                del tmp_dic["tokenizer"]
                del tmp_dic["train_word_map"]
                del tmp_dic["dev_word_map"]
                del tmp_dic["test_word_map"]
                
                if "train_desc_input" in  tmp_dic.keys():
                    del tmp_dic["train_desc_input"] 
                if "dev_desc_input" in  tmp_dic.keys():
                    del tmp_dic["dev_desc_input"] 
                if "test_desc_input" in  tmp_dic.keys():
                    del tmp_dic["test_desc_input"] 
                
                if "train_syn" in  tmp_dic.keys():
                    del tmp_dic["train_syn"] 
                if "dev_syn" in  tmp_dic.keys():
                    del tmp_dic["dev_syn"] 
                if "test_syn" in  tmp_dic.keys():
                    del tmp_dic["test_syn"] 
                json.dump(tmp_dic, f, indent=2)
                
    def init_model(self, model, use_sgd_for_lm, load_ckpt, fp16, learning_rate):
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_lm:
            optimizer = torch.optim.SGD(
                parameters_to_optimize, lr=learning_rate)
        else:
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate)
        # self.scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=200)

        # load model
        if load_ckpt is not None:
            if os.path.isfile(load_ckpt):
                state_dict = torch.load(load_ckpt)
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        print('ignore {}'.format(name))
                        continue
                    print('load {} from {}'.format(name, load_ckpt))
                    own_state[name].copy_(param)
                print("Successfully loaded checkpoint '%s'" % load_ckpt)
            else:
                raise Exception("No checkpoint found at '%s'" % load_ckpt)

        if fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.record_path = self.record_path
        return model, optimizer

    def __get_emmissions__(self, logits, tags_list):
        # split [num_of_query_tokens, num_class] into [[num_of_token_in_sent, num_class], ...]
        emmissions = []
        current_idx = 0
        for tags in tags_list:
            emmissions.append(logits[current_idx:current_idx+len(tags)])
            current_idx += len(tags)
        # assert current_idx == logits.size()[0]
        return emmissions

    def viterbi_decode(self, logits, query_tags):
        emissions_list = self.__get_emmissions__(logits, query_tags)
        pred = []
        for i in range(len(query_tags)):
            # sent_scores = emissions_list[i].cpu()
            sent_scores = emissions_list[i]
            sent_len, n_label = sent_scores.shape
            sent_probs = F.softmax(sent_scores, dim=1)
            start_probs = torch.zeros(sent_len).cuda() + 1e-6
            sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
            feats = self.viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label+1))
            vit_labels = self.viterbi_decoder.viterbi(feats)
            pred.append(vit_labels.view(sent_len)-1)
            # vit_labels = vit_labels.view(sent_len)
            # vit_labels = vit_labels.detach().cpu().numpy().tolist()
            # for label in vit_labels:
            #     pred.append(label-1)
        # return torch.tensor(pred)
        return torch.cat(pred)
    
    def get_enhanced_embedding(self, model, target_classes, sent_num, data_flag):
        if data_flag == "train" and self.args.prompt_semantic == "description":
            enhanced_input = self.train_desc_input
        elif data_flag == "train" and self.args.prompt_semantic == "synonyms":
            enhanced_input = self.train_syn
        elif data_flag == "train" and self.args.prompt_semantic == "random":
            enhanced_input = self.train_word_map
        elif data_flag == "train" and self.args.prompt_semantic == "only_word":
            enhanced_input = self.train_word_map
        elif data_flag == "train" and self.args.prompt_semantic == "mismatch":
            # enhanced_input = self.train_desc_input
            enhanced_input = self.train_word_map
        elif data_flag == "train" and self.args.prompt_semantic == "out_desc":
            enhanced_input = self.train_desc_input
        elif data_flag == "train" and self.args.prompt_semantic == "out_word":
            enhanced_input = self.train_word_map
            
        elif data_flag == "dev" and self.args.prompt_semantic == "description":
            enhanced_input = self.dev_desc_input
        elif data_flag == "dev" and self.args.prompt_semantic == "synonyms":
            enhanced_input = self.dev_syn
        elif data_flag == "dev" and self.args.prompt_semantic == "random":
            enhanced_input = self.dev_word_map
        elif data_flag == "dev" and self.args.prompt_semantic == "only_word":
            enhanced_input = self.dev_word_map
        elif data_flag == "dev" and self.args.prompt_semantic == "mismatch":
            # enhanced_input = self.dev_desc_input
            enhanced_input = self.dev_word_map
        elif data_flag == "dev" and self.args.prompt_semantic == "out_desc":
            enhanced_input = self.dev_desc_input
        elif data_flag == "dev" and self.args.prompt_semantic == "out_word":
            enhanced_input = self.dev_word_map
            
        elif data_flag == "test" and self.args.prompt_semantic == "description":
            enhanced_input = self.test_desc_input
        elif data_flag == "test" and self.args.prompt_semantic == "synonyms":
            enhanced_input = self.test_syn
        elif data_flag == "test" and self.args.prompt_semantic == "random":
            enhanced_input = self.test_word_map
        elif data_flag == "test" and self.args.prompt_semantic == "only_word":
            enhanced_input = self.test_word_map
        elif data_flag == "test" and self.args.prompt_semantic == "mismatch":
            # enhanced_input = self.test_desc_input
            enhanced_input = self.test_word_map
        elif data_flag == "test" and self.args.prompt_semantic == "out_desc":
            enhanced_input = self.test_desc_input
        elif data_flag == "test" and self.args.prompt_semantic == "out_word":
            enhanced_input = self.test_word_map
        else:
            enhanced_input = None
            
        # if (self.args.prompt_semantic == "description") or (self.args.prompt_semantic == "mismatch") or (self.args.prompt_semantic == "out_desc"):
        if (self.args.prompt_semantic == "description")  or (self.args.prompt_semantic == "out_desc"):
            enhanced_embedding, _, _ = model(enhanced_input, only_descr=True, train_flag="pass")
        elif self.args.prompt_semantic == "synonyms" :
            enhanced_embedding = model.get_syn_embedding(enhanced_input, sent_num, target_classes, data_flag)
        elif (self.args.prompt_semantic == "only_word") or (self.args.prompt_semantic == "out_word") or (self.args.prompt_semantic == "mismatch") or (self.args.prompt_semantic == "random"):
            enhanced_embedding = model.get_only_word_embedding(enhanced_input, sent_num, target_classes, self.tokenizer, data_flag)
        elif self.args.prompt_semantic == "continual" :
            enhanced_embedding = ""
        else: 
            enhanced_embedding = None
            
        return enhanced_embedding
    
    def train(self, model, model_name,  
              learning_rate=1e-1, train_iter=30000, val_iter=1000, val_interval=2000,
              adapt_steps=5, load_ckpt=None, save_ckpt=None, warmup_step=300,
              grad_iter=1, fp16=False, use_sgd_for_lm=False):

        model, optimizer = self.init_model(model, use_sgd_for_lm, load_ckpt, fp16, learning_rate)
        model.train()
        
        # Training
        train_loss = 0
        train_c2t_loss = 0
        train_c2c_loss = 0
        best_iter = 0
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        iter_sample = 0
        pred_cnt = 1e-9
        label_cnt = 1e-9
        correct_cnt = 0

        print("Start training...")
        with tqdm(self.train_data_loader, total=train_iter, disable=False, desc="Training", ncols=170) as tbar:
            for it, (support, query) in enumerate(tbar):
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'target_classes' and k != 'sentence_num' and k != 'class_indexs' and \
                            k != 'labels' and k != 'label2tag':
                                support[k] = support[k].cuda()
                                # query[k] = query[k].cuda()
                    label = torch.cat(support['labels'], 0)
                    label = label.cuda()
                
                enhanced_embedding = self.get_enhanced_embedding(model, support['target_classes'], support['sentence_num'], data_flag="train")
                logits, pred, loss, token2class_loss, class2class_loss, token2token_loss = model(support['inputs'], tagging_labels=support['tagging_labels'], 
                                            index_labels=support['index_labels'], target_classes=support['target_classes'], sentence_num=support['sentence_num'], 
                                           prompt_tags_index=support['prompt_tags_index'], prompt_pos=support['prompt_pos'], enhanced_embedding=enhanced_embedding, 
                                           train_flag="train")

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                if (it % grad_iter == 0):
                    optimizer.step()
                    # self.scheduler.step()
                    optimizer.zero_grad()

                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                
                train_loss += loss.data.item()
                train_c2t_loss += token2class_loss.data.item()
                train_c2c_loss += class2class_loss.data.item()
                
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                iter_sample += 1
                train_precision = correct_cnt / pred_cnt *100
                train_recall = correct_cnt / label_cnt *100
                train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-9)  # 1e-9 for error'float division by zero'
                
                tbar.set_postfix_str("loss: {:2.6f} ({:2.6f})({:2.6f})({:2.6f}) | Correct:{}, P: {:3.3f}, R: {:3.3f}, F1: {:3.3f}"\
                                            .format(loss.data.item(), token2class_loss.item(), class2class_loss.item(), token2token_loss.item(),
                                                    correct_cnt, train_precision, train_recall, train_f1))
                
                if (it % val_interval == 0) and it>1:
                    if os.path.exists(self.record_path):
                        os.remove(self.record_path)
                        
                    val_precision, val_recall, val_f1, val_fp_error, val_fn_error, val_within_error, val_outer_error, val_loss, val_correct_cnt, span_correct_ratio \
                        = self.eval(model, val_iter, adapt_steps, word_map=self.args.dev_word_map)
                    
                    model.train()
                    if val_f1 >= best_f1:
                        print(f'Best checkpoint! Saving to: {save_ckpt}\n')
                        torch.save({'state_dict': model.state_dict()}, save_ckpt)
                        best_f1 = val_f1
                        best_iter = it
                        best_precision = val_precision
                        best_recall = val_recall
                
                    with open(self.res_path+"/"+str(self.train_index)+"_performanc.txt", "a") as f:
                        f.write("train_iter: "+str(it)+"\r\n")
                        f.write("train_loss: {:6.6f} ({:6.6f}, {:6.6f}) | Correct:{:5.1f}, P: {:2.2f}, R: {:2.2f}, F1: {:2.2f}"\
                                            .format(train_loss/val_interval, train_c2t_loss/val_interval, train_c2c_loss/val_interval, correct_cnt, train_precision, train_recall, train_f1)+"\r")
                        f.write("val_loss  : {:6.6f} ({:6.6f}, {:6.6f}) | Correct:{:5.1f}, P: {:2.2f}, R: {:2.2f}, F1: {:2.2f}"\
                                            .format(val_loss, val_loss, val_loss, val_correct_cnt, val_precision, val_recall, val_f1)+"\r")
                        f.write("\n\r")
                        f.write("val_fp_error  : {:5.2f} | val_fn_error:{:5.2f}, val_within_error: {:5.2f}, val_outer_error: {:5.2f}, span_correct_ratio: {:5.2f}"\
                                            .format(val_fp_error, val_fn_error, val_within_error, val_outer_error, span_correct_ratio)+"\r")
                        f.write("\n\r")
                        f.write("best epoch {0}, best_f1 {1:2.2f} ".format(best_iter, best_f1))
                        f.write("\r===========================================================")
                        f.write("\n")
                        f.write("\n")
                        
                    train_loss = 0 
                    train_c2t_loss = 0
                    train_c2c_loss = 0
        
                    iter_sample = 0.
                    pred_cnt = 1e-9
                    label_cnt = 1e-9
                    correct_cnt = 0
                
                if it >= min(10000, self.args.train_iter):
                    break

        print("\n####################\n")
        print("Finish training " + model_name)
        return best_precision, best_recall, best_f1

    def eval(self, model, eval_iter,  adapt_steps=5, ckpt=None, word_map=None): 
        
        if self.args.hight_source_flag:
            val_precision, val_recall, val_f1, val_fp_error, val_fn_error, val_within_error, val_outer_error, \
                val_total_loss, val_correct_cnt, val_span_correct_ratio = self.eval_with_high_source(model, eval_iter, adapt_steps, ckpt, word_map)
        else:
            val_precision, val_recall, val_f1, val_fp_error, val_fn_error, val_within_error, val_outer_error, \
                val_total_loss, val_correct_cnt, val_span_correct_ratio = self.eval_no_high_source(model, eval_iter, adapt_steps, ckpt, word_map)

        if self.args.only_test is True:
            with open(self.res_path+"/"+str(self.train_index)+"_performanc.txt", "a") as f:
                f.write("val_loss  : {:6.6f} | Correct:{:5.1f}, P: {:2.2f}, R: {:2.2f}, F1: {:2.2f}"\
                                    .format(val_total_loss,  val_correct_cnt, val_precision, val_recall, val_f1)+"\r")
                f.write("\n\r")
                f.write("val_fp_error  : {:5.2f} | val_fn_error:{:5.2f}, val_within_error: {:5.2f}, val_outer_error: {:5.2f}, span_correct_ratio: {:5.2f}"\
                                    .format(val_fp_error, val_fn_error, val_within_error, val_outer_error, val_span_correct_ratio)+"\r")
                f.write("\r===========================================================")
                f.write("\n")
                f.write("\n")
                
        return val_precision, val_recall, val_f1, val_fp_error, val_fn_error, val_within_error, val_outer_error, val_total_loss, val_correct_cnt, val_span_correct_ratio
    
    def eval_no_high_source(self, model, eval_iter,  adapt_steps=None, ckpt=None, word_map=None):
        print("")
        model.eval()

        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = torch.load(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        print('ignore {}'.format(name))
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        if word_map is None:
            word_map = self.args.dev_word_map

        total_loss = 0
        pred_cnt = 1e-9  # pred entity cnt
        label_cnt = 1e-9  # true label entity cnt
        correct_cnt = 0  # correct predicted entity cnt

        fp_cnt = 0  # misclassify O as I-
        fn_cnt = 0  # misclassify I- as O
        total_token_cnt = 0  # total token cnt
        within_cnt = 0  # span correct but of wrong fine-grained type
        outer_cnt = 0  # span correct but of wrong coarse-grained type
        total_correct_span_cnt = 0 # span correct
        total_span_cnt = 0 # span 

        eval_iter = min(eval_iter, len(eval_dataset))

        with tqdm(eval_dataset, total=eval_iter, disable=False, desc="Evaling", ncols=170) as tbar:
            for it, (_, query) in enumerate(tbar):
                if torch.cuda.is_available():
                    for k in query:
                        if k != 'target_classes' and k != 'sentence_num' and k != 'labels' and k != 'label2tag':
                            query[k] = query[k].cuda()
                    label = torch.cat(query['labels'], 0).cuda()

                with torch.no_grad():
                    enhanced_embedding = self.get_enhanced_embedding(model, query['target_classes'], query['sentence_num'], data_flag="dev")
                    logits, preds, loss, token2class_loss, class2class_loss, token2token_loss = model(query['inputs'], tagging_labels=query['tagging_labels'],  
                                                index_labels=query['index_labels'],  target_classes=query['target_classes'],  
                                                sentence_num=query['sentence_num'], true_labels=query['labels'], prompt_tags_index=query['prompt_tags_index'], 
                                                prompt_pos=query['prompt_pos'], enhanced_embedding=enhanced_embedding, train_flag="dev", record_path=self.record_path)
                    if self.viterbi:
                        preds = self.viterbi_decode(logits, query['labels'])

                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(preds, label)
                
                # fp, fn, token_cnt, within, outer, total_span, total_correct_span = model.error_analysis(pred, label, query)
                # fn_cnt += fn.data.item()
                # fp_cnt += fp.data.item()
                
                fp, fn, token_cnt, within, outer, total_span, total_correct_span = 1,1,1,1,1,1,1
                fn_cnt += 0
                fp_cnt += 0
                    
                total_loss +=loss
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                total_token_cnt += token_cnt
                outer_cnt += outer
                within_cnt += within
                total_span_cnt += total_span
                total_correct_span_cnt += total_correct_span

                val_precision = correct_cnt / pred_cnt  *100
                val_recall = correct_cnt / label_cnt  *100
                val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-9) 
                
                fp_error = fp_cnt / total_token_cnt
                fn_error = fn_cnt / total_token_cnt
                within_error = within_cnt / total_span_cnt
                outer_error = outer_cnt / total_span_cnt
                span_correct_ratio =  total_correct_span / total_span_cnt *100
                
                tbar.set_postfix_str("loss: {:2.6f} ({:2.6f})({:2.6f})({:2.6f}) | Correct:{}, P: {:3.3f}, R: {:3.3f}, F1: {:3.3f}"\
                                            .format(loss.data.item(), token2class_loss.item(), class2class_loss.item(), token2token_loss.item(), 
                                                    correct_cnt, val_precision, val_recall, val_f1))

                if it + 1 == eval_iter:
                    break

        precision = correct_cnt / pred_cnt *100
        recall = correct_cnt / label_cnt *100
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        print('[EVAL] f1: {:3.4f}, precision: {:3.4f}, recall: {:3.4f}\n'.format(
            f1, precision, recall))

        return precision, recall, f1, fp_error, fn_error, within_error, outer_error, total_loss/eval_iter, correct_cnt, span_correct_ratio
    
    def eval_with_high_source(self, model, eval_iter,  adapt_steps=None, ckpt=None, word_map=None): 
        print("")
        model.eval()
        
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = torch.load(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        print('ignore {}'.format(name))
                        continue
                    own_state[name].copy_(param)
                print("loading success !")
            eval_dataset = self.test_data_loader

        if word_map is None:
            word_map = self.args.dev_word_map

        total_loss = 0
        pred_cnt = 1e-9 # pred entity cnt
        label_cnt = 1e-9 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt
        
        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_correct_span_cnt = 0 # span correct
        total_span_cnt = 0 # span 

        config = BertConfig.from_pretrained(self.args.pretrain_ckpt)
        sentence_encoder = BERTWordEncoder(config, args=self.args)
        sentence_encoder.bert.resize_token_embeddings(len(self.args.tokenizer))
        eval_model = COPNER(sentence_encoder, self.args, word_map=word_map)

        parameters_to_optimize = list(eval_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(parameters_to_optimize, lr=self.args.adapt_lr)
        eval_iter = min(eval_iter, len(eval_dataset))
        
        with tqdm(eval_dataset, total=eval_iter, disable=False, desc="Evaling", ncols=170) as tbar:
            for it, (support, query) in enumerate(tbar):

                # copy params, Prevent updating the parameters of the train model during adapting
                old_state = model.state_dict()
                eval_state = eval_model.state_dict()

                for name, param in eval_state.items():
                    if name not in old_state:
                        print('not find {}'.format(name))

                for name, param in old_state.items():
                    if name not in eval_state:
                        print('ignore {}'.format(name))
                        continue
                    eval_state[name].copy_(param)

                if torch.cuda.is_available():
                    eval_model = eval_model.cuda()

                    for k in support:
                        if k != 'target_classes' and \
                            k != 'sentence_num' and \
                            k != 'labels' and \
                            k != 'label2tag':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    label = torch.cat(query['labels'], 0)
                    label = label.cuda()
                
                if (not self.args.zero_shot):
                    adapt_loss = self.adapt(eval_model, support, optimizer, adapt_steps)

                with torch.no_grad():
                    enhanced_embedding = self.get_enhanced_embedding(model, query['target_classes'], query['sentence_num'], data_flag="dev")
                    logits, pred, loss, token2class_loss, class2class_loss, token2token_loss = eval_model(query['inputs'], tagging_labels=query['tagging_labels'],  
                                                index_labels=query['index_labels'],  target_classes=query['target_classes'],  
                                                sentence_num=query['sentence_num'], true_labels=query['labels'], prompt_tags_index=query['prompt_tags_index'], 
                                                prompt_pos=query['prompt_pos'], enhanced_embedding=enhanced_embedding, train_flag="dev", record_path=self.record_path)
                    if self.viterbi:
                        pred = self.viterbi_decode(logits, query['labels'])

                    tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                    
                    # fp, fn, token_cnt, within, outer, total_span, total_correct_span = model.error_analysis(pred, label, query)
                    # fn_cnt += fn.data.item()
                    # fp_cnt += fp.data.item()
                    
                    fp, fn, token_cnt, within, outer, total_span, total_correct_span = 1,1,1,1,1,1,1
                    fn_cnt += 0
                    fp_cnt += 0
                
                    total_loss +=loss
                    pred_cnt += tmp_pred_cnt
                    label_cnt += tmp_label_cnt
                    correct_cnt += correct

                    total_token_cnt += token_cnt
                    outer_cnt += outer
                    within_cnt += within
                    total_span_cnt += total_span
                    total_correct_span_cnt += total_correct_span

                    val_precision = correct_cnt / pred_cnt  *100
                    val_recall = correct_cnt / label_cnt  *100
                    val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-9) 
                    fp_error = fp_cnt / total_token_cnt  *100
                    fn_error = fn_cnt / total_token_cnt  *100
                    within_error = within_cnt / total_span_cnt  *100
                    outer_error = outer_cnt / total_span_cnt  *100
                    span_correct_ratio =  total_correct_span / total_span_cnt *100
                    
                    tbar.set_postfix_str("adapt_loss: {:2.6f} loss: {:2.6f} ({:2.6f})({:2.6f})({:2.6f}) | Correct:{}, P: {:3.3f}, R: {:3.3f}, F1: {:3.3f}"\
                                                .format(adapt_loss.item(), loss.data.item(), token2class_loss.item(), class2class_loss.item(), token2token_loss.item(), 
                                                        correct_cnt, val_precision, val_recall, val_f1))
                    
                if it + 1 == eval_iter:
                    break
        
        precision = correct_cnt / pred_cnt *100
        recall = correct_cnt / label_cnt *100
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        print('[EVAL] f1: {:3.3f}, precision: {:3.3f}, recall: {:3.3f}\n'.format(f1, precision, recall))
        
        return precision, recall, f1, fp_error, fn_error, within_error, outer_error, total_loss/eval_iter, correct_cnt, span_correct_ratio
    
    def adapt(self, model, support, optimizer, adapt_steps=5):
        model.train()
        
        for i in range(adapt_steps):
            enhanced_embedding = self.get_enhanced_embedding(model, support['target_classes'], support['sentence_num'], data_flag="dev")
            _, pred, loss, _, _, _ = model(support['inputs'], tagging_labels=support['tagging_labels'],  index_labels=support['index_labels'], 
                                        target_classes=support['target_classes'],  sentence_num=support['sentence_num'], 
                                        prompt_tags_index=support['prompt_tags_index'], prompt_pos=support['prompt_pos'], 
                                        enhanced_embedding=enhanced_embedding, train_flag="dev")
            if i ==0:
                save_loss = loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if self.args.adapt_auto:
                if loss> save_loss :
                    break

        model.eval()
        return loss

