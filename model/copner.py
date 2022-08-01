import os
import sys
sys.path.append('..')
import torch.nn.functional as F
from torch import tensor
import json
from collections import OrderedDict
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers import BartConfig
from util.InforNCE import InfoNCE
from .base import FewShotNERModel

class COPNER(FewShotNERModel):
    
    def __init__(self, sentence_encoder, args, word_map):
        FewShotNERModel.__init__(self, sentence_encoder, ignore_index=args.ignore_index)
        self.args = args
        self.tokenizer = args.tokenizer
        self.dropout = nn.Dropout(p=args.drop_ratio)
        self.type_t2c_loss_fct = CrossEntropyLoss(ignore_index=-2, reduction='none')
        self.loss_fct = CrossEntropyLoss()
        self.method = args.dis_method
        self.t2c_tau = args.t2c_tau
        self.c2c_tau = args.c2c_tau
        self.t2t_tau = args.t2t_tau        
        self.my_infoNCE = InfoNCE(temperature=args.t2c_tau)
        
        
        
        if self.args.prompt_semantic=="out_desc" or self.args.prompt_semantic== "out_word":
            self.out_flag = True
        else:
            self.out_flag = False
            
        self.type2word = word_map
        self.word2type = OrderedDict()    
        for key, value in self.type2word.items():
            self.word2type[value[-1]] = key
        
        i = 1
        self.tmp_logit_file = './test_logit_conll_0.json'
        while os.path.exists(self.tmp_logit_file):
            self.tmp_logit_file = './test_logit_conll_{}.json'.format(i)
            i += 1
            
        i = 1    
        self.tmp_dis_file = './test_dis_conll_0.json'
        while os.path.exists(self.tmp_dis_file):
            self.tmp_dis_file = './test_dis_conll_{}.json'.format(i)
            i += 1
                                
    def __dist__(self, x, y, dim, tau=0.05, method=None): 
        x = x.unsqueeze(0)
        y = y.unsqueeze(1)
        if method is None:
            method = self.method
            
        if method == 'dot':
            sim = (x * y).sum(dim)/ tau
        elif method == 'euclidean':
            sim = -(torch.pow(x - y, 2)).sum(dim) / tau
        elif method == 'cosine':
            sim = torch.abs(F.cosine_similarity(x, y, dim=dim) / tau)
        elif method == 'KL':
            kl_mean_1 = F.kl_div(F.log_softmax(x, dim=-1), F.softmax(y, dim=-1), reduction='sum')
            kl_mean_2 = F.kl_div(F.log_softmax(y, dim=-1), F.softmax(x, dim=-1), reduction='sum')

            sim = (kl_mean_1 + kl_mean_2)/2
        return sim

    def get_out_contrastive_rep(self, train_flag, prompt_tags_index, prompt_semantic, hidden_states, enhanced_embedding, inputs, tagging_label, target_classes, true_labels=None):

        hidden_states = self.dropout(hidden_states)
        valid_hs = hidden_states[:, :tagging_label.size(-1), :]    # omit type words hidden states
        if train_flag=="train" :
            tmp_map = list(self.args.train_word_map.values())
        elif train_flag=="dev" :
            tmp_map = list(self.args.dev_word_map.values())
        elif  train_flag=="test" :
            tmp_map = list(self.args.test_word_map.values())
        else:
            raise Exception("error!")
        
        class_indexs = [tmp_map.index([tclass]) for tclass in target_classes]
            
        if prompt_semantic=="out_desc":
            class_rep = enhanced_embedding[class_indexs]
        elif prompt_semantic=="out_word":
            class_rep = enhanced_embedding[0,] 
        else:
            raise Exception("error") 
        
        token_rep = valid_hs[tagging_label != self.tokenizer.pad_token_id, :].view(-1, valid_hs.size(-1)) # (190 768)
        return class_rep, token_rep, torch.tensor(class_indexs).cuda() 
    
    def get_contrastive_rep(self, hidden_states, inputs, tagging_label, target_classes, true_labels=None):
        
        class_indexs = [self.tokenizer.get_vocab()[tclass] for tclass in target_classes]
        hidden_states = self.dropout(hidden_states)
        valid_hs = hidden_states[:, :tagging_label.size(-1), :]    # omit type words hidden states
        
        # recover N-WAY order
        class_rep = []
        for iclass in class_indexs:
            temp_class_rep = hidden_states[inputs.eq(iclass), :].view(-1, hidden_states.size(-1))
            class_rep.append(torch.mean(temp_class_rep, 0))
        
        class_rep = torch.stack(class_rep)    # (6,768)
        token_rep = valid_hs[tagging_label != self.tokenizer.pad_token_id, :].view(-1, valid_hs.size(-1))
        
        return class_rep, token_rep, torch.tensor(class_indexs).cuda() 
    
    def get_syn_embedding(self, enhanced_input, sent_num, target_classes, data_flag):
        if data_flag=="train":
            tmp_map = self.args.train_word_map
        elif data_flag=="dev":
            tmp_map = self.args.dev_word_map
        elif data_flag=="test":
            tmp_map = self.args.test_word_map
        else:
            raise Exception("error") 
        
        new_embedding_map = {}
        for k, v in enhanced_input.items():
            new_embedding_map[tmp_map[k][0]] = self.sentence_encoder.bert.embeddings(torch.tensor(v).unsqueeze(0).cuda())
        
        syn_embedding = []
        for index, N_way in enumerate(target_classes):
            syn_embedding.extend( [torch.stack([torch.mean(new_embedding_map[one_way].squeeze(0), dim=0) for one_way in N_way])] * sent_num[index] )
        return torch.stack(syn_embedding, dim=0)
    
    def get_only_word_embedding(self, enhanced_input, sent_num, target_classes, tokenizer, data_flag):
        if data_flag=="train":
            tmp_map = self.args.train_word_map
        elif data_flag=="dev":
            tmp_map = self.args.dev_word_map
        elif data_flag=="test":
            tmp_map = self.args.test_word_map
        else:
            raise Exception("error") 
        
        new_embedding_map = {}
        for k, v in enhanced_input.items():
            new_embedding_map[tmp_map[k][0]] = self.sentence_encoder.bert.embeddings(torch.tensor(tokenizer.convert_tokens_to_ids(v)).unsqueeze(0).cuda())
        
        syn_embedding = []
        for index, N_way in enumerate(target_classes):
            syn_embedding.extend( [torch.stack([torch.mean(new_embedding_map[one_way].squeeze(0), dim=0) for one_way in N_way])] * sent_num[index] )
        return torch.stack(syn_embedding, dim=0)
    
    def out_forward(self, input_ids, tagging_labels=None,  index_labels=None, target_classes=None, 
                sentence_num = None, true_labels=None, only_descr=False, prompt_tags_index=None, 
                prompt_pos=None, enhanced_embedding=None, prompt_semantic=None, train_flag=None,  record_path=None):
        
        if prompt_semantic is None:
            prompt_semantic = self.args.prompt_semantic
        
        if only_descr:
            lm_logits, hidden_states = self.sentence_encoder(input_ids, None, prompt_tags_index, prompt_pos, prompt_semantic, train_flag) 
            return hidden_states[:, 0, :], "", ""
        
        else:
            
            for index, i in enumerate(prompt_pos[:,0]):
                input_ids[index, i:] = 0
                
            lm_logits, hidden_states = self.sentence_encoder(input_ids, None, prompt_tags_index, 
                                                prompt_pos, prompt_semantic, train_flag) # logits, (encoder_hs, decoder_hs)
            
            span_t2c_loss = None
            type_t2c_loss = None
            
            t2c_loss = None
            c2c_loss = None
            t2t_loss = None
            
            logits_list = []
            total_pred = []
            
            class_rep_list = []
            token_rep_list = []
            class_indexs_list = []
            current_num = 0
            for i, num in enumerate(sentence_num): # each  N-way-k-shot
                current_hs = hidden_states[current_num: current_num+num]
                current_input_ids = input_ids[current_num: current_num+num]
                current_tagging_labels = tagging_labels[current_num: current_num+num]
                current_index_labels = index_labels[current_num: current_num+num][current_tagging_labels != self.tokenizer.pad_token_id].view(-1)
                current_span_labels = torch.where(current_index_labels==0, 0, 1)
                current_target_classes = target_classes[i]
                current_num += num
                
                if true_labels is not None:
                    # true label is only for auxiliary experiment
                    current_true_labels = true_labels[current_num: current_num+num]
                
                class_rep, token_rep, class_indexs = self.get_out_contrastive_rep(train_flag, prompt_tags_index, prompt_semantic, current_hs, enhanced_embedding, current_input_ids, current_tagging_labels, current_target_classes,
                                                                current_true_labels if true_labels is not None else None) # no pad hidden state
                       
                class_rep_list.append(class_rep[1:])  
                class_indexs_list.extend(class_indexs[1:]) 
                token_rep_list.append(token_rep)
                    
                t2c_logits = self.__dist__(class_rep, token_rep, -1, tau=self.t2c_tau).view(-1, len(current_target_classes))  
                temp_t2c_loss = self.loss_fct(t2c_logits,  current_index_labels)
                t2c_loss = temp_t2c_loss if t2c_loss is None else  (temp_t2c_loss + t2c_loss)
                
                t2c_current_logits= [F.softmax(t2c_logits, -1)]
                t2c_current_logits = torch.sum(torch.stack(t2c_current_logits, -1), -1)
                t2c_current_logits = t2c_current_logits.view(-1, t2c_current_logits.size(-1))
                pred =  torch.argmax(t2c_current_logits, -1)
                total_pred.append(pred)   
                logits_list.append(t2c_current_logits)   
                
            
                if self.args.t2t_tau != 0:
                    t2t_logits = self.__dist__(token_rep.unsqueeze(0), token_rep.unsqueeze(1), -1, tau= self.t2t_tau).squeeze()
                    t2t_logits = t2t_logits * (1- torch.eye(t2t_logits.size(0), t2t_logits.size(1))).cuda()
                    temp_t2t_loss = -torch.mean(torch.mean(t2t_logits, dim=0), dim=0) 
                    t2t_loss = temp_t2t_loss if t2t_loss is None else  (temp_t2t_loss + t2t_loss)

            if self.args.c2c_tau != 0:
                c2c_mask = torch.tensor(class_indexs_list).cuda()  
                c2c_mask = (c2c_mask.unsqueeze(0) != c2c_mask.unsqueeze(1))
                class_rep_tensor = torch.cat(class_rep_list, dim=0) #（bz*class_num, 768）
                c2c_logits = self.__dist__(class_rep_tensor, class_rep_tensor, -1, tau = self.c2c_tau) # (bz*sent_num, bz*sent_num) 
                c2c_logits = c2c_logits*c2c_mask   
                c2c_loss = -torch.mean(torch.mean(c2c_logits, dim=0), dim=0) 
                
            # loss need normalzation
            # t2c_loss = self.args.span_weight * span_t2c_loss + self.args.type_weight * type_t2c_loss
            t2c_loss = t2c_loss / len(sentence_num)
            if self.args.t2t_tau != 0:
                t2t_loss = t2t_loss / len(sentence_num) 
            if self.args.c2c_tau != 0:
                c2c_loss = c2c_loss 
            
            preds = torch.cat(total_pred)
            logits = torch.cat(logits_list) 
            
            if self.args.t2t_tau == 0 and self.args.c2c_tau == 0:
                loss = t2c_loss 
                return logits, preds, loss, t2c_loss, t2c_loss, t2c_loss
            elif self.args.t2t_tau == 0 and self.args.c2c_tau != 0:
                loss = t2c_loss  + c2c_loss 
                return logits, preds, loss, t2c_loss, c2c_loss, c2c_loss
            elif self.args.t2t_tau != 0 and self.args.c2c_tau == 0:
                loss = t2c_loss  + t2t_loss
                return logits, preds, loss, t2c_loss, t2t_loss, t2t_loss
            elif self.args.t2t_tau != 0 and self.args.c2c_tau != 0:
                loss = t2c_loss  + c2c_loss + t2t_loss
                return logits, preds, loss, t2c_loss, c2c_loss, t2t_loss
            else:
                raise Exception("loss weight error!")
                
    def forward(self, input_ids, tagging_labels=None,  index_labels=None, target_classes=None, 
                sentence_num = None, true_labels=None, only_descr=False, prompt_tags_index=None, 
                prompt_pos=None, enhanced_embedding=None, prompt_semantic=None, train_flag=None,  record_path=None):
        
        if self.out_flag:
            return self.out_forward(input_ids, tagging_labels=tagging_labels,  index_labels=index_labels, target_classes=target_classes, 
                sentence_num = sentence_num, true_labels=true_labels, only_descr=only_descr, prompt_tags_index=prompt_tags_index, 
                prompt_pos=prompt_pos, enhanced_embedding=enhanced_embedding, prompt_semantic=prompt_semantic, train_flag=train_flag, 
                 record_path=record_path)
        else:
            if prompt_semantic is None:
                prompt_semantic = self.args.prompt_semantic
            
            lm_logits, hidden_states = self.sentence_encoder(input_ids, enhanced_embedding, prompt_tags_index, 
                                                    prompt_pos, prompt_semantic, train_flag) # logits, (encoder_hs, decoder_hs)
            
            if only_descr:
                return hidden_states[:, 0, :], "", ""
            
            else:
                span_t2c_loss = None
                type_t2c_loss = None
                
                t2c_loss = None
                c2c_loss = None
                t2t_loss = None
                
                logits_list = []
                total_pred = []
                
                class_rep_list = []
                token_rep_list = []
                class_indexs_list = []
                current_num = 0
                for i, num in enumerate(sentence_num): # each  N-way-k-shot
                    current_hs = hidden_states[current_num: current_num+num]
                    current_input_ids = input_ids[current_num: current_num+num]
                    current_tagging_labels = tagging_labels[current_num: current_num+num]
                    current_index_labels = index_labels[current_num: current_num+num][current_tagging_labels != self.tokenizer.pad_token_id].view(-1)
                    current_span_labels = torch.where(current_index_labels==0, 0, 1)
                    current_target_classes = target_classes[i]
                    current_num += num
                    
                    if true_labels is not None:
                        # true label is only for auxiliary experiment
                        current_true_labels = true_labels[current_num: current_num+num]
                    
                    class_rep, token_rep, class_indexs = self.get_contrastive_rep(current_hs, current_input_ids, current_tagging_labels, current_target_classes,
                                                                    current_true_labels if true_labels is not None else None) # no pad hidden state
                        
                    class_rep_list.append(class_rep[1:])  
                    class_indexs_list.extend(class_indexs[1:]) 
                    token_rep_list.append(token_rep)
                    
                    # if not self.training and true_labels is not None:
                    #     index_labels = []
                    #     for la in true_labels:
                    #         index_labels.extend(la.tolist())
                    #     target_types = ['O', 'location-GPE', 'location-bodiesofwater', 'location-island', 'location-mountain', 'location-park', 'location-road/railway/highway/transit', 'location-other']
                        
                    #     assert token_rep.size(0) == len(index_labels)
                    #     for i in range(len(index_labels)):
                    #         emd_dict = {'label': '', 'value': []}
                    #         label = int(index_labels[i])
                    #         if label > 0:
                    #             emd_dict['value'] = token_rep[i].cpu().numpy().tolist()
                    #             emd_dict['label'] = target_types[label]
                                
                    #             with open(self.tmp_logit_file, 'a', encoding='utf-8') as fp:
                    #                 json.dump(emd_dict, fp, ensure_ascii=False)
                    #                 fp.write('\n')
                    
                    t2c_logits = self.__dist__(class_rep, token_rep, -1, tau=self.t2c_tau).view(-1, len(current_target_classes))  # class_rep = (5, 768), token_rep=(255, 768)
             
                    # if not self.training and true_labels is not None:
                    #     index_labels = []
                    #     for la in true_labels:
                    #         index_labels.extend(la.tolist())
                        
                    #     assert t2c_logits.size(0) == len(index_labels)
                    #     for i in range(len(index_labels)):
                    #         emd_dict = {'true': 0, 'false': 0, 'ratio': 0, "min_neg": 0}
                    #         label = int(index_labels[i])
                        
                    #         if label > 0:
                    #             emd_dict['true'] = -(t2c_logits[i][label].tolist())
                    #             false_dis = torch.cat([t2c_logits[i][:label], t2c_logits[i][label+1:]], 0)
                                
                    #             emd_dict["min_neg"] = torch.min(abs(false_dis)).tolist()
                
                    #             false_dis = torch.sum(false_dis).tolist()
                    #             false_dis /= (t2c_logits.size(-1) - 1)
                    #             emd_dict['false'] = -false_dis
                    #             emd_dict['ratio'] = emd_dict['true'] / emd_dict['false']
                    #             with open(self.tmp_dis_file, 'a', encoding='utf-8') as fp:
                    #                 json.dump(emd_dict, fp, ensure_ascii=False)
                    #                 fp.write('\n')
                 
            
                    temp_t2c_loss = self.loss_fct(t2c_logits,  current_index_labels)
                    t2c_loss = temp_t2c_loss if t2c_loss is None else  (temp_t2c_loss + t2c_loss)
                    
                    t2c_current_logits= [F.softmax(t2c_logits, -1)]
                    t2c_current_logits = torch.sum(torch.stack(t2c_current_logits, -1), -1)
                    t2c_current_logits = t2c_current_logits.view(-1, t2c_current_logits.size(-1))
                    
                    pred =  torch.argmax(t2c_current_logits, -1)
                    total_pred.append(pred)   
                    logits_list.append(t2c_current_logits)   
                    
                    # if record_path is not None:
                    #     with open(record_path, "a") as f:
                    #         f.write("{:<22s}: {}\n".format("current_index_labels", str(current_index_labels.tolist())))
                    #         f.write("{:<22s}: {}\n".format("pred:", str(pred.tolist())))
                    #         f.write("\n\n")
                
                    if self.args.t2t_tau != 0:
                        t2t_logits = self.__dist__(token_rep.unsqueeze(0), token_rep.unsqueeze(1), -1, tau= self.t2t_tau).squeeze()
                        t2t_logits = t2t_logits * (1- torch.eye(t2t_logits.size(0), t2t_logits.size(1))).cuda()
                        temp_t2t_loss = -torch.mean(torch.mean(t2t_logits, dim=0), dim=0) 
                        t2t_loss = temp_t2t_loss if t2t_loss is None else  (temp_t2t_loss + t2t_loss)

                if self.args.c2c_tau != 0:
                    c2c_mask = torch.tensor(class_indexs_list).cuda()  
                    c2c_mask = (c2c_mask.unsqueeze(0) != c2c_mask.unsqueeze(1))
                    class_rep_tensor = torch.cat(class_rep_list, dim=0) #（bz*class_num, 768）
                    c2c_logits = self.__dist__(class_rep_tensor, class_rep_tensor, -1, tau = self.c2c_tau) # (bz*sent_num, bz*sent_num) 
                    c2c_logits = c2c_logits*c2c_mask   
                    c2c_loss = -torch.mean(torch.mean(c2c_logits, dim=0), dim=0) 
                    
                # loss need normalzation
                # t2c_loss = self.args.span_weight * span_t2c_loss + self.args.type_weight * type_t2c_loss
                t2c_loss = t2c_loss / len(sentence_num)
                if self.args.t2t_tau != 0:
                    t2t_loss = t2t_loss / len(sentence_num) 
                if self.args.c2c_tau != 0:
                    c2c_loss = c2c_loss 
                
                preds = torch.cat(total_pred)
                logits = torch.cat(logits_list) 
                
                if self.args.t2t_tau == 0 and self.args.c2c_tau == 0:
                    loss = t2c_loss 
                    return logits, preds, loss, t2c_loss, t2c_loss, t2c_loss
                elif self.args.t2t_tau == 0 and self.args.c2c_tau != 0:
                    loss = t2c_loss  + c2c_loss 
                    return logits, preds, loss, t2c_loss, c2c_loss, c2c_loss
                elif self.args.t2t_tau != 0 and self.args.c2c_tau == 0:
                    loss = t2c_loss  + t2t_loss
                    return logits, preds, loss, t2c_loss, t2t_loss, t2t_loss
                elif self.args.t2t_tau != 0 and self.args.c2c_tau != 0:
                    loss = t2c_loss  + c2c_loss + t2t_loss
                    return logits, preds, loss, t2c_loss, c2c_loss, t2t_loss
                else:
                    raise Exception("loss weight error!")