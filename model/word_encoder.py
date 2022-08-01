import copy
import math
import random
from torch._C import Size
from transformers.training_args import TrainingArguments
import warnings
from typing import Optional, OrderedDict, Tuple
import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertConfig, BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


logger = logging.getLogger(__name__)

def shift_tokens_right(input_ids: torch.Tensor, shift_step: int=1, pad_token_id: int=0):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, shift_step:] = input_ids[:, :-shift_step].clone()
    shifted_input_ids[:, :shift_step] = pad_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class BERTWordEncoder(BertPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    
    def __init__(self, config: BertConfig, args):
        super().__init__(config)
        self.config = config
        self.args = args
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.tokenizer = self.args.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]

        self.prompt = args.prompt

        if self.prompt == 0 and self.args.template:
            self.spell_length = sum(self.args.template)
            self.prompt_encoder = PromptEncoder(self.spell_length, len(self.tokenizer))

        self.init_weights()

        self.project_layer_0 = nn.Linear(768, args.projector0_dim)
        
    def synonyms_descri_embed_input(self, sentences, enhanced_embedding, prompt_tags_index, prompt_pos):
        """
            des_embedding = (78, 6, 768)
            syn_embedding = (79, 6, 768)
            
            """
        
        raw_embeds = self.bert.embeddings(sentences)  # (sent_total_num, sent_len, 768)
        des_embedding = enhanced_embedding[0][prompt_tags_index]
        syn_embedding = enhanced_embedding[1]
        enhanced_embedding = des_embedding + syn_embedding
        
         # prompt_embeding[:, 0, :] = prompt_embeding[:, 0, :] + torch.mean(prompt_embeding[:, 1:, :], dim=1)
        init_other = prompt_embeding[:, 0, :]
        new_O_list = []
        for index in range(prompt_embeding.size(0)):
            tmp = torch.linalg.qr(prompt_embeding[index].t().clone())[0].t()
            res = init_other[index].clone() - torch.sum( tmp * torch.sum(init_other[index][None, :].clone() * tmp, 1)[:, None], dim=0)
            new_O_list.append(res)
        prompt_embeding[:, 0, :] = torch.stack(new_O_list)
        
        num = prompt_pos.size(1)
        index_0 = [(i,)*num for i in range(raw_embeds.size(0))]
        raw_embeds[index_0, prompt_pos] = (raw_embeds[index_0, prompt_pos] + enhanced_embedding)/2
        # no
        # raw_embeds[index_0, prompt_pos] = enhanced_embedding
            
        return raw_embeds
      
    def synonyms_embed_input(self, sentences, syn_embedding, prompt_pos):
        """
            sentences = (sent_total_num, sent_len)
            raw_embeds = (sent_total_num, sent_len, 768)
            syn_embedding = (sent_total_num, 6, 768)
            prompt_embeding = (sent_total_num, 6, 768)
            """
        raw_embeds = self.bert.embeddings(sentences)  # (sent_total_num, sent_len, 768)
        
        num = prompt_pos.size(1)
        index_0 = [(i,)*num for i in range(raw_embeds.size(0))]
        raw_embeds[index_0, prompt_pos] = (raw_embeds[index_0, prompt_pos] + syn_embedding)/2
            
        return raw_embeds
    
    def descri_embed_input(self, sentences, desc_embedding, prompt_tags_index, prompt_pos):
        """
            sentences = (sent_total_num, sent_len)
            raw_embeds = (sent_total_num, sent_len, 768)
            desc_embedding = (all_class_num, 768)
            prompt_tags_index = (sent_total_num, 6)
            prompt_embeding = (sent_total_num, 6, 768) : (all_class, 768)[(sent_total_num, 6)]
            """
        if desc_embedding is None:
            raw_embeds = self.bert.embeddings(sentences)
        else:
            raw_embeds = self.bert.embeddings(sentences)
            prompt_embeding = desc_embedding[prompt_tags_index]
            
            # dynamatic Other
            if self.args.dynamic_Other:
                init_other = prompt_embeding[:, 0, :]
                new_O_list = []
                for index in range(prompt_embeding.size(0)):
                    tmp = torch.linalg.qr(prompt_embeding[index].t().clone())[0].t()
                    res = init_other[index].clone() - torch.sum( tmp * torch.sum(init_other[index][None, :].clone() * tmp, 1)[:, None], dim=0)
                    new_O_list.append(res)
                prompt_embeding[:, 0, :] = (init_other + torch.stack(new_O_list))/2
                
                num = prompt_pos.size(1)
                index_0 = [(i,)*num for i in range(raw_embeds.size(0))]
                raw_embeds[index_0, prompt_pos] = (raw_embeds[index_0, prompt_pos] + prompt_embeding)/2
            
        return raw_embeds

    def coninual_embed_input(self, sentences):
        sentences_for_embedding = sentences.clone()
        # sentences_for_embedding[(sentences == self.pseudo_token_id)] = self.tokenizer.unk_token_id

        raw_embeds = self.bert.embeddings(sentences_for_embedding)

        if self.prompt == 0 and self.pseudo_token_id is not None:
            bz = sentences.shape[0]
            blocked_indices = torch.nonzero(sentences == self.pseudo_token_id).reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
            replace_embeds = self.prompt_encoder(torch.LongTensor(list(range(self.spell_length))).to(sentences.device))

            for bidx in range(bz):
                for i in range(self.spell_length):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds
    
    def projector(self, raw_embeds, train_flag):
        if train_flag=="train":
            raw_embeds = F.relu(self.project_layer_0(raw_embeds))
        elif train_flag=="dev":
            if self.args.dev_project:
                raw_embeds = F.relu(self.project_layer_0(raw_embeds))
            else:
                raw_embeds = raw_embeds
        elif train_flag=="pass":
            raw_embeds = raw_embeds
        else:
            raise Exception("train_flag error !")
        return raw_embeds
        
    def forward(self, input_ids, enhanced_embedding=None, prompt_tags_index=None, prompt_pos=None, prompt_semantic=None, train_flag=None):

        # if (prompt_semantic == "description") or (prompt_semantic == "mismatch"):
        if (prompt_semantic == "description") :
            inputs_embeds = self.descri_embed_input(input_ids, enhanced_embedding, prompt_tags_index, prompt_pos)
        elif (prompt_semantic == "synonyms") or (prompt_semantic == "random") or (prompt_semantic == "only_word") or (prompt_semantic == "mismatch"):
            inputs_embeds = self.synonyms_embed_input(input_ids, enhanced_embedding, prompt_pos)
        elif prompt_semantic == "continual":
            inputs_embeds = self.coninual_embed_input(input_ids)
        elif (prompt_semantic == "out_desc") or  (prompt_semantic == "out_word"):
            inputs_embeds =  self.bert.embeddings(input_ids)
        else:
            raise Exception("prompt_semantic error !")
        
        attention_mask = (input_ids != self.pad_token_id).bool()
        
        if prompt_pos is not None:
            temp_bz = input_ids.size(0)
            temp_sent_len = input_ids.size(1)
            position_ids = torch.tensor(list(range(temp_sent_len))).repeat(temp_bz, 1).cuda()  
            msk = torch.arange(temp_sent_len).unsqueeze(0).expand(temp_bz, temp_sent_len).cuda()   >= prompt_pos[:, 0].unsqueeze(1) # (temp_bz, temp_sent_len) >= (temp_bz, 1)
            position_ids[msk] = prompt_pos[:, 0].unsqueeze(1).expand(temp_bz, temp_sent_len)[msk]
        else:
            position_ids = None
            
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = self.cls(outputs[0])
        hidden_state = outputs.last_hidden_state
        
        if self.args.if_projector:
            hidden_state = self.projector(hidden_state, train_flag)
        
        return logits, hidden_state
    