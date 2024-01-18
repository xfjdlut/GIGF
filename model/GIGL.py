# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from graph.simpleHGN import SimpleHGN
from graph.type_entity_graph import TypeEntityGraph
from model.GAT import GraphAttenTransformer
from model.Decoder import Decoder

from transformers import (
    BertModel,
    BertTokenizer
)


class GIGL(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.pad_token_id = args.pad_token_id
        self.bos_token_id = args.bos_token_id
        self.eos_token_id = args.eos_token_id
        self.kg_encoder = GraphAttenTransformer.from_pretrained(args.bert_dir)
        self.kg_encoder.resize_token_embeddings(self.vocab_size)
        self.conv_encoder = BertModel.from_pretrained(args.bert_dir)
        self.conv_encoder.resize_token_embeddings(self.vocab_size)
        self.user_profile_encoder = BertModel.from_pretrained(args.bert_dir)
        self.user_profile_encoder.resize_token_embeddings(self.vocab_size)

        self.Decoder = Decoder(args=args)

        self.h_g = TypeEntityGraph(args.type_dic, args.entity_dic, args.train_data, args.dev_data,args.kc_balance).fusion_graph()
        self.graph_embedding = nn.Embedding(self.h_g[-2] + self.h_g[-1], args.embed_dim)
        self.hgn = SimpleHGN(args, self.h_g[0]['adj'])

        self.type_vocab = nn.Linear(args.embed_dim, self.h_g[-2], bias=True)
        self.entity_vocab = nn.Linear(args.embed_dim, self.h_g[-1], bias=True)

    def _init_all_weights(self):
        self.type_vocab.weight.data.normal_(mean=0.0, std=self.args.init_std)
        if self.type_vocab.bias is not None:
            self.type_vocab.bias.data.zero_()

        self.entity_vocab.weight.data.normal_(mean=0.0, std=self.args.init_std)
        if self.entity_vocab.bias is not None:
            self.entity_vocab.bias.data.zero_()

        # nn.init.xavier_uniform_(self.graph_embedding.weight)

    def forward(self, batch, is_test=False):
        """model training"""
        up_ids, up_segs, up_poss, up_mask = batch["user_profile"]
        kg_ids, kg_segs, kg_poss, kg_hops, kg_mask = batch["knowledge"]
        hs_ids, hs_segs, hs_poss, hs_mask = batch["conversation"]
        
        tg_type_ids, tg_entity_ids = batch["target"]
        cur_type_ids, cur_entity_ids = batch["current"]
        pre_type_ids, pre_entity_ids = batch["predict"]

        if(not is_test):
            self.hgn(self.graph_embedding.weight)

        cur_type_embedding =self.graph_embedding(cur_type_ids)
        cur_entity_embedding = self.graph_embedding(cur_entity_ids+15)#15是type的个数，根据type的个数修改

        kg_bert_output = self.kg_encoder(
            input_ids=kg_ids, 
            attention_mask=kg_mask, 
            token_type_ids=kg_segs, 
            position_ids=kg_poss, 
            hops_ids=kg_hops)
        kg_output=kg_bert_output[0]
        kg_gate_input=kg_bert_output[1]

        conv_bert_output = self.conv_encoder(
            input_ids=hs_ids, 
            attention_mask=hs_mask, 
            token_type_ids=hs_segs, 
            position_ids=hs_poss)
        conv_output=conv_bert_output[0]
        conv_gru_input=conv_bert_output[1]

        up_bert_output = self.user_profile_encoder(
            input_ids=up_ids,
            attention_mask=up_mask,
            token_type_ids=up_segs,
            position_ids=up_poss)
        up_output=up_bert_output[0]
        up_gate_input=up_bert_output[1]

        decoder_output = self.Decoder(
            cur_type_embedding,
            cur_entity_embedding,
            kg_gate_input,
            up_gate_input,
            conv_gru_input,
            kg_encoder_hidden_states=kg_output,
            kg_encoder_attention_mask=kg_mask,
            up_encoder_hidden_states=up_output,
            up_encoder_attention_mask=up_mask,
            hs_encoder_hidden_states=conv_output,
            hs_encoder_attention_mask=hs_mask,
        )
        # 预测
        type_logits=self.type_vocab(decoder_output[0])
        entity_logits=self.entity_vocab(decoder_output[1])

        if is_test:
            type_pred = torch.softmax(type_logits, -1)
            _, type_pred_y = type_pred.max(-1)
            entity_pred = torch.softmax(entity_logits, -1)
            _, entity_pred_y = entity_pred.max(-1)
            output = {
                "type_pred":type_pred_y,
                "entity_pred":entity_pred_y
            }
        else:
            loss_fct = CrossEntropyLoss()
            type_loss=loss_fct(type_logits, pre_type_ids.view(-1))
            entity_loss=loss_fct(entity_logits, pre_entity_ids.view(-1))
            loss_all = type_loss + entity_loss

            type_pred = torch.softmax(type_logits, -1)
            _, type_pred_y = type_pred.max(-1)
            type_acc = (torch.eq(type_pred_y, pre_type_ids).float()).sum().item()
            entity_pred = torch.softmax(entity_logits, -1)
            _, entity_pred_y = entity_pred.max(-1)
            entity_acc = (torch.eq(entity_pred_y, pre_entity_ids).float()).sum().item()
            output = {
                "loss":loss_all,
                "type_acc": type_acc,
                "entity_acc": entity_acc
            }
        return output

    def generate(self, inputs):
        pre_type_ids, pre_entity_ids = inputs["predict"]
        model_out = self(inputs, is_test=True)
        return pre_type_ids,pre_entity_ids,model_out["type_pred"],model_out["entity_pred"]
