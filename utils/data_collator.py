# -*- coding: utf-8 -*-
import torch


def max_seq_length(list_l):
    return max(len(l) for l in list_l)

def pad_sequence(list_l, max_len, padding_value=0):
    assert len(list_l) <= max_len
    padding_l = [padding_value] * (max_len - len(list_l))
    padded_list = list_l + padding_l #这个是两个list的拼接
    return padded_list


class data_Collator(object):
    """
    Data collator for planning
    """
    def __init__(self, device, padding_idx=0):
        self.device = device
        self.padding_idx = padding_idx

    def list_to_tensor(self, list_l):
        max_len = max_seq_length(list_l)
        padded_lists = []  #通过padding把tensor对齐
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=self.padding_idx))
        input_tensor = torch.tensor(padded_lists, dtype=torch.long) #从list变成tensor
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor

    def varlist_to_tensor(self, list_vl):
        lens = []
        for list_l in list_vl:
            lens.append(max_seq_length(list_l))
        max_len = max(lens)
        
        padded_lists = []
        for list_seqs in list_vl:
            v_list = []
            for list_l in list_seqs:
                v_list.append(pad_sequence(list_l, max_len, padding_value=self.padding_idx))
            padded_lists.append(v_list)
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    # 返回的是一个attention_mask。对应padding是0，不是padding的是1
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = data_tensor.masked_fill(data_tensor == self.padding_idx, 0)
        attention_mask = attention_mask.masked_fill(attention_mask != self.padding_idx, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask
    
    def custom_collate(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        up_ids = []
        up_segs=[]
        up_poss=[]
        kg_ids, kg_segs, kg_poss, kg_hops = [], [], [], []
        hs_ids, hs_segs, hs_poss = [], [], []
        tg_type_ids, cur_type_ids, pre_type_ids = [], [], []
        tg_entity_ids, cur_entity_ids, pre_entity_ids = [], [], []
        for sample in mini_batch:
            up_ids.append(sample.user_profile_ids)
            up_segs.append(sample.user_profile_segs)
            up_poss.append(sample.user_profile_poss)
            kg_ids.append(sample.knowledge_ids)
            kg_segs.append(sample.knowledge_segs)
            kg_poss.append(sample.knowledge_poss)
            kg_hops.append(sample.knowledge_hops)
            hs_ids.append(sample.conversation_ids)
            hs_segs.append(sample.conversation_segs)
            hs_poss.append(sample.conversation_poss)
            tg_type_ids.append(sample.tar_type_ids)
            tg_entity_ids.append(sample.tar_entity_ids)
            cur_type_ids.append(sample.cur_type_ids)
            cur_entity_ids.append(sample.cur_entity_ids)
            pre_type_ids.append(sample.pre_type_ids)
            pre_entity_ids.append(sample.pre_entity_ids)

        batch_up_ids = self.list_to_tensor(up_ids)
        batch_up_segs = self.list_to_tensor(up_segs)
        batch_up_poss = self.list_to_tensor(up_poss)
        batch_up_masks = self.get_attention_mask(batch_up_ids)
        
        batch_kg_ids = self.list_to_tensor(kg_ids)
        batch_kg_segs = self.list_to_tensor(kg_segs)
        batch_kg_poss = self.list_to_tensor(kg_poss)
        batch_kg_hops = self.list_to_tensor(kg_hops)
        batch_kg_masks = self.get_attention_mask(batch_kg_ids)

        batch_hs_ids = self.list_to_tensor(hs_ids)
        batch_hs_segs = self.list_to_tensor(hs_segs)
        batch_hs_poss = self.list_to_tensor(hs_poss)
        batch_hs_masks = self.get_attention_mask(batch_hs_ids)

        batch_tg_type_ids = torch.tensor(tg_type_ids).to(self.device).contiguous()
        batch_tg_entity_ids = torch.tensor(tg_entity_ids).to(self.device).contiguous()
        batch_cur_type_ids = torch.tensor(cur_type_ids).to(self.device).contiguous()
        batch_cur_entity_ids = torch.tensor(cur_entity_ids).to(self.device).contiguous()
        batch_pre_type_ids = torch.tensor(pre_type_ids).to(self.device).contiguous()
        batch_pre_entity_ids = torch.tensor(pre_entity_ids).to(self.device).contiguous()

        # 通过转换得到所有数据的mask矩阵，其实就是padding补齐后mask,对于label，没有需要mask的，所有的type和entity都是字典下标
        collated_batch = {
            "user_profile": [batch_up_ids,batch_up_segs,batch_up_poss, batch_up_masks],
            "knowledge": [batch_kg_ids, batch_kg_segs, batch_kg_poss, batch_kg_hops, batch_kg_masks],
            "conversation": [batch_hs_ids, batch_hs_segs, batch_hs_poss, batch_hs_masks],
            "target": [batch_tg_type_ids,batch_tg_entity_ids],
            "current":[batch_cur_type_ids,batch_cur_entity_ids],
            "predict":[batch_pre_type_ids,batch_pre_entity_ids]
        }

        return collated_batch
