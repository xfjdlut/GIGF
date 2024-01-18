from transformers import GPT2LMHeadModel
import torch.nn as nn


class ResGen(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")#初始化模型

    def forward(self, input_ids, lm_labels):
        outputs = self.gpt2(input_ids, labels=lm_labels, return_dict=True)
        encode_state = outputs['logits']
        loss_g=None
        if lm_labels is not None:
            loss_g=outputs['loss']
        return loss_g, encode_state

