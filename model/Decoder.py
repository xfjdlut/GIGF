# -*- coding: utf-8 -*-
import math
import random

import torch
from torch import nn
from typing import Optional, Tuple

from model.ModelOutput import ModelOutput


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

ACT2FN = {
    "relu": nn.functional.relu,
    "gelu": gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}

class DecoderOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


# attention
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        init_std: float = 0.01,
        is_decoder: bool = False,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.init_std = init_std
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _init_all_weights(self):
        self.k_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.k_proj.bias is not None:
            self.k_proj.bias.data.zero_()
        self.v_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.v_proj.bias is not None:
            self.v_proj.bias.data.zero_()
        self.q_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.q_proj.bias is not None:
            self.q_proj.bias.data.zero_()
        self.out_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.out_proj.bias is not None:
            self.out_proj.bias.data.zero_()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling

        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        # 维度变换
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )


        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # None
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class FusionGate(nn.Module):
    def __init__(self, d_model, init_std=0.01):
        super().__init__()
        self.init_std = init_std
        self.gate_layer = nn.Linear(d_model*2, 1, bias=True)
        self.sigmod = nn.Sigmoid()
    
    def _init_all_weights(self):
        self.gate_layer.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.gate_layer.bias is not None:
            self.gate_layer.bias.data.zero_()

    def forward(self, k_state, h_state):
        gate = self.sigmod(self.gate_layer(torch.cat([k_state, h_state],  -1)))
        f_state = gate * k_state + (1 - gate) * h_state
        return f_state


class Norm_FusionGate(nn.Module):
    def __init__(self, d_model, init_std=0.01):
        super().__init__()
        self.init_std = init_std
        self.gate_layer = nn.Linear(d_model, 1, bias=True)
        self.sigmod = nn.Sigmoid()

    def _init_all_weights(self):
        self.gate_layer.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.gate_layer.bias is not None:
            self.gate_layer.bias.data.zero_()

    def forward(self, k_state, h_state, u_state):
        gate = self.sigmod(self.gate_layer(u_state))
        f_state = gate * k_state + (1 - gate) * h_state
        return f_state


class DecoderLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.init_std = args.init_std
        self.dropout = args.dropout
        assert args.activation_function in ("relu", "gelu", "tanh", "sigmoid")
        self.activation_fn = ACT2FN[args.activation_function]
        self.activation_dropout = args.dropout

        # 初始化图模型
        self.kg_gate_layer =Norm_FusionGate(d_model=self.embed_dim*2, init_std=self.init_std)

        self.kg_hs = nn.Linear(self.embed_dim*2,self.embed_dim)
        self.kg_hs_sigmiod= nn.Sigmoid()
        self.kg_hs_layer_norm = nn.LayerNorm(self.embed_dim)

        # 四个多头注意力层
        self.type_context_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.dropout,
            init_std=args.init_std,
            is_decoder=True
        )
        self.entity_context_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.dropout,
            init_std=args.init_std,
            is_decoder=True
        )
        self.type_hs_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.dropout,
            init_std=args.init_std,
            is_decoder=True
        )
        self.entity_kg_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.dropout,
            init_std=args.init_std,
            is_decoder=True
        )
        self.type_context_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.entity_context_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.type_hs_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.entity_kg_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fusion_layer = FusionGate(d_model=self.embed_dim, init_std=self.init_std)
        self.fc1 = nn.Linear(self.embed_dim,args.decoder_ffn_dim)
        self.fc2 = nn.Linear(args.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def _init_all_weights(self):
        self.kg_gate_layer._init_all_weights()
        self.type_context_attn._init_all_weights()
        self.entity_context_attn._init_all_weights()
        self.type_hs_attn._init_all_weights()
        self.entity_kg_attn._init_all_weights()
        self.fusion_layer._init_all_weights()
        self.fc1.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.fc1.bias is not None:
            self.fc1.bias.data.zero_()
        self.fc2.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.fc2.bias is not None:
            self.fc2.bias.data.zero_()

        self.kg_hs.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.kg_hs.bias is not None:
            self.kg_hs.bias.data.zero_()

    def forward(
        self,
        cur_type_embedding,
        cur_entity_embedding,
        kg_gate_input,
        up_gate_input,
        hs_gru_input,
        kg_encoder_hidden_states: torch.Tensor = None,              # (batch_size, seq_len, hidden_size)
        kg_encoder_attention_mask: Optional[torch. Tensor] = None,  # (batch_size, 1, tgt_len, src_len)
        up_encoder_hidden_states: torch.Tensor = None,              # (batch_size, seq_len, hidden_size)
        up_encoder_attention_mask: Optional[torch.Tensor] = None,   # (batch_size, 1, tgt_len, src_len)
        hs_encoder_hidden_states: torch.Tensor = None,              # (batch_size, seq_len, hidden_size)
        hs_encoder_attention_mask: Optional[torch.Tensor] = None,   # (batch_size, 1, tgt_len, src_len)
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False
    ):
        # kg_gate
        cur_embedding=torch.cat((cur_type_embedding,cur_entity_embedding),-1)
        kg_gate_output = self.kg_gate_layer(kg_gate_input, up_gate_input, cur_embedding)

        context= torch.cat((hs_gru_input, kg_gate_output), -1)
        context=self.kg_hs(context)
        context=self.kg_hs_sigmiod(context)
        context=self.kg_hs_layer_norm(context)

        cur_type_embedding = cur_type_embedding.unsqueeze(dim=1)  # 首先修改type和entity embedding的维度-->(batch,1,embedding_dim)
        cur_entity_embedding = cur_entity_embedding.unsqueeze(dim=1)

        residual=cur_type_embedding
        type_context_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        cur_type_embedding, _, type_context_attn_present_key_value = self.type_context_attn(
            hidden_states=cur_type_embedding,
            key_value_states=context,
            attention_mask=None,
            layer_head_mask=layer_head_mask,
            past_key_value=type_context_attn_past_key_value,
            output_attentions=output_attentions,
        )
        cur_type_embedding=nn.functional.dropout(cur_type_embedding, p=0.4, training=self.training)
        cur_type_embedding=residual+cur_type_embedding
        cur_type_embedding = self.type_context_attn_layer_norm(cur_type_embedding)

        residual = cur_entity_embedding
        entity_context_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        cur_entity_embedding, _, entity_context_attn_present_key_value = self.entity_context_attn(
            hidden_states=cur_entity_embedding,
            key_value_states=context,
            attention_mask=None,
            layer_head_mask=layer_head_mask,
            past_key_value=entity_context_attn_past_key_value,
            output_attentions=output_attentions,
        )
        cur_entity_embedding = nn.functional.dropout(cur_entity_embedding, p=self.dropout, training=self.training)
        cur_entity_embedding = residual + cur_entity_embedding
        cur_entity_embedding = self.entity_context_attn_layer_norm(cur_entity_embedding)

        residual = cur_entity_embedding
        entity_kg_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        entity_kg_hidden_states, _, up_present_key_value = self.type_hs_attn(
            hidden_states=cur_entity_embedding,
            key_value_states=kg_encoder_hidden_states,
            attention_mask=kg_encoder_attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=entity_kg_attn_past_key_value,
            output_attentions=output_attentions,
        )
        cur_entity_embedding = nn.functional.dropout(entity_kg_hidden_states, p=self.dropout, training=self.training)
        cur_entity_embedding = residual + cur_entity_embedding
        cur_entity_embedding = self.entity_kg_attn_layer_norm(cur_entity_embedding)

        cur_entity_embedding = self.fusion_layer(cur_entity_embedding,entity_kg_hidden_states)

        ffn_entity_embedding = self.activation_fn(self.fc1(cur_entity_embedding))
        ffn_entity_embedding = nn.functional.dropout(ffn_entity_embedding, p=self.activation_dropout,training=self.training)
        ffn_entity_embedding = self.activation_fn(self.fc2(ffn_entity_embedding))
        ffn_entity_embedding = nn.functional.dropout(ffn_entity_embedding, p=self.dropout, training=self.training)

        final_entity_embedding = cur_entity_embedding + ffn_entity_embedding  # 残差
        final_entity_embedding = self.final_layer_norm(final_entity_embedding)

        cur_type_embedding = cur_type_embedding.squeeze(dim=1)
        final_entity_embedding= final_entity_embedding.squeeze(dim=1)

        return cur_type_embedding, final_entity_embedding

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.device=args.device
        self.embed_dim = args.embed_dim
        self.vocab_size = args.vocab_size
        self.decoder_layers = args.decoder_layers
        self.dropout = args.dropout
        self.layerdrop = args.decoder_layerdrop
        self.padding_idx = args.pad_token_id
        self.max_position_embeddings = args.max_position_embeddings
        self.embed_scale = math.sqrt(args.embed_dim) if args.scale_embedding else 1.0
        self.init_std = args.init_std
        self.output_attentions = args.output_attentions
        self.output_hidden_states = args.output_hidden_states
        self.use_cache = args.use_cache
        self.fusion_decoder_layers = nn.ModuleList([DecoderLayer(args) for _ in range(self.decoder_layers)])
        # gate
        self._init_all_weights()
        self.gradient_checkpointing = False


    def _init_module_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _init_all_weights(self):
        for layer in self.fusion_decoder_layers:
            layer._init_all_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        cur_type_embedding,
        cur_entity_embedding,
        kg_gate_input,
        up_gate_input,
        hs_gru_input,
        kg_encoder_hidden_states=None,      # (batch_size, kg_seq_len, hidden_size)
        kg_encoder_attention_mask=None,     # (batch_size, kg_seq_len)
        up_encoder_hidden_states=None,      # (batch_size, up_seq_len, hidden_size)
        up_encoder_attention_mask=None,     # (batch_size, up_seq_len)
        hs_encoder_hidden_states=None,      # (batch_size, hs_seq_len, hidden_size)
        hs_encoder_attention_mask=None,     # (batch_size, hs_seq_len)
        head_mask=None,                     # (decoder_layers, decoder_attention_heads)
        cross_attn_head_mask=None,          # (decoder_layers, decoder_attention_heads)
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.use_cache

        # 计算mask矩阵
        type_hs_attention_mask=_expand_mask(hs_encoder_attention_mask,  cur_type_embedding.dtype,tgt_len=1)
        entity_kg_attention_mask = _expand_mask(kg_encoder_attention_mask, cur_type_embedding.dtype, tgt_len=1)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and kg_encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None
        #  decoder层
        layer_outputs=[]
        for idx, decoder_layer in enumerate(self.fusion_decoder_layers):
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_outputs = decoder_layer(
                    cur_type_embedding,
                    cur_entity_embedding,  # (batch_size, target_seq_len)
                    kg_gate_input,
                    up_gate_input,
                    hs_gru_input,
                    kg_encoder_hidden_states=kg_encoder_hidden_states,
                    kg_encoder_attention_mask=entity_kg_attention_mask,
                    up_encoder_hidden_states=up_encoder_hidden_states,
                    up_encoder_attention_mask=up_encoder_attention_mask,
                    hs_encoder_hidden_states=hs_encoder_hidden_states,
                    hs_encoder_attention_mask=type_hs_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                    output_attentions=output_attentions
            )
            cur_type_embedding=layer_outputs[0]
            cur_entity_embedding=layer_outputs[1]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if kg_encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        return DecoderOutput(
            final_type_embedding=layer_outputs[0],
            final_entity_embedding=layer_outputs[1],
        )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
