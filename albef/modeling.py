import math
import torch
import random
from torch import Tensor, nn
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from transformers import BertPreTrainedModel, ViTModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertOnlyMLMHead, BertIntermediate, BertOutput, BertPooler
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

#
#
#
@torch.no_grad()
def all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(context, tensor):
        tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor)
        return tuple(tensors_gather)

    @staticmethod
    def backward(context, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors. Graph remains connected for backward grad computation.
    """
    world_size = torch.distributed.get_world_size()

    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)



#
# copied from transformers/models/bert/modeling_bert.py & Albef/models/xbert.py
#
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        #
        # NOTE: Q and K/V have the same input dimenssion.
        #
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

        #
        # copied from ablef - Albef/models/xbert.py
        #
        self.save_attention = False
        self.attention_prob = None
        self.attention_grad = None

    #
    # copied from ablef - Albef/models/xbert.py
    #
    def save_attention_prob(self, attention_prob):
        self.attention_prob = attention_prob
    #
    # copied from ablef - Albef/models/xbert.py
    #
    def get_attention_prob(self):
        return self.attention_prob
    #
    # copied from ablef - Albef/models/xbert.py
    #
    def save_attention_grad(self, attention_grad):
        self.attention_grad = attention_grad
    #
    # copied from ablef - Albef/models/xbert.py
    #
    def get_attention_grad(self):
        return self.attention_grad


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_prob = F.softmax(attention_scores, dim=-1)

        #
        # copied from Albef/models/xbert.py
        #
        if self.save_attention:
            self.save_attention_prob(attention_prob)
            if attention_prob.requires_grad:
                attention_prob.register_hook(self.save_attention_grad)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_prob = self.dropout(attention_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attention_prob = attention_prob * head_mask

        context_layer = torch.matmul(attention_prob, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_prob) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs




#
# copied from transformers/models/bert/modeling_bert.py, modified to address attention shift, v1, equals to v0
#
class BertSelfOutput(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #
        # for gated cross attention, to address attention shift.
        #
        if is_cross_attention:
            self.gate = nn.Linear(config.hidden_size, 1)
        else:
            self.gate = None


    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        #
        # for gated cross attention, to address attention shift.
        #
        if self.gate is not None:
            hidden_states = hidden_states * torch.sigmoid(self.gate(hidden_states + input_tensor))

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states





#
# copied from transformers/models/bert/modeling_bert.py & Albef/models/xbert.py
#
class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config, is_cross_attention)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs





#
# customer defined BertLayer, supporting cross attention (query-only cross attention) for encoding.
#
class BertLayer(nn.Module):
    def __init__(self, config, has_cross_attention):
        """
        layer num is used to contral cross attention.
        """
        super().__init__()
        self.config = config
        self.has_cross_attention = has_cross_attention
        #
        # self attention.
        #
        self.attention = BertAttention(config)
        #
        # cross attention.
        #
        if self.has_cross_attention:
            #
            # NOTE: Q and K/V have the same input dimenssion.
            #
            self.crossattention = BertAttention(config, is_cross_attention=True)
        #
        #
        #
        self.intermediate = BertIntermediate(config)
        #
        # skip layernom
        #
        self.output = BertOutput(config)



    #
    # support hidden states, return layer output (hidden states).
    #
    def forward(
        self,
        hidden_state,
        attention_mask=None,
        encoder_hidden_state=None,
        encoder_attention_mask=None,
    ):
        #
        # self attention: qkv + skip layer norm.
        #
        attention_output = self.attention(hidden_state, attention_mask)[0]
        #
        # cross attention: qkv + skip layer norm. hidden states are not masked.
        #
        if self.has_cross_attention:
            #
            # cross attention: qkv + skip layer norm.
            #
            attention_output = self.crossattention(
                hidden_states = attention_output,
                encoder_hidden_states = encoder_hidden_state,
                encoder_attention_mask = encoder_attention_mask,
            )[0]
            #
            # intermediate.
            #
            intermediate_output = self.intermediate(attention_output)
            #
            # skip layernom.
            #
            layer_output = self.output(intermediate_output, attention_output)
        else:
            #
            # intermediate.
            #
            intermediate_output = self.intermediate(attention_output)
            #
            # skip layernom
            #
            layer_output = self.output(intermediate_output, attention_output)

        return layer_output


    #
    # chunking intermediate.
    #
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



#
# support hidden states, return the last layer hidden state.
#
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, i >= config.fusion_layer) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_state,
        attention_mask=None,
        encoder_hidden_state=None,
        encoder_attention_mask=None,
        mode=None,
    ):
        if mode=='low':
            from_layer = 0
            last_layer = self.config.fusion_layer
        elif mode=='top':
            from_layer = self.config.fusion_layer
            last_layer = self.config.num_hidden_layers
        else:
            from_layer = 0
            last_layer = self.config.num_hidden_layers

        #
        # low layers apply self attention, top layers apply self attention + cross attention.
        #
        for i in range(from_layer, last_layer):
            hidden_state = self.layer[i](
                hidden_state,
                attention_mask,
                encoder_hidden_state,
                encoder_attention_mask,
            )

        #
        # only return the last hidden state.
        #
        return hidden_state



#
# support input ids and hidden states, return the last layer hidden states and the pooler states.
#
class BertModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        #
        # text embedding table.
        #
        self.embeddings = BertEmbeddings(config)
        #
        # bert encoder supporting self attention and cross attention.
        #
        self.encoder = BertEncoder(config)
        #
        # pooler = Dense(sequence[0])
        #
        self.pooler = BertPooler(config)
        #
        # initialize weights and apply final processing.
        #
        self.post_init()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings


    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    #
    # Provided a padding mask of dimensions [batch_size, seq_length]
    # - if the model is a decoder, apply a causal mask in addition to the padding mask
    # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
    #
    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, is_decoder=False):
        if is_decoder:
            batch_size, seq_length = input_shape
            #
            # [seq_length]
            #
            seq_ids = torch.arange(seq_length, device=device)
            #
            # [batch_size, num_heads, seq_length, seq_length]
            #
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            #
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask causal and
            # attention masks must have same type with pytorch version < 1.3
            #
            causal_mask = causal_mask.to(attention_mask.dtype)

            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat(
                    [
                        torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                        causal_mask,
                    ],
                    axis=-1,
                )
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            #
            # [batch_size, num_heads, seq_length, seq_length]
            #
            extended_attention_mask = attention_mask[:, None, None, :]

        #
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for masked positions, this operation
        # will create a tensor which is 0.0 for positions we want to attend and the dtype's smallest value for masked
        # positions. Since we are adding it to the raw scores before the softmax, this is effectively the same as
        # removing these entirely.
        #
        extended_attention_mask = extended_attention_mask.to(self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        return extended_attention_mask


    #
    # support cross attention. Different with BertPreTrainedModel::forward().
    #
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        hidden_state=None,
        encoder_hidden_state=None,
        encoder_attention_mask=None,
        mode=None,
        is_decoder=False,
    ):
        if input_ids is not None:
            input_embedding = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
            device = input_ids.device
        else:
            input_embedding = hidden_state
            device = hidden_state.device

        input_shape = input_embedding.size()[:-1]

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        #
        # apply a causal mask in addition to the padding mask for decoder.
        #
        attention_mask = self.get_extended_attention_mask(attention_mask=attention_mask, input_shape=input_shape, device=device, is_decoder=is_decoder)

        #
        # support encoder hidden state.
        #
        if encoder_hidden_state is not None:
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_state.size()[:-1], device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        #
        # only return the last hidden state.
        #
        hidden_state = self.encoder(
            input_embedding,
            attention_mask=attention_mask,
            encoder_hidden_state=encoder_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            mode=mode,
        )
        #
        # pooler = Dense(sequence[0])
        #
        pooler_output = self.pooler(hidden_state)
        #
        # only return last hidden state and pooled output.
        #
        return hidden_state, pooler_output



#
# for MLM training, support text input id and encodder hidden states
#
class BertMLMModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config=None, bert_model=None):
        super().__init__(config)

        if bert_model is not None:
            self.bert = BertModel.from_pretrained(bert_model, ignore_mismatched_sizes=True)
            self.config = self.bert.config
            self.cls = BertOnlyMLMHead(self.config)
        else:
            self.bert = BertModel(config)
            self.config = config
            self.cls = BertOnlyMLMHead(self.config)

        #
        # initialize weights and apply final processing.
        #
        self.post_init()


    def get_output_embeddings(self):
        return self.cls.predictions.decoder


    def set_output_embeddings(self, embeddings):
        self.cls.predictions.decoder = embeddings


    #
    # momentum distilling with soft label and return logits. Different with BertPreTrainedModel::forward().
    #
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        encoder_hidden_state=None,
        encoder_attention_mask=None,
        return_logit=False,
        label=None,
        label_soft=None,
        alpha=0,
    ):
        #
        # only return the last hidden states. lower layers ignore encoder hidden states.
        #
        hidden_state, _ = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_state=encoder_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
        )
        #
        # [batch_size, seq_length, vocab_size]
        #
        logit = self.cls(hidden_state)
        if return_logit:
            return logit
        #
        # hard label cross entropy loss.
        #
        loss = F.cross_entropy(logit.view(-1, self.config.vocab_size), label.view(-1))
        #
        # soft label cross entropy loss.
        #
        if label_soft is not None:
            info = -torch.sum(F.log_softmax(logit, dim=-1) * label_soft, dim=-1)
            loss = alpha * (info[label != -100].mean()) + (1.0 - alpha) * loss

        return loss



#
# For Causal Language Modeling training. PreTrainedModel inherits GenerationMixin before 4.50.
#
class BertCLMModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.post_init()


    def get_output_embeddings(self):
        return self.cls.predictions.decoder


    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings


    #
    # overwrite BertPreTrainedModel::forward()
    #
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        label=None,
        weight=None,
        return_logit=False,
        reduction='mean',
        **kwargs
    ):
        hidden_state, _ = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_state=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            is_decoder=True,
        )

        logit = self.cls(hidden_state)

        if return_logit:
            return logit

        loss = None

        if label is not None:
            #
            # shift logits and target ids by one, [batch_size, seq_length - 1, vocab_size]
            #
            shift_logit = logit[:, :-1, :].contiguous()
            shift_label = label[:, 1:].contiguous()
            #
            # optional label_smoothing = 0.1
            #
            if weight is None:
                loss = F.cross_entropy(shift_logit.view(-1, self.config.vocab_size), shift_label.view(-1), reduction=reduction)
            else:
                loss = weight.repeat_interleave(shift_label.size(1), dim=0) * F.cross_entropy(shift_logit.view(-1, self.config.vocab_size), shift_label.view(-1), reduction='none')
                if reduction == 'sum':
                    loss = loss.sum()
                elif reduction == 'mean':
                    loss = loss.mean()

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logit,
        )


    #
    # overwrite.
    #
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **model_kwargs
    ):
        input_shape = input_ids.shape
        #
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        #
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        #
        # cut decoder_input_ids if past_key_values is used
        #
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "is_decoder": True,
        }


    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past



#
# Masked Language Modeling: Image-Text-Contrastive, Image-Text-Match, Masked-Language-Model, Cross-Match (pseudo query, optional).
#
class AlbefMLMModel(nn.Module):
    def __init__(self, tokenizer, bert_config=None, vit_config=None, mlm_model=None, vit_model=None, vit_frozen=True, queue_size=65536, world_size=1, global_rank=0):
        super().__init__()

        self.tokenizer = tokenizer

        if mlm_model is not None:
            self.mlm_model = BertMLMModel.from_pretrained(mlm_model, ignore_mismatched_sizes=True)
            self.bert_config = self.mlm_model.config
        else:
            self.mlm_model = BertMLMModel(config=bert_config)
            self.bert_config = bert_config

        self.mlm_model_m = BertMLMModel(config=self.bert_config)

        if vit_model is not None:
            self.vit_model = ViTModel.from_pretrained(vit_model, ignore_mismatched_sizes=True)
            self.vit_config = self.vit_model.config
        else:
            self.vit_model = ViTModel(config=vit_config)
            self.vit_config = vit_config

        self.vit_model_m = ViTModel(config=self.vit_config)

        #
        # frozen vit model.
        #
        self.vit_frozen = vit_frozen
        if self.vit_frozen:
            for param in self.vit_model.parameters():
                param.requires_grad = False

        #
        # momentum contrastive + momentum distilling.
        #
        self.model_pairs = [[self.mlm_model, self.mlm_model_m], [self.vit_model, self.vit_model_m]]
        self.copy_params()

        self.queue_size = queue_size
        self.register_buffer("queue_image", torch.randn(self.bert_config.hidden_size, self.queue_size))
        self.register_buffer("queue_text", torch.randn(self.bert_config.hidden_size, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.queue_image = F.normalize(self.queue_image, dim=0)
        self.queue_text = F.normalize(self.queue_text, dim=0)

        #
        # a relatively large momentum (e.g., m = 0.999, our default) works much better than a smaller value (e.g., m = 0.9), suggesting that a slowly evolving key encoder is a core to making use of a queue.
        #
        self.momentum = 0.999

        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        self.itm_head = nn.Linear(self.bert_config.hidden_size, 2)

        self.world_size = world_size
        self.global_rank = global_rank

        #
        # save cross attention for consistency constraint.
        #
        for i in range(self.bert_config.fusion_layer, self.bert_config.num_hidden_layers):
            self.mlm_model.bert.encoder.layer[i].crossattention.self.save_attention = True



    #
    # adjust vit position from the original model.
    #
    def load(self, model_file, vit_position=-1, ignore_queue=True):
        params = torch.load(model_file, map_location='cpu')

        states = {}

        for key, value in params.items():
            if vit_position > 0 and "vit" in key and "position" in key:
                states[key] = value[:, :vit_position, :]
                continue

            if ignore_queue and "queue" in key:
                continue

            states[key] = value

        self.load_state_dict(states, strict=False)

        self.copy_params()


    #
    # pixel values: [batch * 3, chanel, hight, width]
    #
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        alpha=0,
        count=8,
        masked_inputs=None,
        masked_attention=None,
        masked_labels=None,
        apply_gather=False,
        apply_cross=False,
    ):

        with torch.no_grad():
            self.temperature.clamp_(0.001, 0.5)
        #
        # extract the cls tensor, NOT the pooler output, because MLM model has no pooler layer.
        #
        if self.vit_frozen:
            with torch.no_grad():
                image_hidden_state = self.vit_model(pixel_values=pixel_values).last_hidden_state
        else:
            image_hidden_state = self.vit_model(pixel_values=pixel_values).last_hidden_state

        #
        # normalize L2
        #
        image_cls_state = F.normalize(image_hidden_state[:, 0, :], dim=-1)

        text_hidden_state, _ = self.mlm_model.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mode='low')
        text_cls_state = F.normalize(text_hidden_state[:, 0, :], dim=-1)


        with torch.no_grad():
            batch_size = pixel_values.size(0)

        #
        # ================= ITC ========================
        #
        # momentum ITC
        #
        with torch.no_grad():
            #
            # update momentum models.
            #
            self._momentum_update()
            #
            # rum model forward to get image cls.
            #
            image_hidden_state_m = self.vit_model_m(pixel_values=pixel_values).last_hidden_state

        #
        # normalize, L2.
        #
        image_cls_state_m = F.normalize(image_hidden_state_m[:, 0, :], dim=-1)
        #
        #
        #
        with torch.no_grad():
            #
            # gather all image cls, [ world_size x batch_size x hidden_size ] or [ 1 x batch_size x hidden_size ]
            #
            if apply_gather:
                image_cls_state_g = all_gather(image_cls_state_m)
            else:
                image_cls_state_g = image_cls_state_m

            #
            # [ world_size x batch_size + queue_size ]
            #
            image_cls_state_all = torch.cat([image_cls_state_g.t(), self.queue_image.clone().detach()], dim=1)
            #
            # run model forward to get text cls.
            #
            text_hidden_state_m, _ = self.mlm_model_m.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mode='low')
            text_cls_state_m = F.normalize(text_hidden_state_m[:, 0, :], dim=-1)
            #
            # gather all text cls, [ world_size x batch_size x hidden_size ] or [ 1 x batch_size x hidden_size ]
            #
            if apply_gather:
                text_cls_state_g = all_gather(text_cls_state_m)
            else:
                text_cls_state_g = text_cls_state_m

            #
            # [ world_size x batch_size + queue_size ]
            #
            text_cls_state_all = torch.cat([text_cls_state_g.t(), self.queue_text.clone().detach()], dim=1)
            #
            # [ batch_size, world_size x batch_size + queue_size ]
            #
            sim_i2t_m = image_cls_state_m @ text_cls_state_all / self.temperature
            sim_t2i_m = text_cls_state_m @ image_cls_state_all / self.temperature
            #
            # [ batch_size,  world_size x batch_size + queue_size ]
            #
            sim_targets = torch.zeros(sim_i2t_m.size()).to(pixel_values.device)

            if apply_gather:
                sim_targets[:, self.global_rank * batch_size : self.global_rank * batch_size + batch_size].fill_diagonal_(1)
            else:
                sim_targets[:, 0 : batch_size].fill_diagonal_(1)
            #
            # [ batch_size, world_size x batch_size + queue_size ]
            #
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        #
        # [ batch_size, world_size x batch_size + queue_size ]
        #
        sim_i2t = image_cls_state @ text_cls_state_all / self.temperature
        sim_t2i = text_cls_state @ image_cls_state_all / self.temperature
        #
        # in-world InfoNCE.
        #
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t+loss_t2i)/2

        #
        # update momentum queue.
        #
        self._dequeue_and_enqueue(image_cls_state_g, text_cls_state_g)

        #
        # ================= ITM ========================
        #
        # in-batch ITM
        #
        _, fusion_pooler_output_pos = self.mlm_model.bert(
            hidden_state = text_hidden_state,
            attention_mask = attention_mask,
            encoder_hidden_state = image_hidden_state,
            mode = 'top',
        )
        #
        #
        #
        with torch.no_grad():
            if apply_gather:
                weights_i2t = sim_i2t[:, :self.world_size * batch_size].clone()
                weights_t2i = sim_t2i[:, :self.world_size * batch_size].clone()

                weights_i2t[:, self.global_rank * batch_size : self.global_rank * batch_size + batch_size].fill_diagonal_(-10000.0)
                weights_t2i[:, self.global_rank * batch_size : self.global_rank * batch_size + batch_size].fill_diagonal_(-10000.0)
            else:
                weights_i2t = sim_i2t[:, : batch_size].clone()
                weights_t2i = sim_t2i[:, : batch_size].clone()

                weights_i2t[:, 0 : batch_size].fill_diagonal_(-10000.0)
                weights_t2i[:, 0 : batch_size].fill_diagonal_(-10000.0)

            weights_i2t = F.softmax(weights_i2t, dim=1)
            weights_t2i = F.softmax(weights_t2i, dim=1)

            if apply_gather:
                weights_i2t[:, self.global_rank * batch_size : self.global_rank * batch_size + batch_size].fill_diagonal_(0)
                weights_t2i[:, self.global_rank * batch_size : self.global_rank * batch_size + batch_size].fill_diagonal_(0)
            else:
                weights_i2t[:, 0 : batch_size].fill_diagonal_(0)
                weights_t2i[:, 0 : batch_size].fill_diagonal_(0)

        #
        # in-world input ids.
        #
        if apply_gather:
            input_ids_g = all_gather(input_ids)
        else:
            input_ids_g = input_ids

        #
        # in-world attention masks.
        #
        if apply_gather:
            text_attention_mask_g = all_gather(attention_mask)
        else:
            text_attention_mask_g = attention_mask

        #
        # [ batch_size * world_size, length, hidden_size ]
        #
        if apply_gather:
            text_hidden_state_g = all_gather_with_grad(text_hidden_state)
            image_hidden_state_g = all_gather_with_grad(image_hidden_state)
        else:
            text_hidden_state_g = text_hidden_state
            image_hidden_state_g = image_hidden_state

        image_hidden_state_neg = []

        for b in range(batch_size):
            #
            # in-world hard negative sampling.
            #
            idx = torch.multinomial(weights_t2i[b], 1).item()
            image_hidden_state_neg.append(image_hidden_state_g[idx])

        #
        # [ batch_size, length, hidden_size ]
        #
        image_hidden_state_neg = torch.stack(image_hidden_state_neg, dim=0)

        text_hidden_state_neg = []
        attention_mask_neg = []

        for b in range(batch_size):
            #
            # in-world hard negative sampling.
            #
            idx = torch.multinomial(weights_i2t[b], 1).item()
            text_hidden_state_neg.append(text_hidden_state_g[idx])
            attention_mask_neg.append(text_attention_mask_g[idx])
        #
        # [ batch_size, length, hidden_size ]
        #
        text_hidden_state_neg = torch.stack(text_hidden_state_neg, dim=0)
        attention_mask_neg = torch.stack(attention_mask_neg, dim=0)
        #
        # [ batch_size * 2, length, hidden_size ]
        #
        text_hidden_state_all = torch.cat([text_hidden_state, text_hidden_state_neg], dim=0)
        text_attention_mask_all = torch.cat([attention_mask, attention_mask_neg], dim=0)
        #
        # [ batch_size * 2, length, hidden_size ]
        #
        image_hidden_state_all = torch.cat([image_hidden_state_neg, image_hidden_state], dim=0)
        #
        # [ batch_size * 2, length, hidden_size ]
        #
        _, fusion_pooler_output_neg = self.mlm_model.bert(
            hidden_state = text_hidden_state_all,
            attention_mask = text_attention_mask_all,
            encoder_hidden_state = image_hidden_state_all,
            mode = 'top',
        )
        #
        # positive and negative, [batch * 3, 1, hidden]
        #
        vl_states = torch.cat([fusion_pooler_output_pos, fusion_pooler_output_neg], dim=0)
        #
        # [batch * 3, 1, 2]
        #
        vl_output = self.itm_head(vl_states)
        #
        # [batch * 3, 1]
        #
        itm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(2*batch_size, dtype=torch.long)], dim=0).to(pixel_values.device)
        #
        # in-world cross entropy.
        #
        loss_itm = F.cross_entropy(vl_output, itm_labels, label_smoothing=0.01)
        #
        # ================= MLM ========================
        #
        # emperically, momentum distilling is not neccessary for MLM
        #
        inputs = input_ids.clone()
        labels = input_ids.clone()
        #
        # momoentum distilling.
        #
        with torch.no_grad():
            logit_m = self.mlm_model_m(
                masked_inputs,
                attention_mask=masked_attention,
                token_type_ids=token_type_ids,
                encoder_hidden_state=image_hidden_state_m,
                return_logit = True,
            )

        loss_mlm = self.mlm_model(
            masked_inputs,
            attention_mask=masked_attention,
            token_type_ids=token_type_ids,
            encoder_hidden_state=image_hidden_state,
            label=masked_labels,
            label_soft=F.softmax(logit_m, dim=-1),
            alpha=alpha,
        )
        #
        # ============== CROSS MATCH ===================
        #
        # in-world cross match.
        #
        loss_cross = 0.0

        if apply_cross is True:
            #
            # random cross matching
            #
            loss_cross = self.cross_match(weights_i2t, input_ids, input_ids_g, attention_mask, image_hidden_state, count=count)

        #
        # ============== Consistency ===================
        #
        loss_consistency = 0.0
        #
        # attention consistency for addressing cross-attention shift.
        #
        for i in range(self.bert_config.fusion_layer + 1, self.bert_config.num_hidden_layers):
            loss_consistency -= ((self.mlm_model.bert.encoder.layer[i].crossattention.self.get_attention_prob().mean(1) + 0.0000000001).log() * self.mlm_model.bert.encoder.layer[self.bert_config.fusion_layer].crossattention.self.get_attention_prob().mean(1).detach()).mean()

        #
        # =================== loss ===================
        #
        return loss_mlm, loss_ita, loss_itm, loss_cross, loss_consistency



    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False


    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    #
    # dequeue, enqueue.
    #
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_cls_state, text_cls_state):
        batch_size = image_cls_state.shape[0]
        ptr = int(self.queue_ptr)
        #
        # for simplicity
        #
        assert self.queue_size % batch_size == 0
        #
        # replace the keys at ptr (dequeue and enqueue)
        #
        self.queue_image[:, ptr:ptr + batch_size] = image_cls_state.T
        self.queue_text[:, ptr:ptr + batch_size] = text_cls_state.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr



    #
    # token: [CLS  a   b   c   d   e   f   g  SEP PAD PAD]
    # type:  [ 1   1   1   1   1   1   1   1   1   0   0 ]
    # mask:  [ 1   1   1   1   1   1   1   1   1   0   0 ]
    #
    #  ->
    #
    # token: [CLS  a'  b'  c' SEP  e   f   g  SEP PAD PAD]
    # type:  [ 0   0   0   0   0   1   1   1   1   0   0 ]
    # mask:  [ 1   1   1   1   1   1   1   1   1   0   0 ]
    #
    def cross_match(self, weights, input_ids, input_ids_g, attention_mask, image_hidden_state, count=10):
        if count == 0:
            return 0

        input_ids_pos = input_ids.clone()
        input_ids_neg = input_ids.clone()

        batch_size, _ = input_ids.size()

        for i in range(batch_size):
            #
            # cross positive sep.
            #
            input_ids_pos[i, count] = self.tokenizer.sep_token_id
            #
            # in-batch hard negative sampling.
            #
            j = torch.multinomial(weights[i], 1).item()
            #
            # cross negative sep.
            #
            input_ids_neg[i, count] = self.tokenizer.sep_token_id
            #
            # cross negative tokens.
            #
            input_ids_neg[i, 1:count] = input_ids_g[j, 1:count]

        #
        # clone the mask ids as the new type ids.
        #
        token_type_ids = attention_mask.clone()
        #
        # cross token type ids.
        #
        token_type_ids[:, 0:count+1] = 0

        #
        #
        #
        _, pooler_output_pos = self.mlm_model.bert(
            input_ids_pos,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_state=image_hidden_state,
        )

        _, pooler_output_neg = self.mlm_model.bert(
            input_ids_neg,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_state=image_hidden_state,
        )

        #
        # positive and negative, [batch * 2, 1, hidden]
        #
        hidden_state = torch.cat([pooler_output_pos, pooler_output_neg], dim=0)
        #
        # [batch * 2, 1, 2]
        #
        logit = self.itm_head(hidden_state)

        label = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)], dim=0).to(input_ids.device)

        return F.cross_entropy(logit, label, label_smoothing=0.01)



#
# Causal Language Modeling: Image-Text-Contrastive, Image-Text-Match, Masked-Language-Model, Cross-Match, Causal-Language-Model.
#
class AlbefCLMModel(nn.Module):
    def __init__(self, tokenizer, mlm_config=None, vit_config=None, clm_config=None, mlm_model=None, vit_model=None, clm_model=None, vit_frozen=True, queue_size=65536):
        super().__init__()

        self.tokenizer = tokenizer

        if mlm_model is not None:
            self.mlm_model = BertMLMModel.from_pretrained(mlm_model, ignore_mismatched_sizes=True)
            self.mlm_config = self.mlm_model.config
        else:
            self.mlm_model = BertMLMModel(config=mlm_config)
            self.mlm_config = mlm_config

        self.mlm_model_m = BertMLMModel(config=self.mlm_config)

        if vit_model is not None:
            self.vit_model = ViTModel.from_pretrained(vit_model, ignore_mismatched_sizes=True)
            self.vit_config = self.vit_model.config
        else:
            self.vit_model = ViTModel(config=vit_config)
            self.vit_config = vit_config

        self.vit_model_m = ViTModel(config=self.vit_config)

        if clm_model is not None:
            self.clm_model = BertCLMModel.from_pretrained(clm_model, ignore_mismatched_sizes=True)
            self.clm_config = self.clm_config.config
        else:
            self.clm_model = BertCLMModel(config=clm_config)
            self.clm_config = clm_config

        assert self.mlm_config.hidden_size == self.vit_config.hidden_size
        assert self.mlm_config.hidden_size == self.clm_config.hidden_size

        #
        # frozen vit model.
        #
        self.vit_frozen = vit_frozen
        if self.vit_frozen:
            for param in self.vit_model.parameters():
                param.requires_grad = False

        #
        # momentum contrastive.
        #
        self.model_pairs = [[self.mlm_model, self.mlm_model_m], [self.vit_model, self.vit_model_m]]

        self.copy_params()

        self.queue_size = queue_size
        self.register_buffer("queue_image", torch.randn(self.mlm_config.hidden_size, self.queue_size))
        self.register_buffer("queue_text", torch.randn(self.mlm_config.hidden_size, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.queue_image = F.normalize(self.queue_image, dim=0)
        self.queue_text = F.normalize(self.queue_text, dim=0)

        #
        # a relatively large momentum (e.g., m = 0.999, our default) works much better than a smaller value (e.g., m = 0.9), suggesting that a slowly evolving key encoder is a core to making use of a queue.
        #
        self.momentum = 0.999

        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        #
        # image text matching head.
        #
        self.itm_head = nn.Linear(self.mlm_config.hidden_size, 2)

        #
        # save cross attention for consistency constraint.
        #
        for i in range(self.mlm_config.fusion_layer, self.mlm_config.num_hidden_layers):
            self.mlm_model.bert.encoder.layer[i].crossattention.self.save_attention = True


    #
    # load a CLM model.
    #
    def load(self, model_file, ignore_queue=True):
        if ignore_queue:
            params = torch.load(model_file, map_location='cpu')
            states = {}
            for key, value in params.items():
                if "queue" in key:
                    continue
                states[key] = value
            self.load_state_dict(states, strict=False)
        else:
            self.load_state_dict(torch.load(model_file, map_location='cpu'))

        self.copy_params()


    #
    # load a MLM model
    #
    def load_model(self, model_file, ignore_queue=True):
        params = torch.load(model_file, map_location='cpu')
        states = {}
        for key, value in params.items():
            if ignore_queue and "queue" in key:
                continue

            states[key] = value

            if "mlm_model.bert" in key:
                key = key.replace("mlm_model.bert", "clm_model.bert")
                if key not in params:
                    states[key] = value

        self.load_state_dict(states, strict=False)
        self.copy_params()



    #
    # pixel values: [batch * 3, chanel, hight, width]
    #
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        alpha=0,
        masked_inputs=None,
        masked_attention=None,
        masked_labels=None,
        caption_ids=None,
        caption_mask=None,
        caption_type=None,
        weight=None,
    ):
        with torch.no_grad():
            self.temperature.clamp_(0.001, 0.5)
        #
        # extract the cls tensor, NOT the pooler output, because MLM model has no pooler layer.
        #
        if self.vit_frozen:
            with torch.no_grad():
                image_hidden_state = self.vit_model(pixel_values=pixel_values).last_hidden_state
        else:
            image_hidden_state = self.vit_model(pixel_values=pixel_values).last_hidden_state

        #
        # normalize L2
        #
        image_cls_state = F.normalize(image_hidden_state[:, 0, :], dim=-1)

        text_hidden_state, _ = self.mlm_model.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mode='low')
        text_cls_state = F.normalize(text_hidden_state[:, 0, :], dim=-1)


        with torch.no_grad():
            batch_size = pixel_values.size(0)

        #
        # ================= ITC ========================
        #
        # momentum ITC
        #
        with torch.no_grad():
            self._momentum_update()
            #
            # rum model forward to get image cls.
            #
            image_hidden_state_m = self.vit_model_m(pixel_values=pixel_values).last_hidden_state

        #
        # normalize, L2.
        #
        image_cls_state_m = F.normalize(image_hidden_state_m[:, 0, :], dim=-1)
        #
        #
        #
        with torch.no_grad():
            #
            # [ batch_size + queue_size ]
            #
            image_cls_state_all = torch.cat([image_cls_state_m.t(), self.queue_image.clone().detach()], dim=1)
            #
            # run model forward to get text cls.
            #
            text_hidden_state_m, _ = self.mlm_model_m.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mode='low')
            text_cls_state_m = F.normalize(text_hidden_state_m[:, 0, :], dim=-1)
            #
            # [ batch_size + queue_size ]
            #
            text_cls_state_all = torch.cat([text_cls_state_m.t(), self.queue_text.clone().detach()], dim=1)
            #
            # [ batch_size, batch_size + queue_size ]
            #
            sim_i2t_m = image_cls_state_m @ text_cls_state_all / self.temperature
            sim_t2i_m = text_cls_state_m @ image_cls_state_all / self.temperature
            #
            # [ batch_size, batch_size + queue_size ]
            #
            sim_targets = torch.zeros(sim_i2t_m.size()).to(pixel_values.device)

            sim_targets[:, 0 : batch_size].fill_diagonal_(1)

            #
            # [ batch_size, batch_size + queue_size ]
            #
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        #
        # [ batch_size, batch_size + queue_size ]
        #
        sim_i2t = image_cls_state @ text_cls_state_all / self.temperature
        sim_t2i = text_cls_state @ image_cls_state_all / self.temperature
        #
        # in-world InfoNCE.
        #
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t+loss_t2i)/2

        #
        # update momentum queue.
        #
        self._dequeue_and_enqueue(image_cls_state_m, text_cls_state_m)

        #
        # ================= ITM ========================
        #
        # in-batch ITM
        #
        fusion_hidden_state_pos, fusion_pooler_output_pos = self.mlm_model.bert(
            hidden_state = text_hidden_state,
            attention_mask = attention_mask,
            encoder_hidden_state = image_hidden_state,
            mode = 'top',
        )
        #
        #
        #
        with torch.no_grad():
            weights_i2t = sim_i2t[:, : batch_size].clone()
            weights_t2i = sim_t2i[:, : batch_size].clone()

            weights_i2t[:, 0 : batch_size].fill_diagonal_(-10000.0)
            weights_t2i[:, 0 : batch_size].fill_diagonal_(-10000.0)

            weights_i2t = F.softmax(weights_i2t, dim=1)
            weights_t2i = F.softmax(weights_t2i, dim=1)

            weights_i2t[:, 0 : batch_size].fill_diagonal_(0)
            weights_t2i[:, 0 : batch_size].fill_diagonal_(0)

        #
        # [ batch_size, length, hidden_size ]
        #
        image_hidden_state_neg = []

        for b in range(batch_size):
            #
            # in-world hard negative sampling.
            #
            idx = torch.multinomial(weights_t2i[b], 1).item()
            image_hidden_state_neg.append(image_hidden_state[idx])

        #
        # [ batch_size, length, hidden_size ]
        #
        image_hidden_state_neg = torch.stack(image_hidden_state_neg, dim=0)

        text_hidden_state_neg = []
        attention_mask_neg = []

        for b in range(batch_size):
            #
            # in-world hard negative sampling.
            #
            idx = torch.multinomial(weights_i2t[b], 1).item()
            text_hidden_state_neg.append(text_hidden_state[idx])
            attention_mask_neg.append(attention_mask[idx])
        #
        # [ batch_size, length, hidden_size ]
        #
        text_hidden_state_neg = torch.stack(text_hidden_state_neg, dim=0)
        attention_mask_neg = torch.stack(attention_mask_neg, dim=0)
        #
        # [ batch_size * 2, length, hidden_size ]
        #
        text_hidden_state_all = torch.cat([text_hidden_state, text_hidden_state_neg], dim=0)
        text_attention_mask_all = torch.cat([attention_mask, attention_mask_neg], dim=0)
        #
        # [ batch_size * 2, length, hidden_size ]
        #
        image_hidden_state_all = torch.cat([image_hidden_state_neg, image_hidden_state], dim=0)
        #
        # [ batch_size * 2, length, hidden_size ]
        #
        _, fusion_pooler_output_neg = self.mlm_model.bert(
            hidden_state = text_hidden_state_all,
            attention_mask = text_attention_mask_all,
            encoder_hidden_state = image_hidden_state_all,
            mode = 'top',
        )
        #
        # positive and negative, [batch * 3, 1, hidden]
        #
        vl_states = torch.cat([fusion_pooler_output_pos, fusion_pooler_output_neg], dim=0)
        #
        # [batch * 3, 1, 2]
        #
        vl_output = self.itm_head(vl_states)
        #
        # [batch * 3, 1]
        #
        itm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(batch_size * 2, dtype=torch.long)], dim=0).to(pixel_values.device)
        #
        # in-world cross entropy.
        #
        loss_itm = F.cross_entropy(vl_output, itm_labels, label_smoothing=0.01)

        #
        # ================= MLM ========================
        #
        # emperically, momentum distilling is not neccessary for MLM
        #
        with torch.no_grad():
            logit_m = self.mlm_model_m(
                masked_inputs,
                attention_mask=masked_attention,
                token_type_ids=token_type_ids,
                encoder_hidden_state=image_hidden_state_m,
                return_logit = True,
            )

        loss_mlm = self.mlm_model(
            masked_inputs,
            attention_mask=masked_attention,
            token_type_ids=token_type_ids,
            encoder_hidden_state=image_hidden_state,
            label=masked_labels,
            label_soft=F.softmax(logit_m, dim=-1),
            alpha=alpha,
        )

        #
        # ============== CROSS MATCH ===================
        #
        # in-world cross match.
        #
        loss_cross = self.cross_match(weights_i2t, weights_t2i, input_ids, caption_ids, attention_mask, token_type_ids, image_hidden_state)

        #
        # ================== CLM =======================
        #
        # causal language modeling.
        #
        loss_clm = self.clm_model(
            input_ids=caption_ids,
            attention_mask=caption_mask,
            token_type_ids=caption_type,
            encoder_hidden_states=fusion_hidden_state_pos,
            label=caption_ids.masked_fill(caption_ids == self.tokenizer.pad_token_id, -100),
            weight=weight
        ).loss

        #
        # ============== Consistency ===================
        #
        loss_consistency = 0.0
        #
        # attention consistency for addressing cross-attention shift.
        #
        for i in range(self.mlm_config.fusion_layer + 1, self.mlm_config.num_hidden_layers):
            loss_consistency -= ((self.mlm_model.bert.encoder.layer[i].crossattention.self.get_attention_prob().mean(1) + 0.0000000001).log() * self.mlm_model.bert.encoder.layer[self.mlm_config.fusion_layer].crossattention.self.get_attention_prob().mean(1).detach()).mean()


        #
        # ================== loss =======================
        #
        return loss_mlm, loss_ita, loss_itm, loss_cross, loss_clm, loss_consistency



    #
    # generation with beam search.
    #
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        prompt_ids=None,
        num_beams=3,
        max_length=16,
        min_length=1,
        repetition_penalty=1.0,
        num_return_sequences=3,
        output_scores=True,
        return_dict_in_generate=True,
    ):
        #
        # step 0. run image encoding with frozen vit model.
        #
        image_encoder_output = self.vit_model(pixel_values = pixel_values)
        #
        # step 1. run fusion encoding.
        #
        encoder_hidden_states, _ = self.mlm_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_state=image_encoder_output.last_hidden_state,
        )

        encoder_attention_mask = torch.ones(encoder_hidden_states.size()[:-1], dtype=torch.long).to(encoder_hidden_states.device)

        #
        # [batch_size, seq_length]
        #
        output = self.clm_model.generate(input_ids=torch.full((input_ids.size(0), 1), fill_value=self.tokenizer.cls_token_id, device=input_ids.device) if prompt_ids is None else prompt_ids,
                                        max_length=max_length,
                                        min_length=min_length,
                                        num_beams=num_beams,
                                        eos_token_id=self.tokenizer.sep_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        repetition_penalty=repetition_penalty,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=encoder_attention_mask,
                                        output_scores=output_scores,
                                        num_return_sequences=num_return_sequences,
                                        return_dict_in_generate=return_dict_in_generate,
                                        )

        return output


    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False


    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)


    #
    # dequeue, enqueue.
    #
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_cls_state, text_cls_state):
        batch_size = image_cls_state.shape[0]
        ptr = int(self.queue_ptr)
        #
        # for simplicity
        #
        assert self.queue_size % batch_size == 0
        #
        # replace the keys at ptr (dequeue and enqueue)
        #
        self.queue_image[:, ptr:ptr + batch_size] = image_cls_state.T
        self.queue_text[:, ptr:ptr + batch_size] = text_cls_state.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr



    #
    # token: [cls  a   b   c  sep pad pad] [cls  a   b   c   d   e   f   g  sep pad pad]
    # type:  [ 1   1   1   1   1   0   0 ] [ 1   1   1   1   1   1   1   1   1   0   0]
    # mask:  [ 1   1   1   1   1   0   0 ] [ 1   1   1   1   1   1   1   1   1   0   0]
    #
    #  ->
    #
    # token: [cls  a   b   c  sep  a   b   c   d   e  sep]
    # type:  [ 0   0   0   0   0   1   1   1   1   1   1 ]
    # mask:  [ 1   1   1   1   1   1   1   1   1   1   1 ]
    #
    def cross_match(self, weights_p2q, weights_q2p, input_ids, caption_ids, attention_mask, token_type_ids, image_hidden_state):
        input_ids_pos = []
        input_ids_neg = []

        attention_mask_pos = []
        attention_mask_neg = []

        token_type_ids_pos = []
        token_type_ids_neg = []

        image_hidden_state_neg = []

        batch_size, _ = input_ids.size()

        for i in range(batch_size):
            #
            # positive
            #
            input_ids_, attention_mask_, token_type_ids_ = self.cross_tokens(caption_ids[i], input_ids[i])

            input_ids_pos.append(input_ids_)
            attention_mask_pos.append(attention_mask_)
            token_type_ids_pos.append(token_type_ids_)

            #
            # in-batch hard negative sampling.
            #
            j = torch.multinomial(weights_p2q[i], 1).item()

            input_ids_, attention_mask_, token_type_ids_ = self.cross_tokens(caption_ids[j], input_ids[i])

            input_ids_neg.append(input_ids_)
            attention_mask_neg.append(attention_mask_)
            token_type_ids_neg.append(token_type_ids_)
            image_hidden_state_neg.append(image_hidden_state[i])

            #
            # in-batch hard negative sampling.
            #
            j = torch.multinomial(weights_q2p[i], 1).item()

            input_ids_, attention_mask_, token_type_ids_ = self.cross_tokens(caption_ids[i], input_ids[j])

            input_ids_neg.append(input_ids_)
            attention_mask_neg.append(attention_mask_)
            token_type_ids_neg.append(token_type_ids_)
            image_hidden_state_neg.append(image_hidden_state[j])

        #
        # positive input
        #
        input_ids_pos = torch.stack(input_ids_pos, dim=0)
        attention_mask_pos = torch.stack(attention_mask_pos, dim=0)
        token_type_ids_pos = torch.stack(token_type_ids_pos, dim=0)
        #
        # positive forward.
        #
        _, pooler_output_pos = self.mlm_model.bert(
            input_ids_pos,
            attention_mask=attention_mask_pos,
            token_type_ids=token_type_ids_pos,
            encoder_hidden_state=image_hidden_state,
        )

        #
        # negative input.
        #
        input_ids_neg = torch.stack(input_ids_neg, dim=0)
        attention_mask_neg = torch.stack(attention_mask_neg, dim=0)
        token_type_ids_neg = torch.stack(token_type_ids_neg, dim=0)
        image_hidden_state_neg = torch.stack(image_hidden_state_neg, dim=0)
        #
        # negative forward.
        #
        _, pooler_output_neg = self.mlm_model.bert(
            input_ids_neg,
            attention_mask=attention_mask_neg,
            token_type_ids=token_type_ids_neg,
            encoder_hidden_state=image_hidden_state_neg,
        )
        #
        # positive and negative, [batch * 3, 1, hidden]
        #
        hidden_state = torch.cat([pooler_output_pos, pooler_output_neg], dim=0)
        #
        # [batch * 3, 1, 2]
        #
        logit = self.itm_head(hidden_state)
        #
        # [batch * 3, 1]
        #
        label = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(batch_size * 2, dtype=torch.long)], dim=0).to(input_ids.device)

        return F.cross_entropy(logit, label, label_smoothing=0.01)




    #
    # token: [cls  a   b   c  sep pad pad] [cls  a   b   c   d   e   f   g  sep pad pad]
    # type:  [ 1   1   1   1   1   0   0 ] [ 1   1   1   1   1   1   1   1   1   0   0]
    # mask:  [ 1   1   1   1   1   0   0 ] [ 1   1   1   1   1   1   1   1   1   0   0]
    #
    #  ->
    #
    # token: [cls  a   b   c  sep  a   b   c   d   e  sep]
    # type:  [ 0   0   0   0   0   1   1   1   1   1   1 ]
    # mask:  [ 1   1   1   1   1   0   0   1   1   1   1 ]
    #
    def cross_tokens(self, query_token_ids, photo_token_ids):

        device = photo_token_ids.device

        length = photo_token_ids.size()[0]

        query_token_ids = query_token_ids[query_token_ids != self.tokenizer.pad_token_id]
        photo_token_ids = photo_token_ids[photo_token_ids != self.tokenizer.pad_token_id]

        query_token_ids = query_token_ids.tolist()[1:-1]
        photo_token_ids = photo_token_ids.tolist()[1:-1]

        query_length = len(query_token_ids)
        photo_length = len(photo_token_ids)

        if photo_length > length - query_length - 3:
            #
            # truncate photo length
            #
            photo_length = length - query_length - 3
            #
            # truncate photo tokens
            #
            photo_token_ids = photo_token_ids[:photo_length]

        #
        # padding token length.
        #
        padding_length = length - query_length - photo_length - 3

        input_ids = [self.tokenizer.cls_token_id] + query_token_ids + [self.tokenizer.sep_token_id] + photo_token_ids + [self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id] * padding_length

        attention_mask = [1] * (query_length + 2) + [1] * (photo_length + 1) + [0] * padding_length

        token_type_ids = [0] * (query_length + 2) + [1] * (photo_length + 1) + [0] * padding_length

        return torch.LongTensor(input_ids).to(device), torch.LongTensor(attention_mask).to(device), torch.LongTensor(token_type_ids).to(device)




#
# Perplexity Model.
#
class AlbefPPLModel(nn.Module):
    def __init__(self, tokenizer, mlm_config=None, vit_config=None, clm_config=None, mlm_model=None, vit_model=None, clm_model=None):
        super().__init__()

        self.tokenizer = tokenizer

        if mlm_model is not None:
            self.mlm_model = BertMLMModel.from_pretrained(mlm_model, ignore_mismatched_sizes=True)
            self.mlm_config = self.mlm_model.config
        else:
            self.mlm_model = BertMLMModel(config=mlm_config)
            self.mlm_config = mlm_config

        if vit_model is not None:
            self.vit_model = ViTModel.from_pretrained(vit_model, ignore_mismatched_sizes=True)
            self.vit_config = self.vit_model.config
        else:
            self.vit_model = ViTModel(config=vit_config)
            self.vit_config = vit_config

        if clm_model is not None:
            self.clm_model = BertCLMModel.from_pretrained(clm_model, ignore_mismatched_sizes=True)
            self.clm_config = self.clm_config.config
        else:
            self.clm_model = BertCLMModel(config=clm_config)
            self.clm_config = clm_config

        #
        # image text matching head.
        #
        self.itm_head = nn.Linear(self.mlm_config.hidden_size, 2)

    #
    # load a CLM model.
    #
    def load(self, model_file, ignore_queue=True):
        params = torch.load(model_file, map_location='cpu')
        states = {}
        for key, value in params.items():
            if 'queue' in key or 'mlm_model_m' in key or 'vit_model_m' in key or 'clm_model_m' in key:
                continue
            states[key] = value
        self.load_state_dict(states, strict=False)



    #
    # generation with beam search.
    #
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        caption_ids=None,
        caption_mask=None,
        caption_type=None,
    ):
        #
        # step 0. run image encoding with frozen vit model.
        #
        image_encoder_output = self.vit_model(pixel_values = pixel_values)
        #
        # step 1. run fusion encoding.
        #
        encoder_hidden_states, _ = self.mlm_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_state=image_encoder_output.last_hidden_state,
        )
        #
        # step 2. run causal language modeling.
        #
        loss = self.clm_model(
            input_ids=caption_ids,
            attention_mask=caption_mask,
            token_type_ids=caption_type,
            encoder_hidden_states=encoder_hidden_states,
            label=caption_ids.masked_fill(caption_ids==self.tokenizer.pad_token_id, -100),
            reduction='none',
        ).loss
        #
        # ppl = exp( -1.0 / N * sum( logp(y|x) ) ) = exp( loss / N )
        #
        # [batch_size, seq_length - 1] -> [batch_size]
        #
        score = torch.sum(loss.view(input_ids.shape[0], -1), dim=-1) / torch.sum(caption_mask[:, 1:], dim=-1)
        #
        # [batch_size], NOTE: this is the generation probability, NOT the real perplexity.
        #
        ppl = torch.exp(score)

        return ppl


    #
    # generation with beam search.
    #
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        prompt_ids=None,
        num_beams=3,
        max_length=16,
        min_length=1,
        repetition_penalty=1.0,
        num_return_sequences=3,
        output_scores=True,
        return_dict_in_generate=True,
    ):
        #
        # step 0. run image encoding with frozen vit model.
        #
        image_encoder_output = self.vit_model(pixel_values = pixel_values)
        #
        # step 1. run fusion encoding.
        #
        encoder_hidden_states, _ = self.mlm_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_state=image_encoder_output.last_hidden_state,
        )

        encoder_attention_mask = torch.ones(encoder_hidden_states.size()[:-1], dtype=torch.long).to(encoder_hidden_states.device)

        #
        # [batch_size, seq_length]
        #
        output = self.clm_model.generate(input_ids=torch.full((input_ids.size(0), 1), fill_value=self.tokenizer.cls_token_id, device=input_ids.device) if prompt_ids is None else prompt_ids,
                                        max_length=max_length,
                                        min_length=min_length,
                                        num_beams=num_beams,
                                        eos_token_id=self.tokenizer.sep_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        repetition_penalty=repetition_penalty,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=encoder_attention_mask,
                                        output_scores=output_scores,
                                        num_return_sequences=num_return_sequences,
                                        return_dict_in_generate=return_dict_in_generate,
                                        )

        return output




#
# Cross Match Modeling: Image-Text-Contrastive, Image-Text-Match, Masked-Language-Model, Cross-Match.
#
class AlbefCMMModel(nn.Module):
    def __init__(self, tokenizer, bert_config=None, vit_config=None, mlm_model=None, vit_model=None, vit_frozen=True, queue_size=65536):
        super().__init__()

        self.tokenizer = tokenizer

        if mlm_model is not None:
            self.mlm_model = BertMLMModel.from_pretrained(mlm_model, ignore_mismatched_sizes=True)
            self.bert_config = self.mlm_model.config
        else:
            self.mlm_model = BertMLMModel(config=bert_config)
            self.bert_config = bert_config

        self.mlm_model_m = BertMLMModel(config=self.bert_config)

        if vit_model is not None:
            self.vit_model = ViTModel.from_pretrained(vit_model, ignore_mismatched_sizes=True)
            self.vit_config = self.vit_model.config
        else:
            self.vit_model = ViTModel(config=vit_config)
            self.vit_config = vit_config

        self.vit_model_m = ViTModel(config=self.vit_config)

        #
        # frozen vit model.
        #
        self.vit_frozen = vit_frozen
        if self.vit_frozen:
            for param in self.vit_model.parameters():
                param.requires_grad = False

        #
        # momentum contrastive + momentum distilling.
        #
        self.model_pairs = [[self.mlm_model, self.mlm_model_m], [self.vit_model, self.vit_model_m]]
        self.copy_params()

        self.queue_size = queue_size
        self.register_buffer("queue_image", torch.randn(self.bert_config.hidden_size, self.queue_size))
        self.register_buffer("queue_text", torch.randn(self.bert_config.hidden_size, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.queue_image = F.normalize(self.queue_image, dim=0)
        self.queue_text = F.normalize(self.queue_text, dim=0)

        #
        # a relatively large momentum (e.g., m = 0.999, our default) works much better than a smaller value (e.g., m = 0.9), suggesting that a slowly evolving key encoder is a core to making use of a queue.
        #
        self.momentum = 0.999

        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)

        self.itm_head = nn.Linear(self.bert_config.hidden_size, 2)

        #
        # save cross attention for consistency constraint.
        #
        for i in range(self.bert_config.fusion_layer, self.bert_config.num_hidden_layers):
            self.mlm_model.bert.encoder.layer[i].crossattention.self.save_attention = True



    #
    # load a CMM model or a CLM model.
    #
    def load(self, model_file, ignore_queue=True):
        if ignore_queue:
            params = torch.load(model_file, map_location='cpu')
            states = {}
            for key, value in params.items():
                if "queue" in key:
                    continue
                states[key] = value
            self.load_state_dict(states, strict=False)
        else:
            self.load_state_dict(torch.load(model_file, map_location='cpu'))

        self.copy_params()



    #
    # pixel values: [batch * 3, chanel, hight, width]
    #
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        alpha=0,
        masked_inputs=None,
        masked_attention=None,
        masked_labels=None,
        caption_ids=None,
        caption_mask=None,
        caption_type=None,
        weight=None,
    ):
        with torch.no_grad():
            self.temperature.clamp_(0.001, 0.5)
        #
        # extract the cls tensor, NOT the pooler output, because MLM model has no pooler layer.
        #
        if self.vit_frozen:
            with torch.no_grad():
                image_hidden_state = self.vit_model(pixel_values=pixel_values).last_hidden_state
        else:
            image_hidden_state = self.vit_model(pixel_values=pixel_values).last_hidden_state

        #
        # normalize L2
        #
        image_cls_state = F.normalize(image_hidden_state[:, 0, :], dim=-1)

        text_hidden_state, _ = self.mlm_model.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mode='low')
        text_cls_state = F.normalize(text_hidden_state[:, 0, :], dim=-1)


        with torch.no_grad():
            batch_size = pixel_values.size(0)

        #
        # ================= ITC ========================
        #
        # momentum ITC
        #
        with torch.no_grad():
            self._momentum_update()
            #
            # rum model forward to get image cls.
            #
            image_hidden_state_m = self.vit_model_m(pixel_values=pixel_values).last_hidden_state

        #
        # normalize, L2.
        #
        image_cls_state_m = F.normalize(image_hidden_state_m[:, 0, :], dim=-1)
        #
        #
        #
        with torch.no_grad():
            #
            # [ batch_size + queue_size ]
            #
            image_cls_state_all = torch.cat([image_cls_state_m.t(), self.queue_image.clone().detach()], dim=1)
            #
            # run model forward to get text cls.
            #
            text_hidden_state_m, _ = self.mlm_model_m.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mode='low')
            text_cls_state_m = F.normalize(text_hidden_state_m[:, 0, :], dim=-1)
            #
            # [ batch_size + queue_size ]
            #
            text_cls_state_all = torch.cat([text_cls_state_m.t(), self.queue_text.clone().detach()], dim=1)
            #
            # [ batch_size, batch_size + queue_size ]
            #
            sim_i2t_m = image_cls_state_m @ text_cls_state_all / self.temperature
            sim_t2i_m = text_cls_state_m @ image_cls_state_all / self.temperature
            #
            # [ batch_size, batch_size + queue_size ]
            #
            sim_targets = torch.zeros(sim_i2t_m.size()).to(pixel_values.device)

            sim_targets[:, 0 : batch_size].fill_diagonal_(1)

            #
            # [ batch_size, batch_size + queue_size ]
            #
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        #
        # [ batch_size, batch_size + queue_size ]
        #
        sim_i2t = image_cls_state @ text_cls_state_all / self.temperature
        sim_t2i = text_cls_state @ image_cls_state_all / self.temperature
        #
        # in-world InfoNCE.
        #
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t+loss_t2i)/2

        #
        # update momentum queue.
        #
        self._dequeue_and_enqueue(image_cls_state_m, text_cls_state_m)

        #
        # ================= ITM ========================
        #
        # in-batch ITM
        #
        _, fusion_pooler_output_pos = self.mlm_model.bert(
            hidden_state = text_hidden_state,
            attention_mask = attention_mask,
            encoder_hidden_state = image_hidden_state,
            mode = 'top',
        )
        fusion_pooler_output_pos = self.dropout(fusion_pooler_output_pos)
        #
        #
        #
        with torch.no_grad():
            weights_i2t = sim_i2t[:, : batch_size].clone()
            weights_t2i = sim_t2i[:, : batch_size].clone()

            weights_i2t[:, 0 : batch_size].fill_diagonal_(-10000.0)
            weights_t2i[:, 0 : batch_size].fill_diagonal_(-10000.0)

            weights_i2t = F.softmax(weights_i2t, dim=1)
            weights_t2i = F.softmax(weights_t2i, dim=1)

            weights_i2t[:, 0 : batch_size].fill_diagonal_(0)
            weights_t2i[:, 0 : batch_size].fill_diagonal_(0)

        #
        # [ batch_size, length, hidden_size ]
        #
        image_hidden_state_neg = []

        for b in range(batch_size):
            #
            # in-world hard negative sampling.
            #
            idx = torch.multinomial(weights_t2i[b], 1).item()
            image_hidden_state_neg.append(image_hidden_state[idx])

        #
        # [ batch_size, length, hidden_size ]
        #
        image_hidden_state_neg = torch.stack(image_hidden_state_neg, dim=0)

        text_hidden_state_neg = []
        attention_mask_neg = []

        for b in range(batch_size):
            #
            # in-world hard negative sampling.
            #
            idx = torch.multinomial(weights_i2t[b], 1).item()
            text_hidden_state_neg.append(text_hidden_state[idx])
            attention_mask_neg.append(attention_mask[idx])
        #
        # [ batch_size, length, hidden_size ]
        #
        text_hidden_state_neg = torch.stack(text_hidden_state_neg, dim=0)
        attention_mask_neg = torch.stack(attention_mask_neg, dim=0)
        #
        # [ batch_size * 2, length, hidden_size ]
        #
        text_hidden_state_all = torch.cat([text_hidden_state, text_hidden_state_neg], dim=0)
        text_attention_mask_all = torch.cat([attention_mask, attention_mask_neg], dim=0)
        #
        # [ batch_size * 2, length, hidden_size ]
        #
        image_hidden_state_all = torch.cat([image_hidden_state_neg, image_hidden_state], dim=0)
        #
        # [ batch_size * 2, length, hidden_size ]
        #
        _, fusion_pooler_output_neg = self.mlm_model.bert(
            hidden_state = text_hidden_state_all,
            attention_mask = text_attention_mask_all,
            encoder_hidden_state = image_hidden_state_all,
            mode = 'top',
        )
        fusion_pooler_output_neg = self.dropout(fusion_pooler_output_neg)
        #
        # positive and negative, [batch * 3, 1, hidden]
        #
        vl_states = torch.cat([fusion_pooler_output_pos, fusion_pooler_output_neg], dim=0)
        #
        # [batch * 3, 1, 2]
        #
        vl_output = self.itm_head(vl_states)
        #
        # [batch * 3, 1]
        #
        itm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(batch_size * 2, dtype=torch.long)], dim=0).to(pixel_values.device)
        #
        # in-world cross entropy.
        #
        loss_itm = F.cross_entropy(vl_output, itm_labels, label_smoothing=0.01)

        #
        # ================= MLM ========================
        #
        loss_mlm = 0.0
        #
        # emperically, momentum distilling is not neccessary for CMM
        #
        if masked_inputs is not None and masked_labels is not None:
            with torch.no_grad():
                logit_m = self.mlm_model_m(
                    masked_inputs,
                    attention_mask=masked_attention,
                    token_type_ids=token_type_ids,
                    encoder_hidden_state=image_hidden_state_m,
                    return_logit=True,
                )

            loss_mlm = self.mlm_model(
                masked_inputs,
                attention_mask=masked_attention,
                token_type_ids=token_type_ids,
                encoder_hidden_state=image_hidden_state,
                label=masked_labels,
                label_soft=F.softmax(logit_m, dim=-1),
                alpha=alpha,
            )

        #
        # ============== CROSS MATCH ===================
        #
        # in-world cross match.
        #
        loss_cross = self.cross_match(weights_i2t, weights_t2i, input_ids, caption_ids, attention_mask, token_type_ids, image_hidden_state, weight)


        #
        # ============== Consistency ===================
        #
        loss_consistency = 0.0
        #
        # attention consistency for addressing cross-attention shift.
        #
        for i in range(self.bert_config.fusion_layer + 1, self.bert_config.num_hidden_layers):
            loss_consistency -= ((self.mlm_model.bert.encoder.layer[i].crossattention.self.get_attention_prob().mean(1) + 0.0000000001).log() * self.mlm_model.bert.encoder.layer[self.bert_config.fusion_layer].crossattention.self.get_attention_prob().mean(1).detach()).mean()


        #
        # ================== loss =======================
        #
        return loss_mlm, loss_ita, loss_itm, loss_cross, loss_consistency



    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False


    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)


    #
    # dequeue, enqueue.
    #
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_cls_state, text_cls_state):
        batch_size = image_cls_state.shape[0]
        ptr = int(self.queue_ptr)
        #
        # for simplicity
        #
        assert self.queue_size % batch_size == 0
        #
        # replace the keys at ptr (dequeue and enqueue)
        #
        self.queue_image[:, ptr:ptr + batch_size] = image_cls_state.T
        self.queue_text[:, ptr:ptr + batch_size] = text_cls_state.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr



    #
    # token: [cls  a   b   c  sep pad pad] [cls  a   b   c   d   e   f   g  sep pad pad]
    # type:  [ 1   1   1   1   1   0   0 ] [ 1   1   1   1   1   1   1   1   1   0   0]
    # mask:  [ 1   1   1   1   1   0   0 ] [ 1   1   1   1   1   1   1   1   1   0   0]
    #
    #  ->
    #
    # token: [cls  a   b   c  sep  a   b   c   d   e  sep]
    # type:  [ 0   0   0   0   0   1   1   1   1   1   1 ]
    # mask:  [ 1   1   1   1   1   1   1   1   1   1   1 ]
    #
    def cross_match(self, weights_p2q, weights_q2p, input_ids, caption_ids, attention_mask, token_type_ids, image_hidden_state, weight=None):
        input_ids_pos = []
        input_ids_neg = []

        attention_mask_pos = []
        attention_mask_neg = []

        token_type_ids_pos = []
        token_type_ids_neg = []

        image_hidden_state_neg = []

        batch_size, _ = input_ids.size()

        weight_pos = []
        weight_neg = []

        for i in range(batch_size):
            #
            # positive
            #
            input_ids_, attention_mask_, token_type_ids_ = self.cross_tokens(caption_ids[i], input_ids[i])

            input_ids_pos.append(input_ids_)
            attention_mask_pos.append(attention_mask_)
            token_type_ids_pos.append(token_type_ids_)

            if weight is not None:
                weight_pos.append(weight[i])

            #
            # in-batch hard negative sampling.
            #
            j = torch.multinomial(weights_p2q[i], 1).item()

            input_ids_, attention_mask_, token_type_ids_ = self.cross_tokens(caption_ids[j], input_ids[i])

            input_ids_neg.append(input_ids_)
            attention_mask_neg.append(attention_mask_)
            token_type_ids_neg.append(token_type_ids_)
            image_hidden_state_neg.append(image_hidden_state[i])

            if weight is not None:
                weight_neg.append(weight[i])

            #
            # in-batch hard negative sampling.
            #
            j = torch.multinomial(weights_q2p[i], 1).item()

            input_ids_, attention_mask_, token_type_ids_ = self.cross_tokens(caption_ids[i], input_ids[j])

            input_ids_neg.append(input_ids_)
            attention_mask_neg.append(attention_mask_)
            token_type_ids_neg.append(token_type_ids_)
            image_hidden_state_neg.append(image_hidden_state[j])

            if weight is not None:
                weight_neg.append(weight[j])

        #
        # positive input
        #
        input_ids_pos = torch.stack(input_ids_pos, dim=0)
        attention_mask_pos = torch.stack(attention_mask_pos, dim=0)
        token_type_ids_pos = torch.stack(token_type_ids_pos, dim=0)
        #
        # positive forward.
        #
        _, pooler_output_pos = self.mlm_model.bert(
            input_ids_pos,
            attention_mask=attention_mask_pos,
            token_type_ids=token_type_ids_pos,
            encoder_hidden_state=image_hidden_state,
        )
        pooler_output_pos = self.dropout(pooler_output_pos)
        #
        # negative input.
        #
        input_ids_neg = torch.stack(input_ids_neg, dim=0)
        attention_mask_neg = torch.stack(attention_mask_neg, dim=0)
        token_type_ids_neg = torch.stack(token_type_ids_neg, dim=0)
        image_hidden_state_neg = torch.stack(image_hidden_state_neg, dim=0)
        #
        # negative forward.
        #
        _, pooler_output_neg = self.mlm_model.bert(
            input_ids_neg,
            attention_mask=attention_mask_neg,
            token_type_ids=token_type_ids_neg,
            encoder_hidden_state=image_hidden_state_neg,
        )
        pooler_output_neg = self.dropout(pooler_output_neg)
        #
        # positive and negative, [batch * 3, 1, hidden]
        #
        hidden_state = torch.cat([pooler_output_pos, pooler_output_neg], dim=0)
        #
        # [batch * 3, 1, 2]
        #
        logit = self.itm_head(hidden_state)
        #
        #
        #
        device = input_ids.device
        #
        # [batch * 3, 1]
        #
        label = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(batch_size * 2, dtype=torch.long)], dim=0).to(device)
        #
        # [batch, 1]
        #
        if weight is not None:
            loss = F.cross_entropy(logit, label, label_smoothing=0.01, reduction='none') * torch.cat([torch.tensor(weight_pos, dtype=torch.float32), torch.tensor(weight_neg, dtype=torch.float32)]).to(device)
            return loss.mean()
        else:
            loss = F.cross_entropy(logit, label, label_smoothing=0.01)
            return loss



    #
    # token: [cls  a   b   c  sep pad pad] [cls  a   b   c   d   e   f   g  sep pad pad]
    # type:  [ 1   1   1   1   1   0   0 ] [ 1   1   1   1   1   1   1   1   1   0   0]
    # mask:  [ 1   1   1   1   1   0   0 ] [ 1   1   1   1   1   1   1   1   1   0   0]
    #
    #  ->
    #
    # token: [cls  a   b   c  sep  a   b   c   d   e  sep]
    # type:  [ 0   0   0   0   0   1   1   1   1   1   1 ]
    # mask:  [ 1   1   1   1   1   0   0   1   1   1   1 ]
    #
    def cross_tokens(self, query_token_ids, photo_token_ids):

        device = photo_token_ids.device

        length = photo_token_ids.size()[0]

        query_token_ids = query_token_ids[query_token_ids != self.tokenizer.pad_token_id]
        photo_token_ids = photo_token_ids[photo_token_ids != self.tokenizer.pad_token_id]

        query_token_ids = query_token_ids.tolist()[1:-1]
        photo_token_ids = photo_token_ids.tolist()[1:-1]

        query_length = len(query_token_ids)
        photo_length = len(photo_token_ids)

        if photo_length > length - query_length - 3:
            #
            # truncate photo length
            #
            photo_length = length - query_length - 3
            #
            # truncate photo tokens
            #
            photo_token_ids = photo_token_ids[:photo_length]

        #
        # padding token length.
        #
        padding_length = length - query_length - photo_length - 3

        input_ids = [self.tokenizer.cls_token_id] + query_token_ids + [self.tokenizer.sep_token_id] + photo_token_ids + [self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id] * padding_length

        attention_mask = [1] * (query_length + 2) + [1] * (photo_length + 1) + [0] * padding_length

        token_type_ids = [0] * (query_length + 2) + [1] * (photo_length + 1) + [0] * padding_length

        return torch.LongTensor(input_ids).to(device), torch.LongTensor(attention_mask).to(device), torch.LongTensor(token_type_ids).to(device)







#
# Relevance(query, photo, image)
#
class AlbefRELModel_ORDINAL_V4(nn.Module):
    def __init__(self, bert_config=None, vit_config=None, bert_model=None, vit_model=None, vit_frozen=True):
        super().__init__()

        if bert_model is not None:
            self.bert_model = BertModel.from_pretrained(bert_model, ignore_mismatched_sizes=True)
            self.bert_config = self.bert_model.config
        else:
            self.bert_model = BertModel(config=bert_config)
            self.bert_config = bert_config

        if vit_model is not None:
            self.vit_model = ViTModel.from_pretrained(vit_model, ignore_mismatched_sizes=True)
            self.vit_config = self.vit_model.config
        else:
            self.vit_model = ViTModel(config=vit_config)
            self.vit_config = vit_config

        #
        # frozen vit model.
        #
        self.vit_frozen = vit_frozen

        if self.vit_frozen:
            for param in self.vit_model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)

        #
        # hierachical logits - Left, Middle, Right.
        #
        self.head_L = nn.Linear(self.bert_config.hidden_size, 1)
        self.head_M = nn.Linear(self.bert_config.hidden_size, 1)
        self.head_R = nn.Linear(self.bert_config.hidden_size, 1)

        #
        # save cross attention for consistency constraint.
        #
        for i in range(self.bert_config.fusion_layer, self.bert_config.num_hidden_layers):
            self.bert_model.encoder.layer[i].crossattention.self.save_attention = True


    #
    # support MLM, CLM, CMM, REL.
    #
    def load(self, model_file):
        params = torch.load(model_file, map_location='cpu')

        states = {}

        for key, value in params.items():
            if "itm_head" in key:
                key = key.replace("itm_head", "head_M")
                if "weight" in key:
                    value = value[1, :].view(1, -1)
                if "bias" in key:
                    value = value[1].view(1)

            if "head_0" in key:
                key = key.replace("head_0", "head_L")

            if "head_1" in key:
                key = key.replace("head_1", "head_M")

            if "head_2" in key:
                key = key.replace("head_2", "head_R")

            if "mlm_model.bert" in key:
                key = key.replace("mlm_model.bert", "bert_model")

            states[key] = value

        self.load_state_dict(states, strict=False)



    def load_model(self, bert_model_file, vit_model_file):
        self.bert_model.load_state_dict(torch.load(bert_model_file, map_location='cpu'))
        self.vit_model.load_state_dict(torch.load(vit_model_file, map_location='cpu'))



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        label=None,
    ):
        #
        # step 0. run image encoding with frozen vit model.
        #
        if self.vit_frozen:
            with torch.no_grad():
                image_hidden_state = self.vit_model(pixel_values = pixel_values).last_hidden_state
        else:
            image_hidden_state = self.vit_model(pixel_values = pixel_values).last_hidden_state

        #
        # step 1. run fusion encoding.
        #
        _, fusion_pooler_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_state=image_hidden_state,
        )
        #
        # step 2. run dropout.
        #
        pooler_output = self.dropout(fusion_pooler_output)
        #
        # step 3. run hierachical logistical regression - Left, Middle(Root), Right.
        #
        logit_L = self.head_L(pooler_output)
        logit_M = self.head_M(pooler_output)
        logit_R = self.head_R(pooler_output)

        if label is None:
            return logit_L, logit_M, logit_R

        #
        # step 4. calculate loss.
        #
        P_L = torch.sigmoid(logit_L)
        P_M = torch.sigmoid(logit_M)
        P_R = torch.sigmoid(logit_R)

        P_1 = (1.0 - P_M) * P_L
        P_2 = P_M * (1.0 - P_R)
        P_3 = P_M * P_R
        #
        # ordinal regression, [batch, 1]
        #
        loss = F.binary_cross_entropy_with_logits(logit_L, (label>0.0).float(), weight=(label<=1.0).float()) + F.binary_cross_entropy_with_logits(logit_M, (label>1.0).float()) + F.binary_cross_entropy_with_logits(logit_R, (label>2.0).float(), weight=(label>=2.0).float()) + F.mse_loss(1.0 * P_1 + 2.0 * P_2 + 3.0 * P_3, label.float())
        #
        # consistency loss, [batch, head, text_length, image_length]
        #
        for i in range(self.bert_config.fusion_layer + 1, self.bert_config.num_hidden_layers):
            loss -= ((self.bert_model.encoder.layer[i].crossattention.self.get_attention_prob().mean(1) + 0.0000000001).log() * self.bert_model.encoder.layer[self.bert_config.fusion_layer].crossattention.self.get_attention_prob().mean(1).detach()).mean()

        return logit_L, logit_M, logit_R, loss






#
# Relevance(query, photo, image) -- SET PREDICTION
#
class AlbefRELModel_ORDINAL_V5(nn.Module):
    def __init__(self, bert_config=None, vit_config=None, bert_model=None, vit_model=None, vit_frozen=True):
        super().__init__()

        if bert_model is not None:
            self.bert_model = BertModel.from_pretrained(bert_model, ignore_mismatched_sizes=True)
            self.bert_config = self.bert_model.config
        else:
            self.bert_model = BertModel(config=bert_config)
            self.bert_config = bert_config

        if vit_model is not None:
            self.vit_model = ViTModel.from_pretrained(vit_model, ignore_mismatched_sizes=True)
            self.vit_config = self.vit_model.config
        else:
            self.vit_model = ViTModel(config=vit_config)
            self.vit_config = vit_config

        #
        # frozen vit model.
        #
        self.vit_frozen = vit_frozen

        if self.vit_frozen:
            for param in self.vit_model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        #
        # hierachical logits - Left, Middle, Right.
        #
        self.head_L = nn.Linear(self.bert_config.hidden_size, 1)
        self.head_M = nn.Linear(self.bert_config.hidden_size, 1)
        self.head_R = nn.Linear(self.bert_config.hidden_size, 1)
        #
        # set prediction dense
        #
        self.dense_1 = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.dense_2 = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.dense_3 = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        #
        # set prediction logits
        #
        self.head_1 = nn.Linear(self.bert_config.hidden_size, 1)
        self.head_2 = nn.Linear(self.bert_config.hidden_size, 1)
        self.head_3 = nn.Linear(self.bert_config.hidden_size, 1)
        #
        # save cross attention for consistency constraint.
        #
        for i in range(self.bert_config.fusion_layer, self.bert_config.num_hidden_layers):
            self.bert_model.encoder.layer[i].crossattention.self.save_attention = True


    #
    # support MLM, CLM, CMM, REL.
    #
    def load(self, model_file):
        params = torch.load(model_file, map_location='cpu')
        states = {}
        for key, value in params.items():
            if "mlm_model.bert" in key:
                key = key.replace("mlm_model.bert", "bert_model")
            states[key] = value
        self.load_state_dict(states, strict=False)



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        label=None,
    ):
        #
        # step 0. run image encoding with frozen vit model.
        #
        if self.vit_frozen:
            with torch.no_grad():
                image_hidden_state = self.vit_model(pixel_values = pixel_values).last_hidden_state
        else:
            image_hidden_state = self.vit_model(pixel_values = pixel_values).last_hidden_state

        #
        # step 1. run fusion encoding.
        #
        fusion_hidden_states, fusion_pooler_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_state=image_hidden_state,
        )
        #
        # step 2. get cls tensor.
        #
        pooler_output = self.dropout(fusion_pooler_output)
        #
        # step 3. run hierachical logits.
        #
        logit_L = self.head_L(pooler_output)
        logit_M = self.head_M(pooler_output)
        logit_R = self.head_R(pooler_output)
        #
        # step 4. get set prediction tensor.
        #
        pooler_output_1 = self.dropout(self.dense_1(fusion_hidden_states[:, 1, :]))
        pooler_output_2 = self.dropout(self.dense_2(fusion_hidden_states[:, 2, :]))
        pooler_output_3 = self.dropout(self.dense_3(fusion_hidden_states[:, 3, :]))
        #
        # step 5. get set prediction logits.
        #
        logit_1 = self.head_1(pooler_output_1)
        logit_2 = self.head_2(pooler_output_2)
        logit_3 = self.head_3(pooler_output_3)

        if label is None:
            return logit_L, logit_M, logit_R, logit_1, logit_2, logit_3

        #
        # step 6. get hierachical probability.
        #
        P_L = torch.sigmoid(logit_L)
        P_M = torch.sigmoid(logit_M)
        P_R = torch.sigmoid(logit_R)

        P_1 = (1.0 - P_M) * P_L
        P_2 = P_M * (1.0 - P_R)
        P_3 = P_M * P_R
        #
        # step 7. get set prediction probability.
        #
        S_1 = torch.sigmoid(logit_1)
        S_2 = torch.sigmoid(logit_2)
        S_3 = torch.sigmoid(logit_3)
        #
        # step 8. run label matching.
        #
        label_1, label_2, label_3 = self.label_match(S_1, S_2, S_3, label)
        #
        # ordinal regression, [batch, 1]
        #
        loss = F.binary_cross_entropy_with_logits(logit_L, (label>0.0).float(), weight=(label<=1.0).float()) + F.binary_cross_entropy_with_logits(logit_M, (label>1.0).float()) + F.binary_cross_entropy_with_logits(logit_R, (label>2.0).float(), weight=(label>=2.0).float()) + F.mse_loss(1.0 * P_1 + 2.0 * P_2 + 3.0 * P_3, label.float()) +  F.binary_cross_entropy_with_logits(logit_1, label_1) + F.binary_cross_entropy_with_logits(logit_2, label_2) + F.binary_cross_entropy_with_logits(logit_3, label_3) + F.mse_loss(S_1 + S_2 + S_3, label.float())
        #
        # consistency loss, [batch, head, text_length, image_length]
        #
        for i in range(self.bert_config.fusion_layer + 1, self.bert_config.num_hidden_layers):
            loss -= ((self.bert_model.encoder.layer[i].crossattention.self.get_attention_prob().mean(1) + 0.0000000001).log() * self.bert_model.encoder.layer[self.bert_config.fusion_layer].crossattention.self.get_attention_prob().mean(1).detach()).mean()

        return logit_L, logit_M, logit_R, logit_1, logit_2, logit_3, loss



    #
    # SET PREDICTION Label Matching
    #
    def label_match(self, score_1, score_2, score_3, label):
        #
        # [batch_size]
        #
        label_1 = []
        label_2 = []
        label_3 = []

        for s_1, s_2, s_3, l in zip(score_1.tolist(), score_2.tolist(), score_3.tolist(), label.tolist()):
            if l[0] == 0:
                label_1.append(0)
                label_2.append(0)
                label_3.append(0)
                continue

            if l[0] == 1:
                if s_1 >= s_2 and s_1 >= s_3:
                    label_1.append(1)
                    label_2.append(0)
                    label_3.append(0)
                    continue
                if s_2 >= s_1 and s_2 >= s_3:
                    label_1.append(0)
                    label_2.append(1)
                    label_3.append(0)
                    continue
                if s_3 >= s_1 and s_3 >= s_2:
                    label_1.append(0)
                    label_2.append(0)
                    label_3.append(1)
                    continue

            if l[0] == 2:
                if s_1 <= s_2 and s_1 <= s_3:
                    label_1.append(0)
                    label_2.append(1)
                    label_3.append(1)
                    continue
                if s_2 <= s_1 and s_2 <= s_3:
                    label_1.append(1)
                    label_2.append(0)
                    label_3.append(1)
                    continue
                if s_3 <= s_1 and s_3 <= s_2:
                    label_1.append(1)
                    label_2.append(1)
                    label_3.append(0)
                    continue

            if l[0] == 3:
                label_1.append(1)
                label_2.append(1)
                label_3.append(1)
                continue

        return torch.FloatTensor(label_1).view(-1, 1).to(label.device), torch.FloatTensor(label_2).view(-1, 1).to(label.device),  torch.FloatTensor(label_3).view(-1, 1).to(label.device)





#
# weighted momentum distilling.
#
class AlbefCLKModel(nn.Module):
    def __init__(self, bert_config=None, vit_config=None, bert_model=None, vit_model=None, vit_frozen=True):
        super().__init__()

        if bert_model is not None:
            self.bert_model = BertModel.from_pretrained(bert_model, ignore_mismatched_sizes=True)
            self.bert_config = self.bert_model.config
        else:
            self.bert_model = BertModel(config=bert_config)
            self.bert_config = bert_config

        self.bert_model_m = BertModel(config=self.bert_config)

        if vit_model is not None:
            self.vit_model = ViTModel.from_pretrained(vit_model, ignore_mismatched_sizes=True)
            self.vit_config = self.vit_model.config
        else:
            self.vit_model = ViTModel(config=vit_config)
            self.vit_config = vit_config

        self.vit_model_m = ViTModel(config=self.vit_config)

        assert self.bert_config.hidden_size == self.vit_config.hidden_size

        #
        # frozen vit model.
        #
        self.vit_frozen = vit_frozen

        if self.vit_frozen:
            for param in self.vit_model.parameters():
                param.requires_grad = False


        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.dropout_m = nn.Dropout(self.bert_config.hidden_dropout_prob)

        self.head = nn.Linear(self.bert_config.hidden_size, 1)
        self.head_m = nn.Linear(self.bert_config.hidden_size, 1)

        self.model_pairs = [[self.bert_model, self.bert_model_m], [self.vit_model, self.vit_model_m], [self.dropout, self.dropout_m], [self.head, self.head_m]]
        self.copy_params()
        #
        # a relatively large momentum (e.g., m = 0.999, our default) works much better than a smaller value (e.g., m = 0.9), suggesting that a slowly evolving key encoder is a core to making use of a queue.
        #
        self.momentum = 0.9999


    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location='cpu'))
        self.copy_params()


    def load_model(self, bert_model_file, vit_model_file):
        self.bert_model.load_state_dict(torch.load(bert_model_file, map_location='cpu'))
        self.vit_model.load_state_dict(torch.load(vit_model_file, map_location='cpu'))
        self.copy_params()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        label=None,
    ):
        #
        # step 0. run image encoding with frozen vit model.
        #
        if self.vit_frozen:
            with torch.no_grad():
                image_hidden_state = self.vit_model(pixel_values = pixel_values).last_hidden_state
        else:
            image_hidden_state = self.vit_model(pixel_values = pixel_values).last_hidden_state

        #
        # step 1. run fusion encoding.
        #
        _, fusion_pooler_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_state=image_hidden_state,
        )
        #
        # step 2. run dropout.
        #
        pooler_output = self.dropout(fusion_pooler_output)
        #
        # step 3. run binary classification.
        #
        logit = self.head(pooler_output)
        #
        # step 4. momentum distilling loss.
        #
        with torch.no_grad():
            #
            # update momentum model.
            #
            self._momentum_update()
            #
            # run momentum prediction.
            #
            image_hidden_state_m = self.vit_model_m(pixel_values = pixel_values).last_hidden_state
            _, fusion_pooler_output_m = self.bert_model_m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                encoder_hidden_state=image_hidden_state_m,
            )
            pooler_output_m = self.dropout_m(fusion_pooler_output_m)
            logit_m = self.head_m(pooler_output_m)
            score_m = torch.sigmoid(logit_m)

        loss = F.binary_cross_entropy_with_logits(input=logit, target=score_m, weight=label)

        return loss



    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False



    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)




#
# weighted momentum distilling.
#
class AlbefCLKModel_ORDINAL(nn.Module):
    def __init__(self, bert_config=None, vit_config=None, bert_model=None, vit_model=None, vit_frozen=True):
        super().__init__()

        if bert_model is not None:
            self.bert_model = BertModel.from_pretrained(bert_model, ignore_mismatched_sizes=True)
            self.bert_config = self.bert_model.config
        else:
            self.bert_model = BertModel(config=bert_config)
            self.bert_config = bert_config

        self.bert_model_m = BertModel(config=self.bert_config)

        if vit_model is not None:
            self.vit_model = ViTModel.from_pretrained(vit_model, ignore_mismatched_sizes=True)
            self.vit_config = self.vit_model.config
        else:
            self.vit_model = ViTModel(config=vit_config)
            self.vit_config = vit_config

        self.vit_model_m = ViTModel(config=self.vit_config)

        assert self.bert_config.hidden_size == self.vit_config.hidden_size

        #
        # frozen vit model.
        #
        self.vit_frozen = vit_frozen

        if self.vit_frozen:
            for param in self.vit_model.parameters():
                param.requires_grad = False


        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.dropout_m = nn.Dropout(self.bert_config.hidden_dropout_prob)

        #
        # hierachical logits - Left, Middle, Right.
        #
        self.head_L = nn.Linear(self.bert_config.hidden_size, 1)
        self.head_M = nn.Linear(self.bert_config.hidden_size, 1)
        self.head_R = nn.Linear(self.bert_config.hidden_size, 1)

        self.head_L_m = nn.Linear(self.bert_config.hidden_size, 1)
        self.head_M_m = nn.Linear(self.bert_config.hidden_size, 1)
        self.head_R_m = nn.Linear(self.bert_config.hidden_size, 1)

        self.model_pairs = [[self.bert_model, self.bert_model_m], [self.vit_model, self.vit_model_m], [self.dropout, self.dropout_m], [self.head_L, self.head_L_m], [self.head_M, self.head_M_m], [self.head_R, self.head_R_m]]
        self.copy_params()

        #
        # a relatively large momentum (e.g., m = 0.999, our default) works much better than a smaller value (e.g., m = 0.9), suggesting that a slowly evolving key encoder is a core to making use of a queue.
        #
        self.momentum = 0.9999


    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location='cpu'), strict=False)
        self.copy_params()


    def load_model(self, bert_model_file, vit_model_file):
        self.bert_model.load_state_dict(torch.load(bert_model_file, map_location='cpu'))
        self.vit_model.load_state_dict(torch.load(vit_model_file, map_location='cpu'))
        self.copy_params()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        label=None,
    ):
        #
        # step 0. run image encoding with frozen vit model.
        #
        if self.vit_frozen:
            with torch.no_grad():
                image_hidden_state = self.vit_model(pixel_values = pixel_values).last_hidden_state
        else:
            image_hidden_state = self.vit_model(pixel_values = pixel_values).last_hidden_state

        #
        # step 1. run fusion encoding.
        #
        _, fusion_pooler_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_state=image_hidden_state,
        )
        #
        # step 2. run dropout.
        #
        pooler_output = self.dropout(fusion_pooler_output)
        #
        # step 3. run hierachical logistical regression - Left, Middle, Right.
        #
        logit_L = self.head_L(pooler_output)
        logit_M = self.head_M(pooler_output)
        logit_R = self.head_R(pooler_output)
        #
        # step 4. momentum distilling loss.
        #
        with torch.no_grad():
            #
            # update momentum model.
            #
            self._momentum_update()
            #
            # run momentum prediction.
            #
            image_hidden_state_m = self.vit_model_m(pixel_values = pixel_values).last_hidden_state
            _, fusion_pooler_output_m = self.bert_model_m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                encoder_hidden_state=image_hidden_state_m,
            )
            pooler_output_m = self.dropout_m(fusion_pooler_output_m)

            logit_L_m = self.head_L_m(pooler_output_m)
            logit_M_m = self.head_M_m(pooler_output_m)
            logit_R_m = self.head_R_m(pooler_output_m)

        loss = F.binary_cross_entropy_with_logits(input=logit_L, target=torch.sigmoid(logit_L_m), weight=label) + F.binary_cross_entropy_with_logits(input=logit_M, target=torch.sigmoid(logit_M_m), weight=label) + F.binary_cross_entropy_with_logits(input=logit_R, target=torch.sigmoid(logit_R_m), weight=label)

        return loss



    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False



    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)





#
# Image-Text-Generation.
#
class AlbefVQAModel(nn.Module):
    def __init__(self, tokenizer, bert_config=None, vit_config=None, clm_config=None, bert_model=None, vit_model=None, clm_model=None):
        super().__init__()

        self.tokenizer = tokenizer

        if bert_model is not None:
            self.bert_model = BertModel.from_pretrained(bert_model, ignore_mismatched_sizes=True)
            self.bert_config = self.bert_model.config
        else:
            self.bert_model = BertModel(config=bert_config)
            self.bert_config = bert_config

        if vit_model is not None:
            self.vit_model = ViTModel.from_pretrained(vit_model, ignore_mismatched_sizes=True)
            self.vit_config = self.vit_model.config
        else:
            self.vit_model = ViTModel(config=vit_config)
            self.vit_config = vit_config

        if clm_model is not None:
            self.clm_model = BertCLMModel.from_pretrained(clm_model, ignore_mismatched_sizes=True)
            self.clm_config = self.clm_config.config
        else:
            self.clm_model = BertCLMModel(config=clm_config)
            self.clm_config = clm_config

        assert self.bert_config.hidden_size == self.vit_config.hidden_size
        assert self.bert_config.hidden_size == self.clm_config.hidden_size


    #
    # load the whole model.
    #
    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location='cpu'))


    #
    # load component models.
    #
    def load_model(self, bert_model_file=None, vit_model_file=None, clm_model_file=None):
        if bert_model_file is not None:
            self.bert_model.load_state_dict(torch.load(bert_model_file, map_location='cpu'))
        if vit_model_file is not None:
            self.vit_model.load_state_dict(torch.load(vit_model_file, map_location='cpu'))
        if clm_model_file is not None:
            self.clm_model.load_state_dict(torch.load(clm_model_file, map_location='cpu'))


    def forward(
        self,
        question_id=None,
        question_mask=None,
        question_type=None,
        answer_id=None,
        answer_mask=None,
        answer_type=None,
        pixel_values=None,
        weight=None,
    ):
        #
        # step 0. run image encoding with frozen vit model.
        #
        image_encoder_output = self.vit_model(pixel_values = pixel_values)
        #
        # step 1. run fusion encoding.
        #
        encoder_hidden_states, _ = self.bert_model(
            input_ids=question_id,
            attention_mask=question_mask,
            token_type_ids=question_type,
            encoder_hidden_state=image_encoder_output.last_hidden_state,
        )
        #
        # step 2. run causal language modeling.
        #
        loss = self.clm_model(
            input_ids=answer_id,
            attention_mask=answer_mask,
            token_type_ids=answer_type,
            encoder_hidden_states=encoder_hidden_states,
            label=answer_id.masked_fill(answer_id == self.tokenizer.pad_token_id, -100),
            weight=weight,
        ).loss

        return loss


    #
    # generation with beam search.
    #
    @torch.no_grad()
    def generate(
        self,
        question_id=None,
        question_mask=None,
        question_type=None,
        pixel_values=None,
        prompt_ids=None,
        num_beams=3,
        max_length=16,
        min_length=1,
        repetition_penalty=1.0,
        num_return_sequences=3,
        output_scores=True,
        return_dict_in_generate=True,
    ):
        #
        # step 0. run image encoding with frozen vit model.
        #
        image_encoder_output = self.vit_model(pixel_values = pixel_values)
        #
        # step 1. run fusion encoding.
        #
        encoder_hidden_states, _ = self.bert_model(
            input_ids=question_id,
            attention_mask=question_mask,
            token_type_ids=question_type,
            encoder_hidden_state=image_encoder_output.last_hidden_state,
        )

        encoder_attention_mask = torch.ones(encoder_hidden_states.size()[:-1], dtype=torch.long).to(encoder_hidden_states.device)

        #
        # [batch_size, seq_length]
        #
        output = self.clm_model.generate(input_ids=torch.full((input_ids.size(0), 1), fill_value=self.tokenizer.cls_token_id, device=input_ids.device) if prompt_ids is None else prompt_ids,
                                        max_length=max_length,
                                        min_length=min_length,
                                        num_beams=num_beams,
                                        eos_token_id=self.tokenizer.sep_token_id,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        repetition_penalty=repetition_penalty,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=encoder_attention_mask,
                                        output_scores=output_scores,
                                        num_return_sequences=num_return_sequences,
                                        return_dict_in_generate=return_dict_in_generate,
                                        )

        return output




#
# Fast Gradient Method.
#
class FGM():
    def __init__(self, model, embedding_name='embeddings.word_embeddings.weight'):
        self.model = model
        self.param = dict()
        self.embedding_name = embedding_name

    #
    # attack word embedding.
    #
    def attack(self, epsilon=0.5):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embedding_name in name:
                self.param[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = epsilon * param.grad / norm
                    param.data.add_(r_adv)

    #
    # restore word embedding.
    #
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embedding_name in name:
                assert name in self.param
                param.data = self.param[name]
        self.param = dict()



#
# 0-1 distribution KL loss.
#
def binary_kl_loss(p, q, reduction='none', sigmoid_target=False):
    #
    # convert logits to sigmoid probability.
    #
    if not sigmoid_target:
        p = p.sigmoid()
        q = q.sigmoid()

    p = torch.concat([1 - p, p], axis=1)
    q = torch.concat([1 - q, q], axis=1)

    p_loss = F.kl_div(p.log(), q, reduction=reduction).sum()
    q_loss = F.kl_div(q.log(), p, reduction=reduction).sum()

    return (p_loss + q_loss) / 2

