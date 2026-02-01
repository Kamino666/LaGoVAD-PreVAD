import torch.nn as nn
import torch

from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from typing import Union, Tuple, Optional


# copy from https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/clip/modeling_clip.py
class SoftPromptCLIPTextModel(nn.Module):
    def __init__(self,
                 model_name_or_path='openai/clip-vit-base-patch32',
                 num_soft_prompts=16):
        super().__init__()
        self.num_soft_prompts = num_soft_prompts
        self.model = CLIPTextModel.from_pretrained(model_name_or_path)
        self.config = self.model.config
        if num_soft_prompts > 0:
            self.prompt_embedding = nn.Embedding(num_soft_prompts, self.config.hidden_size)
        else:
            self.prompt_embedding = None

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            use_soft_prompt=True,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        m = self.model.text_model

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        if self.num_soft_prompts == 0 or self.prompt_embedding is None or use_soft_prompt is False:  # Vanilla
            hidden_states = m.embeddings(input_ids=input_ids, position_ids=position_ids)
        else:  # Learnable prompt
            # =====Mods START=====
            # 把前后的learnable prompt放到插入到文本输入中，形成新的hidden_states, input_ids和attention_mask
            input_lengths = input_ids.argmax(dim=-1) - 1  # B
            pre_num = self.num_soft_prompts // 2
            post_num = self.num_soft_prompts // 2

            soft_ids = torch.arange(self.num_soft_prompts, device=input_ids.device)
            soft_ids = soft_ids.unsqueeze(0).expand(input_ids.shape[0], -1)  # B,L
            soft_embed = self.prompt_embedding(soft_ids)  # B,L,E

            expand_input_ids = torch.empty(
                input_ids.shape[0],
                self.num_soft_prompts + input_ids.shape[1],
                dtype=input_ids.dtype, device=input_ids.device
            ).fill_(input_ids.max())  # B,L+S
            expand_input_ids[:, 0] = input_ids[:, 0]
            expand_input_ids[:, 1 + pre_num: 1 + pre_num + input_ids.shape[1] - 1] = input_ids[:, 1:]
            mod_hidden_states = m.embeddings(input_ids=expand_input_ids)  # B,L+S,E
            mod_hidden_states[:, 1: 1 + pre_num] = soft_embed[:, :pre_num]
            expand_input_ids[:, 1: 1 + pre_num] = -1
            expand_attention_mask = attention_mask.new_zeros(expand_input_ids.shape[0], expand_input_ids.shape[1])
            for bi in range(input_shape[0]):
                mod_hidden_states[bi, 1 + pre_num + input_lengths[bi]: 1 + pre_num + input_lengths[bi] + post_num] = soft_embed[bi,-post_num:]
                expand_input_ids[bi, 1 + pre_num + input_lengths[bi]: 1 + pre_num + input_lengths[bi] + post_num] = -1
                expand_attention_mask[bi, :expand_input_ids[bi].argmax() + 1] = 1

            hidden_states = mod_hidden_states
            attention_mask = expand_attention_mask
            input_ids = expand_input_ids
            input_shape = hidden_states.shape[:-1]
            # =====Mods END=====

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None and not m._use_flash_attention_2:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = m.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = m.final_layer_norm(last_hidden_state)

        if m.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to
                # `eos_token_id`) Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (
                # e.g. prepared by the tokenizer)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == m.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
