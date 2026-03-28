#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.append(".") # Adds higher directory to python modules path.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.global_step_counter = 0
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_ids_cd: torch.LongTensor = None,
        tokenizer=None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_cd: Optional[torch.FloatTensor] = None,
        images_cd_lr: Optional[torch.FloatTensor] = None,
        images_cd_ud: Optional[torch.FloatTensor] = None,
        cd_beta: Optional[torch.FloatTensor] = None,
        cd_alpha: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        # 新增参数：用于多层attention head masking
        use_rel: Optional[bool] = False,
        layer_heads_config: Optional[dict] = None,
        relation_word_positions: Optional[dict] = None,
        return_extra_attentions: bool = False,
    ) -> Union[Tuple, Tuple[CausalLMOutputWithPast, Optional[Tuple], Optional[Tuple]]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        if not use_rel:
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            # 当 use_rel=False 时，不传递额外参数，以兼容标准 transformers 版本
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        else:
            # 处理多层配置：如果提供了layer_heads_config，优先使用；否则使用向后兼容的参数
            if layer_heads_config is not None:
                # 使用新的多层配置
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    use_rel=use_rel,
                    layer_heads_config=layer_heads_config,
                    relation_word_positions=relation_word_positions,
                    return_extra_attentions=return_extra_attentions,
                )


        # 处理新的返回值格式：outputs 现在是一个元组 (BaseModelOutputWithPast, attentions_pre_softmax, attentions_enhanced, attentions_pre_softmax_after_softmax)
        if isinstance(outputs, tuple) and len(outputs) > 3:
            # 新的返回值格式：包含增强后的注意力和增强前经过softmax的注意力
            model_output = outputs[0]
            attentions_pre_softmax = outputs[1]
            attentions_enhanced = outputs[2]
            attentions_pre_softmax_after_softmax = outputs[3]
            hidden_states = model_output.last_hidden_state
        elif isinstance(outputs, tuple) and len(outputs) > 2:
            # 包含增强后的注意力
            model_output = outputs[0]
            attentions_pre_softmax = outputs[1]
            attentions_enhanced = outputs[2]
            attentions_pre_softmax_after_softmax = None
            hidden_states = model_output.last_hidden_state
        elif isinstance(outputs, tuple) and len(outputs) > 1:
            # 旧的返回值格式：只有pre_softmax
            model_output = outputs[0]
            attentions_pre_softmax = outputs[1]
            attentions_enhanced = None
            attentions_pre_softmax_after_softmax = None
            hidden_states = model_output.last_hidden_state
        else:
            # 更旧的返回值格式（向后兼容）
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs[0]
            attentions_pre_softmax = None
            attentions_enhanced = None
            attentions_pre_softmax_after_softmax = None
        
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if return_extra_attentions and attentions_pre_softmax is not None:
                output += (attentions_pre_softmax,)
            if return_extra_attentions and attentions_enhanced is not None:
                output += (attentions_enhanced,)
            if return_extra_attentions and attentions_pre_softmax_after_softmax is not None:
                output += (attentions_pre_softmax_after_softmax,)
            return (loss,) + output if loss is not None else output

        # 根据 return_extra_attentions 决定是否返回额外注意力权重
        if return_extra_attentions:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=model_output.past_key_values if isinstance(outputs, tuple) else outputs.past_key_values,
                hidden_states=model_output.hidden_states if isinstance(outputs, tuple) else outputs.hidden_states,
                attentions=model_output.attentions if isinstance(outputs, tuple) else outputs.attentions,
            ), attentions_pre_softmax, attentions_enhanced, attentions_pre_softmax_after_softmax
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=model_output.past_key_values if isinstance(outputs, tuple) else outputs.past_key_values,
                hidden_states=model_output.hidden_states if isinstance(outputs, tuple) else outputs.hidden_states,
                attentions=model_output.attentions if isinstance(outputs, tuple) else outputs.attentions,
            )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                 # 新增：传递多层attention head mask参数
                "use_rel": kwargs.get("use_rel"),
                "layer_heads_config": kwargs.get("layer_heads_config"),
                "relation_word_positions": kwargs.get("relation_word_positions"),
                "return_extra_attentions": kwargs.get("return_extra_attentions", False),
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_cd(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        pad_token_id = getattr(self.config, "pad_token_id", 0)  # 默认 0，如果没有设置
        attention_mask = (input_ids != pad_token_id).long().to(input_ids.device)
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_cd", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
