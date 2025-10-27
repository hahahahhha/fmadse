import torch
from torch import nn
from functools import partial
from typing import Dict, Optional

from mod import *



@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

        self.weight_datatype = None
        self.weight_wq_bits = None
        self.weight_is_mx = False
        self.weight_group_size = None
        self.use_weight_dtype_for_activation = False

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self._quantize_input(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @torch.no_grad()
    def _quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_weight_dtype_for_activation and self.weight_datatype:
            bits = self.weight_wq_bits if self.weight_wq_bits is not None else 0
            if bits > 0:
                return quantize_tensor_to_datatype(
                    x,
                    wq_bits=bits,
                    datatype=self.weight_datatype,
                    if_mx=self.weight_is_mx,
                )
        return self.act_quant(x)

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", group_size=128, wq_bits=4, datatype='mixed', quantize_output=False,
        datatype_support: List[str]=['fp3','fp4','fp6_e2m3','fp6_e3m2','fp8_e4m3','fp8_e5m2','int4','int8'],if_mx_support:bool=False,
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8
            )
        # AHMED: Add other weight quantization methods here.
        elif weight_quant == "fmadse":
            grouped_weight = module.weight.view(-1, group_size)
            quantized_grouped_weight, selected_datatype = search_datatype(
                grouped_weight,
                wq_bits=wq_bits,
                datatype_support=datatype_support,
                if_mx_support=if_mx_support,
                group_size=None,
            )
            new_module.weight = quantized_grouped_weight.view_as(module.weight)
            new_module.weight_datatype = selected_datatype
            new_module.weight_wq_bits = wq_bits
            new_module.weight_is_mx = bool(selected_datatype and selected_datatype.startswith("mx_"))
            new_module.weight_group_size = group_size
            new_module.use_weight_dtype_for_activation = selected_datatype is not None
        # elif weight_quant == "mod_asym":
        #     grouped_weight = module.weight.view(-1, group_size)
        #     quantized_grouped_weight = quant_int_asym(grouped_weight, wq_bits=wq_bits)
        #     new_module.weight = quantized_grouped_weight.view_as(module.weight)
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_opt(
    model, weight_quant="per_tensor", act_quant="per_tensor", group_size=128, wq_bits_list=[4,4,4], datatype="wrong", quantize_bmm_input=True,
    datatype_support: List[str]=['fp3','fp4','fp6_e2m3','fp6_e3m2','fp8_e4m3','fp8_e5m2','int4','int8'],if_mx_support:bool=False,
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )
    assert len(wq_bits_list)==3
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            wq_bits=wq_bits_list[0]
            m.fc1 = W8A8Linear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            wq_bits=wq_bits_list[1]
            m.fc2 = W8A8Linear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            wq_bits=wq_bits_list[2]
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
                wq_bits=wq_bits,
                datatype=datatype,
                quantize_output=quantize_bmm_input,
                datatype_support=datatype_support,
                if_mx_support=if_mx_support
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
                wq_bits=wq_bits,
                datatype=datatype,
                quantize_output=quantize_bmm_input,
                datatype_support=datatype_support,
                if_mx_support=if_mx_support

            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
                wq_bits=wq_bits,
                datatype=datatype,
                quantize_output=quantize_bmm_input,
                datatype_support=datatype_support,
                if_mx_support=if_mx_support
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
    # wq_bits=wq_bits_list[2]
    # model.lm_head = W8A8Linear.from_float(
    #     model.lm_head, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
    # )
    return model


def _collect_layer_datatypes(model: torch.nn.Module) -> Dict[str, Dict[str, Optional[object]]]:
    metadata: Dict[str, Dict[str, Optional[object]]] = {}
    for name, module in model.named_modules():
        weight_dtype = getattr(module, "weight_datatype", None)
        if weight_dtype is None:
            continue

        entry: Dict[str, Optional[object]] = {
            "weight_dtype": weight_dtype,
            "activation_dtype": weight_dtype
            if getattr(module, "use_weight_dtype_for_activation", False)
            else None,
        }

        weight_bits = getattr(module, "weight_wq_bits", None)
        if weight_bits is not None:
            entry["wq_bits"] = weight_bits

        group_size = getattr(module, "weight_group_size", None)
        if group_size is not None:
            entry["group_size"] = group_size

        entry["if_mx"] = bool(getattr(module, "weight_is_mx", False))
        metadata[name] = entry

    return metadata


def quantize_llama_like(
    model, weight_quant="per_channel", act_quant="per_token", group_size=128, wq_bits_list=[4,4,4], datatype="wrong", quantize_bmm_input=False,datatype_support: List[str]=['fp3','fp4','fp6_e2m3','fp6_e3m2','fp8_e4m3','fp8_e5m2','int4','int8'],if_mx_support:bool=False,
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )
    assert len(wq_bits_list)==3
    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            wq_bits=wq_bits_list[0]
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            wq_bits=wq_bits_list[1]
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            wq_bits=wq_bits_list[2]
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
                wq_bits=wq_bits,
                datatype=datatype,
                quantize_output=quantize_bmm_input,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
                wq_bits=wq_bits,
                datatype=datatype,
                quantize_output=quantize_bmm_input,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                group_size=group_size,
                wq_bits=wq_bits,
                datatype=datatype,
                quantize_output=quantize_bmm_input,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
    # wq_bits=wq_bits_list[2]
    # model.lm_head = W8A8Linear.from_float(
    #     model.lm_head, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
    # )
    return model


def quantize_mixtral(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, group_size=128, wq_bits_list=[4,4,4,4], datatype="wrong", datatype_support: List[str]=['fp3','fp4','fp6_e2m3','fp6_e3m2','fp8_e4m3','fp8_e5m2','int4','int8'],if_mx_support:bool=False,
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )
    assert len(wq_bits_list)==4
    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            wq_bits=wq_bits_list[0]
            m.w1 = W8A8Linear.from_float(
                m.w1, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            m.w2 = W8A8Linear.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            m.w3 = W8A8Linear.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            wq_bits=wq_bits_list[1]
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            wq_bits=wq_bits_list[2]
            m.gate = W8A8Linear.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
    wq_bits=wq_bits_list[3]
    model.lm_head = W8A8Linear.from_float(
    model.lm_head, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
    )
    return model


def quantize_falcon(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True,group_size=128, wq_bits_list=[4,4,4], datatype="wrong", datatype_support: List[str]=['fp3','fp4','fp6_e2m3','fp6_e3m2','fp8_e4m3','fp8_e5m2','int4','int8'],if_mx_support:bool=False,
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )
    assert len(wq_bits_list)==3
    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            wq_bits=wq_bits_list[0]
            m.dense_h_to_4h = W8A8Linear.from_float(
                m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            m.dense_4h_to_h = W8A8Linear.from_float(
                m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            wq_bits=wq_bits_list[1]
            m.query_key_value = W8A8Linear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
            m.dense = W8A8Linear.from_float(
                m.dense, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
            )
    wq_bits=wq_bits_list[2]
    model.lm_head = W8A8Linear.from_float(
    model.lm_head, weight_quant=weight_quant, act_quant=act_quant, group_size=group_size, wq_bits=wq_bits, datatype=datatype,datatype_support=datatype_support,if_mx_support=if_mx_support
    )
    return model


def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token", group_size= 128,  datatype="wrong", quantize_bmm_input=False,wq_bits_list=[4,4,4],datatype_support: List[str]=['fp3','fp4','fp6_e2m3','fp6_e3m2','fp8_e4m3','fp8_e5m2','int4','int8'],if_mx_support:bool=False,
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    quantized_model: Optional[torch.nn.Module] = None
    if isinstance(model, OPTPreTrainedModel):
        quantized_model = quantize_opt(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            group_size=group_size,
            datatype=datatype,
            quantize_bmm_input=quantize_bmm_input,
            wq_bits_list=wq_bits_list,
            datatype_support=datatype_support,
            if_mx_support=if_mx_support,
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        quantized_model = quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            group_size=group_size,
            datatype=datatype,
            quantize_bmm_input=quantize_bmm_input,
            wq_bits_list=wq_bits_list,
            datatype_support=datatype_support,
            if_mx_support=if_mx_support,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        quantized_model = quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            group_size=group_size,
            datatype=datatype,
            quantize_bmm_input=quantize_bmm_input,
            wq_bits_list=wq_bits_list,
            datatype_support=datatype_support,
            if_mx_support=if_mx_support,
        )
    elif isinstance(model, FalconPreTrainedModel):
        quantized_model = quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            group_size=group_size,
            datatype=datatype,
            quantize_bmm_input=quantize_bmm_input,
            wq_bits_list=wq_bits_list,
            datatype_support=datatype_support,
            if_mx_support=if_mx_support,
        )

    if quantized_model is None:
        raise ValueError(f"Unsupported model type: {type(model)}")

    layer_metadata = _collect_layer_datatypes(quantized_model)
    setattr(quantized_model, "layer_quant_datatypes", layer_metadata)
    return quantized_model
