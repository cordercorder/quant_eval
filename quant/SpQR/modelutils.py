from contextlib import contextmanager
from typing import Dict, Union

import torch
import torch.nn as nn
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM

from quant_groups import dequantize

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama', 'Yi', 'opt' and 'falcon' are supported"
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")
LLAMA_LIKE = ("llama", "Yi")


@contextmanager
def suspend_nn_inits():
    def skip(*args, **kwargs):
        pass

    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def get_model(model_path, seqlen, load_quantized=None, dtype="bfloat16", device_map = "auto"):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=True).torch_dtype or "auto"
        )  # force transformers 4.29.2 to follow the same rules as 4.30.x
        dtype = torch.bfloat16
    else:
        dtype = getattr(torch, dtype)

    with suspend_nn_inits():
        if load_quantized:
            print("Initializing model with random weights...")
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=device_map
            )
            print("Loading quantized model ...")
            model = load_quantized_model(model, load_quantized)
        else:
            print("Loading pretrained model ...")
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                # local_files_only=True
            )
    model.seqlen = seqlen

    print("Model loaded sucessfully ...")

    return model


def get_model_head(model):
    head = torch.nn.ModuleList()
    if model.config.model_type in LLAMA_LIKE:
        if model.model.norm is not None:
            head.append(model.model.norm)
        head.append(model.lm_head)
    elif model.config.model_type.lower() in FALCON_TYPES:
        if model.transformer.ln_f is not None:
            head.append(model.transformer.ln_f)
        head.append(model.lm_head)
    elif model.config.model_type == "opt":
        if model.model.decoder.final_layer_norm is not None:
            head.append(model.model.decoder.final_layer_norm)
        if model.model.decoder.project_out is not None:
            head.append(model.model.decoder.project_out)
        head.append(model.lm_head)
    elif model.config.model_type == 'qwen':
        if model.transformer.ln_f is not None:
            head.append(model.transformer.ln_f)
        head.append(model.lm_head)
    elif model.config.model_type == 'chatglm':
        if model.transformer.encoder.final_layernorm is not None:
            head.append(model.transformer.encoder.final_layernorm)
        head.append(model.transformer.output_layer)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return head


def get_lm_logits(inps_, model):
    if model.config.model_type in LLAMA_LIKE:
        hidden_states = inps_.unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type.lower() in FALCON_TYPES:
        hidden_states = inps_.unsqueeze(0)
        if model.transformer.ln_f is not None:
            hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type == "opt":
        hidden_states = inps_.unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type == 'qwen':
        hidden_states = inps_.unsqueeze(0)
        if model.transformer.ln_f is not None:
            hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type == 'chatglm':
        hidden_states = inps_.unsqueeze(0)
        if model.transformer.encoder.final_layernorm is not None:
            hidden_states = model.transformer.encoder.final_layernorm(hidden_states)
        lm_logits = model.transformer.output_layer(hidden_states)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return lm_logits


def get_layers(model):
    if model.config.model_type in LLAMA_LIKE:
        return model.model.layers
    elif model.config.model_type.lower() in FALCON_TYPES:
        return model.transformer.h
    elif model.config.model_type == "opt":
        return model.model.decoder.layers
    elif model.config.model_type == 'qwen':
        return model.transformer.h
    elif model.config.model_type == 'chatglm':
        return model.transformer.encoder.layers
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def find_sublayers(module, layers=(nn.Conv2d, nn.Linear)):
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layers):
            res[name] = layer
    return res


def get_sequential_groups(model):
    if model.config.model_type in LLAMA_LIKE:
        return [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
    elif model.config.model_type.lower() in FALCON_TYPES:
        return [
            ["self_attention.query_key_value"],
            ["self_attention.dense"],
            ["mlp.dense_h_to_4h"],
            ["mlp.dense_4h_to_h"],
        ]
    elif model.config.model_type == "opt":
        return [
            ["self_attn.q_proj"],
            ["self_attn.k_proj"],
            ["self_attn.v_proj"],
            ["self_attn.out_proj"],
            ["fc1"],
            ["fc2"],
        ]
    elif model.config.model_type == 'qwen':
        return [
            ["attn.c_attn"],
            ["attn.c_proj"],
            ["mlp.w1", "mlp.w2"],
            ["mlp.c_proj"]
        ]
    elif model.config.model_type == 'chatglm':
        return [
            ["self_attention.query_key_value"],
            ["self_attention.dense"],
            ["mlp.dense_h_to_4h"],
            ["mlp.dense_4h_to_h"]
        ]
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def read_quant_weight_from_file(load_path, block_i, layer_name):
    return torch.load(load_path + "/" + str(block_i) + "/" + layer_name)

def move_list_to_device(params_list : list, device: Union[str, torch.device, torch.dtype])->None:
    for idx in range(len(params_list)):
        if isinstance(params_list[idx], torch.Tensor):
            params_list[idx] = params_list[idx].to(device)
        if isinstance(params_list[idx], list):
            move_dict_to_device(params_list[idx], device)

def move_dict_to_device(quantized_params_dict: Dict[str, torch.Tensor], device: Union[str, torch.device, torch.dtype]) -> None:
    for k, v in quantized_params_dict.items():
        if isinstance(v, torch.Tensor):
            quantized_params_dict[k] = v.to(device)
        if isinstance(v, list):
            move_list_to_device(quantized_params_dict[k], device)


def load_state_dict_from_cpu_to_layer_device(model: nn.Module, state_dict: Dict):
    devices = {name: device for name, device in model.named_parameters()}
    for name in state_dict:
        model.load_state_dict({name: state_dict[name].to(devices[name])}, strict=False)

def load_quantized_model(model, load_path):
    layers = get_layers(model)
    for i in trange(len(layers)):
        layer = layers[i]
        sub_layers = find_sublayers(layer)
        for name in sub_layers:
            quantized_params_dict = read_quant_weight_from_file(load_path, i, name)
            layer_dev = sub_layers[name].weight.device
            device = torch.device('cuda') if  layer_dev == torch.device('cpu') and torch.cuda.is_available() else layer_dev
            move_dict_to_device(quantized_params_dict, device)
            # print(f'{name=}')
            sub_layers[name].weight = nn.Parameter(
                layer_weight_dequantization(quantized_params_dict).to(sub_layers[name].weight.data.dtype)
            )
        layers[i] = layer
    load_state_dict_from_cpu_to_layer_device(model, torch.load(load_path + "/not_quantized_weights.pt"))
    
    # model.load_state_dict(torch.load(load_path + "/not_quantized_weights.pt"), strict=False)
    return model


def layer_weight_dequantization(quantized_params_dict):
    out_dim, in_dim = quantized_params_dict["weight_shape"]
    blocksize = quantized_params_dict["blocksize"]
    keep_last_columns = quantized_params_dict["keep_last_columns"]
    reconstructed_weight = torch.zeros(quantized_params_dict["weight_shape"],device=quantized_params_dict["quant_layer_zeros"][0].device)
    block_start_iter = range(0, in_dim - keep_last_columns, blocksize)
    block_start_iter = block_start_iter
    current_ind = 0
    if quantized_params_dict["quant_layer_scale_qq_scale"]:
        move_list_to_device(quantized_params_dict["quant_layer_zeros"], torch.uint8)
        move_list_to_device(quantized_params_dict["quant_layer_scale"][current_ind], torch.uint8)
    quantized_params_dict["quant_weights"] = quantized_params_dict["quant_weights"].to(torch.uint8)

    for block_start in block_start_iter:
        block_end = min(block_start + blocksize, in_dim)
        for column_index in range(block_start, block_end):
            if column_index % quantized_params_dict["groupsize"] == 0:
                if quantized_params_dict["quant_layer_scale_qq_scale"]:
                    dequantize_zeros = dequantize(
                        quantized_params_dict["quant_layer_zeros"][current_ind],
                        quantized_params_dict["quant_layer_zero_qq_scale"][current_ind],
                        quantized_params_dict["quant_layer_zero_qq_zero"][current_ind],
                    )
                    dequantize_scale = dequantize(
                        quantized_params_dict["quant_layer_scale"][current_ind],
                        quantized_params_dict["quant_layer_scale_qq_scale"][current_ind],
                        quantized_params_dict["quant_layer_scale_qq_zero"][current_ind],
                    )
                else:
                    dequantize_zeros = quantized_params_dict["quant_layer_zeros"][current_ind]
                    dequantize_scale = quantized_params_dict["quant_layer_scale"][current_ind]
                current_ind += 1

            reconstructed_weight[:, column_index] = dequantize(
                quantized_params_dict["quant_weights"][:, column_index].unsqueeze(1),
                dequantize_scale.reshape(-1, 1),
                dequantize_zeros.reshape(-1, 1),
            ).reshape_as(reconstructed_weight[:, column_index])
    # print(f'{quantized_params_dict["outliers_matrix"].device=}')
    # print(f'{type(quantized_params_dict["outliers_matrix"])=}')
    reconstructed_weight = (
        reconstructed_weight * (quantized_params_dict["outliers_matrix"].to_dense() == 0)
        + quantized_params_dict["outliers_matrix"].to_dense()
    )
    invperm = torch.argsort(quantized_params_dict["perm"])
    reconstructed_weight = reconstructed_weight[:, invperm]
    return reconstructed_weight
