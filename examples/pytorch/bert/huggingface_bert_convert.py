import argparse
import configparser
import numpy as np
from pathlib import Path
import os

import torch
from transformers import BertModel

# using numpy extension: https://github.com/GreenWaves-Technologies/bfloat16
# install the library with `pip install bfloat16`
from bfloat16 import bfloat16

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    elif data_type == "bf16":
        return bfloat16
    else:
        assert False, f"Invalid weight data type {data_type}"

def split_and_convert_process(saved_dir, factor, key, val):
    if key.find("attention.output.dense.bias") != -1 or \
       key.find("attention.output.LayerNorm.weight") != -1 or \
       key.find("attention.output.LayerNorm.bias") != -1 or \
       key.find("output.dense.bias") != -1 or \
       key.find("output.LayerNorm.weight") != -1 or \
       key.find("output.LayerNorm.bias") != -1 : 

        saved_path = saved_dir + "/bert." + key + ".bin"
        val.tofile(saved_path)
            
    elif key.find("attention.output.dense.weight") != -1 or key.find("output.dense.weight") != -1:
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = f"{saved_dir}/bert.{key}.{j}.bin"
            split_vals[j].tofile(saved_path)
    
    elif key.find("attention.self.query_key_value.weight") != -1 or \
         key.find("attention.self.query_key_value.bias") != -1 or \
         key.find("intermediate.dense.weight") != -1 or \
         key.find("intermediate.dense.bias") != -1:
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir + "/bert." + key + ".%d.bin" % j
            split_vals[j].tofile(saved_path)
    
    else:
        print("[WARNING] cannot convert key '{}'".format(key))


def split_and_convert(args):
    saved_dir = args.saved_dir + "/%d-gpu/" % args.infer_gpu_num
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    t_gpu_num = args.trained_gpu_num
    i_gpu_num = args.infer_gpu_num
    assert(i_gpu_num % t_gpu_num == 0)

    factor = (int)(i_gpu_num / t_gpu_num)
    
    # load position_embedding from rank 0
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_model = BertModel.from_pretrained(args.in_file).to(torch_device)
    hf_config = vars(bert_model.config)
    print(f"hf_config: {hf_config}")

    print("named parameters:")
    for name, param in bert_model.named_parameters():
        print(f"- {name}")
        
    # save bert weight config to config.ini
    hidden_size = hf_config["hidden_size"]
    head_num = hf_config["num_attention_heads"]
    head_size = hidden_size // head_num
    num_layers = hf_config["num_hidden_layers"]
    
    np_weight_data_type = get_weight_data_type(args.weight_data_type)
    
    try:
        config = configparser.ConfigParser()
        config["bert"] = {}
        config["bert"]["model_name"] = "bert" if hf_config["_name_or_path"] == '' else hf_config["_name_or_path"]
        config["bert"]["vocab_size"] = str(hf_config["vocab_size"])
        config["bert"]["type_vocab_size"] = str(hf_config["type_vocab_size"])
        config["bert"]["max_position_embeddings"] = str(hf_config["max_position_embeddings"])
        config["bert"]["position_embedding_type"] = str(hf_config["position_embedding_type"])
        config["bert"]["hidden_units"] = str(hf_config["hidden_size"])
        config["bert"]["num_layers"] = str(hf_config["num_hidden_layers"])
        config["bert"]["num_heads"] = str(hf_config["num_attention_heads"])
        config["bert"]["size_per_head"] = str(hf_config["hidden_size"] // hf_config["num_attention_heads"])
        config["bert"]["activation_type"] = str(hf_config["hidden_act"])
        config["bert"]["inter_size"] = str(hf_config["intermediate_size"])
        config["bert"]["layer_norm_eps"] = str(hf_config["layer_norm_eps"])
        config["bert"]["weight_data_type"] = args.weight_data_type
        config["bert"]["tensor_para_size"] = str(args.infer_gpu_num)
    
        with open((Path(saved_dir) / f"config.ini").as_posix(), 'w') as configfile:
            config.write(configfile)
    except Exception as e:
        print(f"Fail to save the config in config.ini. due to {e}")

    param_to_weights = lambda param: param.detach().cpu().numpy().astype(np_weight_data_type)
    
    # layer-wise weights, example:
    #   - encoder.layer.0.attention.self.query.weight
    #   - encoder.layer.0.attention.self.query.bias
    #   - encoder.layer.0.attention.self.key.weight
    #   - encoder.layer.0.attention.self.key.bias
    #   - encoder.layer.0.attention.self.value.weight
    #   - encoder.layer.0.attention.self.value.bias
    #   - encoder.layer.0.attention.self.output.dense.weight
    #   - encoder.layer.0.attention.self.output.dense.bias
    #   - encoder.layer.0.attention.output.LayerNorm.weight
    #   - encoder.layer.0.attention.output.LayerNorm.bias
    #   - encoder.layer.0.intermediate.dense.weight
    #   - encoder.layer.0.intermediate.dense.bias
    #   - encoder.layer.0.output.dense.weight
    #   - encoder.layer.0.output.dense.bias
    #   - encoder.layer.0.output.LayerNorm.weight
    #   - encoder.layer.0.output.LayerNorm.bias
    for l in range(num_layers):
        print(f"converting layer {l}")
        
        # first merge QKV into a single weight
        qkv_weights = np.stack([
            param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.attention.self.query.weight']),
            param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.attention.self.key.weight']),
            param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.attention.self.value.weight']),
        ])
        qkv_weights_base_name = f'encoder.layer.{l}.attention.self.query_key_value.weight'
        split_and_convert_process(saved_dir, factor, qkv_weights_base_name, qkv_weights)
        
        # first merge QKV into a single bias
        qkv_bias = np.stack([
            param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.attention.self.query.bias']),
            param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.attention.self.key.bias']),
            param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.attention.self.value.bias']),
        ])
        qkv_bias_base_name = f'encoder.layer.{l}.attention.self.query_key_value.bias'
        split_and_convert_process(saved_dir, factor, qkv_bias_base_name, qkv_bias)
        
        # attention dense
        attn_dense_weight = param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.attention.output.dense.weight'])
        attn_dense_weight_base_name = f'encoder.layer.{l}.attention.output.dense.weight'
        split_and_convert_process(saved_dir, factor, attn_dense_weight_base_name, attn_dense_weight)
        
        # attention bias
        attn_dense_bias = param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.attention.output.dense.bias'])
        attn_dense_weight_base_name = f'encoder.layer.{l}.attention.output.dense.bias'
        split_and_convert_process(saved_dir, factor, attn_dense_weight_base_name, attn_dense_bias)
        
        # layer norm gamma
        attn_ln_weight = param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.attention.output.LayerNorm.weight'])
        attn_ln_weight_base_name = f'encoder.layer.{l}.attention.output.LayerNorm.weight'
        split_and_convert_process(saved_dir, factor, attn_ln_weight_base_name, attn_ln_weight)
        
        # layer norm beta
        attn_ln_bias = param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.attention.output.LayerNorm.bias'])
        attn_ln_bias_base_name = f'encoder.layer.{l}.attention.output.LayerNorm.bias'
        split_and_convert_process(saved_dir, factor, attn_ln_bias_base_name, attn_ln_bias)
        
        # intermediate dense
        intermediate_weight = param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.intermediate.dense.weight'])
        intermediate_weight_base_name = f'encoder.layer.{l}.intermediate.dense.weight'
        split_and_convert_process(saved_dir, factor, intermediate_weight_base_name, intermediate_weight)
        
        # intermediate bias
        intermediate_bias = param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.intermediate.dense.bias'])
        intermediate_bias_base_name = f'encoder.layer.{l}.intermediate.dense.bias'
        split_and_convert_process(saved_dir, factor, intermediate_bias_base_name, intermediate_bias)
        
        # output dense
        output_dense_weight = param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.output.dense.weight'])
        output_dense_weight_base_name = f'encoder.layer.{l}.output.dense.weight'
        split_and_convert_process(saved_dir, factor, output_dense_weight_base_name, output_dense_weight)
        
        # output bias
        output_dense_bias = param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.output.dense.bias'])
        output_dense_bias_base_name = f'encoder.layer.{l}.output.dense.bias'
        split_and_convert_process(saved_dir, factor, output_dense_bias_base_name, output_dense_bias)
    
        # output layer norm gamma
        output_ln_weight = param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.output.LayerNorm.weight'])
        output_ln_weight_base_name = f'encoder.layer.{l}.output.LayerNorm.weight'
        split_and_convert_process(saved_dir, factor, output_ln_weight_base_name, output_ln_weight)
        
        # output layer norm beta
        output_ln_bias = param_to_weights(bert_model.state_dict()[f'encoder.layer.{l}.output.LayerNorm.bias'])
        output_ln_bias_base_name = f'encoder.layer.{l}.output.LayerNorm.bias'
        split_and_convert_process(saved_dir, factor, output_ln_bias_base_name, output_ln_bias)
    
     # final common weights
    for name, param in bert_model.named_parameters():
        if name == "embeddings.word_embeddings.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "bert.embeddings.word_embeddings.weight.bin")
        elif name == "embeddings.position_embeddings.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "bert.embeddings.position_embeddings.weight.bin")
        elif name == "embeddings.token_type_embeddings.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "bert.embeddings.token_type_embeddings.weight.bin")
        elif name == "embeddings.LayerNorm.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "bert.embeddings.LayerNorm.weight.bin")
        elif name == "embeddings.LayerNorm.bias":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "bert.embeddings.LayerNorm.bias.bin") 
        elif name == "pooler.dense.weight":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "bert.pooler.dense.weight.bin")
        elif name == "pooler.dense.bias":
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "bert.pooler.dense.bias.bin")

#python3 huggingface_bert_convert.py -saved_dir=./ -in_file=google-bert/bert-base-uncased -infer_gpu_num=1 -weight_data_type=fp16 -model_name=bert-base
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir', '-o', type=str,
                        help='file name of output file', required=True)
    parser.add_argument('-trained_gpu_num', '-t_g', type=int, help='How many gpus for inference', default=1)
    parser.add_argument('-infer_gpu_num', '-i_g', type=int, help='How many gpus for inference', required=True)
    parser.add_argument('-in_file', '-i', type=str,
                        help='file name of input checkpoint file', required=True)
    parser.add_argument("-weight_data_type", type=str,
                        default="fp16", choices=["fp32", "fp16"])
    parser.add_argument('-model_name', '-m_n', type=str, help='model name', required=True)

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    split_and_convert(args)
