# LightInfer: A Light Inference High Performance Kernel library for LLMs

## Introduction
LightInfer is designed to be a high performance and easy-to-use library for inference of Large Language Models models which provides high level C++/Python APIs of [Composable Kernel](https://github.com/ROCm/composable_kernel.git) on ROCm platform, such as Embedding, Gemm, Normlization, Attention and so on. LightInfer 
is lightweight design, easy scalability, and high performance, applicable to a variety of LLMs for production scenarios.

## Key Features

## Supported Matrix
|Category|LightInfer Layer|CK kernel|CK Version|DataType|Note|
|---|---|---|---|---|---|
|Embedding|Embedding|Embedding|Old|FP32/FP16/BF16| |
|||Embedding+Add+Add+LayerNorm|Old|FP32/FP16/BF16| |
|Normlization|LayerNorm|LayerNorm|New|FP32/FP16/BF16| |
|||LayerNorm+Add|New|FP32/FP16/BF16| |
|||RMSNorm|New|FP32/FP16/BF16| |
|||RMSNorm+Add|New|FP32/FP16/BF16| |
|Gemm|Linear|GemmRCR|Old|FP32/FP16/BF16| |
|||GemmRCR+Bias|Old|FP32/FP16/BF16| |
|||GemmRCR+Bias+Silu|Old|FP32/FP16/BF16| |
|||GemmRCR+Bias+Tanh|Old|FP32/FP16/BF16| |
|||GemmRCR+Bias+Gelu|Old|FP32/FP16/BF16| |
|||GemmRCR+Bias+Permute|Old|FP32/FP16/BF16| |
|||GemmRCR+Bias+Multiply|Old|FP32/FP16/BF16| |
|||GemmRCR+Bias+Add|Old|FP32/FP16/BF16| |
|Attention|MemoryEfficientAttention|Flash Attention Forward|New|FP16/BF16| |
||FuseMultiHeadAttention|LayerNorm,GemmRCR+Bias+Permute,GemmRCR+Bias, Flash Attention Forward, GemmRCR+Bias+Add|New|FP16/BF16| |
||MemoryEfficientAttentionDecoder|Flash Attention AppendKV, Flash Attention SplitKV|New|FP16/BF16| |
|Transformer|FeedForward|Gemm+Bias+Gelu|New|FP32/FP16/BF16| |
||TransformerEncoder|LayerNorm,GemmRCR+Bias+Permute,GemmRCR+Bias, Flash Attention Forward, GemmRCR+Bias+Add, Gemm+Bias+Gelu|New|FP16/BF16| |
||TransformerDecoder|||| |

## Installation

## Getting Started

## Performance Benchmark


## Authors
LightInfer is entirely developed by [Yanxing-Shi](https://github.com/Yanxing-Shi) from Composable Kernel team. 

## Acknowledgement
LightInfer is inspired by the following projects, we would like to thank the authors for their great work:
- [Composable Kernel](https://github.com/ROCm/composable_kernel.git)
- [flash-attention](https://github.com/Dao-AILab/flash-attention.git)
- [xformers](https://github.com/facebookresearch/xformers.git)
- [AITemplate](https://github.com/facebookincubator/AITemplate.git)
- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer.git)







