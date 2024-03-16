## User Guide: Running Llama2 Inference on TPU (v3-8)

No need for guilds. Please refer to the code in `example_chat_completion.py` for implementation details.

## What is this?

This is a modified version of the [Llama2 Google Next Inference](https://github.com/pytorch-tpu/llama/tree/llama2-google-next-inference) Branch. It is specifically designed to run Llama2 7B on TPU v3-8, unlike the Google Next Inference version, which only supports TPU v4 and v5e.

## How does it work?

The modifications allow the utilization of the [PyTorch/XLA GSPMD](https://pytorch.org/blog/pytorch-xla-spmd/) system (on TPU v3-8) with a new Mesh and distribution configuration to shard the weights and cache across the entire TPU mesh. Specifically, the k, v cache has a predefined static size to avoid TPU graph recompilation after each token generation. This new configuration enables Llama2 7B to fit into one TPU v3-8 device with a significant amount of memory remaining to run inference for up to a batch size of 64. The same configuration can also be used to infer Llama2 13B on the same device.

## Why TPU v3-8?

Many of us have easy free access to TPU v3-8 through Kaggle, and who doesn't like running Open-Source LLMs to generate text for free?