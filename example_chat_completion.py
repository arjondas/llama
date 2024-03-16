# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import torch

from llama import Llama

USE_CUDA = os.environ.get('USE_CUDA', False)

if __name__ == "__main__":
    mp = True
    ckpt_dir = '/path/to/llama'
    tokenizer_path = '/path/to/tokenizer.model'
    temperature = 0.5
    top_p = 0.9
    max_seq_len = 1024
    max_batch_size = 8             ## this is where tpu shines
    max_gen_len = None
    dynamo = True

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        dynamo=dynamo
    )

    prompt_batch = [[{'role': 'user', 'content': 'Hello!! How are you?'}]] * 8
    results, tokens = generator.chat_completion(
        prompt_batch,
        max_gen_len=max_gen_len,
        top_p=top_p,
        temperature=temperature
    )

