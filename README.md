# LLM from Scrach Implementation in PyTorch

A minimal PyTorch implementation of a GPT-2-style decoder-only Transformer. It includes causal self-attention, pre-norm residual blocks, learned token & positional embeddings, and a final linear head over the vocabulary.

## Features

- Causal self-attention with multi-head split and triangular mask
- Pre-norm Transformer blocks (LayerNorm → Attention/FFN → residual)
- Learned token + positional embeddings
- Dropout inside blocks for regularization

## Notes & Limitations

- No KV cache, weight decay schedule, or attention dropout (kept tiny & clear).
- Positional embedding is learned; rotary or ALiBi not included.

## Todos: 
Training, add batching, checkpointing, LR warmup/decay, and proper tokenization.