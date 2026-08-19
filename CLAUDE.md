
# CLAUDE.md

This file provides guidance to claude code for this project

## What this is

A toy GPT model implementation

## Layout

- `scripts/` — Contains standalone scripts for training, evaluation, and other utilities.
- `nanogpt/` — The main Python package.
  - `__init__.py` — Package initialization.
  - `tokenizer.py` — `TokenizerBase` ABC plus three tokenizers.
  - `model.py` — The transformer itself (attention, blocks, model config).
  - `data.py` — Dataset loading, BPE tokenizer build/cache, batching.
  - `train.py` — Training loop, sampling, and the Modal remote-training entrypoint. 

Dependencies are managed with `uv` (`pyproject.toml` / `uv.lock`, Python ≥3.13)

## Running

The dev machine (macOS) has no CUDA, so training only ever runs on Modal.
