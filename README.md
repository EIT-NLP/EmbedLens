<h1 align="center">
What Do Visual Tokens Really Encode?
</h1>

<h3 align="center">
Uncovering Sparsity and Redundancy in Multimodal Large Language Models
</h3>

<p align="center">
  <strong>CVPR 2026 Highlight</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.00510">Paper</a>
</p>

## Overview

This repository contains the official implementation of **EmbedLens**, an
embedding-space probing tool for studying how visual tokens are represented and
processed inside multimodal large language models (MLLMs).

EmbedLens reveals a strong semantic sparsity in projected visual tokens. Based
on their nearest language-token embeddings, visual tokens consistently separate
into three groups:

- **Sink tokens**, which align with recurring sink directions.
- **Dead tokens**, which carry little image-specific information.
- **Alive tokens**, which preserve rich, fine-grained visual semantics.

The released code supports inspecting these token groups and evaluating how
token pruning and visual sublayer skipping affect downstream performance.

## Abstract

> Multimodal large language models (MLLMs) project visual tokens into the embedding space of language models, yet the internal structuring and processing of visual semantics remain poorly understood. In this work, we introduce a two-fold analytical framework featuring a novel probing tool, *EmbedLens*, to conduct a fine-grained analysis. We uncover a pronounced semantic sparsity at the input level: visual tokens consistently partition into sink, dead, and alive categories. Remarkably, only the alive tokens, comprising ~60% of the total input, carry image-specific meaning. Furthermore, using a targeted patch-compression benchmark, we demonstrate that these alive tokens already encode rich, fine-grained cues (e.g., objects, colors, and OCR) prior to entering the LLM. Internal visual computations (such as visual attention and feed-forward networks) are redundant for most standard tasks. For the small subset of highly vision-centric tasks that actually benefit from internal processing, we reveal that alive tokens naturally align with intermediate LLM layers rather than the initial embedding space, indicating that shallow-layer processing is unnecessary and that direct mid-layer injection is both sufficient. Ultimately, our findings provide a unified mechanistic view of visual token processing, paving the way for more efficient and interpretable MLLM architectures through selective token pruning, minimized visual computation, and mid-layer injection.

## Repository Structure

| Path | Description |
| --- | --- |
| `semantic_identification.ipynb` | Minimal EmbedLens walkthrough for inspecting the nearest vocabulary tokens of visual embeddings. |
| `apple.png` | Example image used by the notebook. |
| `llava/` | Modified LLaVA implementation with token-pruning and sublayer-skipping modes. |
| `lmms-eval/` | Modified evaluation framework that exposes the intervention settings. |
| `eval_scripts/` | Scripts for reproducing the released downstream evaluations. |

## Environment Setup

The code is tested with Python 3.10, PyTorch 2.1.2, Transformers 4.39.2,
and LLaVA-v1.5-7B. A CUDA-capable GPU is required.

```bash
conda create -n embedlens python=3.10
conda activate embedlens

pip install -e .
pip install -e ./lmms-eval
```

Some evaluation tasks require additional dataset preparation; refer to the
bundled [`lmms-eval` documentation](lmms-eval/docs/README.md) for task-specific
instructions.

## EmbedLens Walkthrough

The notebook `semantic_identification.ipynb` loads `liuhaotian/llava-v1.5-7b`, runs the example image through the model, and retrieves the nearest language-token embeddings for individual visual tokens. Replace `apple.png` and the prompt in the notebook to inspect your own examples.

## Evaluation

All evaluation scripts should be run from the repository root. For example:

```bash
bash eval_scripts/03_1_sinks_pruning/general.sh
```

The scripts are organized by intervention:

| Directory | Evaluation |
| --- | --- |
| `03_1_sinks_pruning/` | Prunes ViT sink tokens, LLM sink tokens, or both. |
| `03_2_mlp_2_skipping/` | Skips selected visual-token updates in early attention/MLP sublayers. |
| `04_dead/` | Compares dead-token pruning with a matched-size alive-token control. |

Each directory provides task-group scripts for general benchmarks,
vision-centric grounding, hallucination, and OCR. Results and sample-level
outputs are written to `./log/`.

The provided scripts use one or two GPUs through `accelerate`. Adjust
`--num_processes` and `CUDA_VISIBLE_DEVICES` to match your hardware.

### Intervention Configuration

Interventions are passed to the modified LLaVA model through `mode` and
`custom_config`. A minimal sink-pruning command is:

```bash
export CUSTOM_CONFIG='{"target_ids":[1141]}'

accelerate launch --num_processes 1 -m lmms_eval \
    --model llava \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager,mode=token_pruning,custom_config="$CUSTOM_CONFIG" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --output_path ./log/vit_sinks
```

The released configurations use the following target IDs:

| Value | Meaning |
| --- | --- |
| `1141` | ViT sink tokens. |
| `26673` | LLM sink tokens. |
| `30296` | Dead tokens. |
| `-1` | Selects non-target tokens as a control; its exact behavior depends on the intervention mode. |

Use `mode=token_pruning` with `target_ids` to remove selected visual-token
groups. Use `mode=skip` with `target_ids` and `skipping_sublayers` to suppress
their internal updates. Sublayer indices are zero-based: an integer `n`
denotes the attention update in decoder layer `n`, while `n + 0.5` denotes its
MLP update.

When `-1` is included in `target_ids`, the other IDs first define the target
group. In `token_pruning` mode, if this group is non-empty, the model randomly
removes the same number of tokens from its complement instead of removing the
target group. Thus, `[-1, 30296, 26673, 1141]` acts as the matched-size random
alive-token control used by the released pruning scripts. If no token matches
the other IDs, the current implementation removes all non-target visual tokens.
In `skip` mode, `-1` selects the entire complement rather than a matched-size
sample, so the same configuration suppresses updates to all alive tokens in the
requested sublayers. Using `[-1]` alone therefore selects all visual tokens in
both modes.


## Acknowledgements

This codebase builds on [LLaVA](https://github.com/haotian-liu/LLaVA) and
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We thank their
authors for making their work publicly available.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{fan2026visualtokensreallyencode,
  title={What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models},
  author={Yingqi Fan and Junlong Tong and Anhao Zhao and Xiaoyu Shen},
  year={2026},
  eprint={2603.00510},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.00510}
}
```