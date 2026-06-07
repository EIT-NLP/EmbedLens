#!/bin/bash
CUDA_VISIBLE_DEVICES=0
CKPT="llava-v1.5-7b"

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --mode token_pruning \
    --target_ids 30296 26673 1141 \
    # --target_ids 30296 -1 26673 1141 \

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT}.jsonl