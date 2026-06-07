export CUSTOM_CONFIG='{"target_ids":[1141]}'

accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager,mode=token_pruning,custom_config="$CUSTOM_CONFIG" \
    --tasks textvqa_val,ocrbench,docvqa_val \
    --batch_size 1 \
    --log_samples \
    --output_path ./log/vit_sinks


export CUSTOM_CONFIG='{"target_ids":[26673]}'

accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager,mode=token_pruning,custom_config="$CUSTOM_CONFIG" \
    --tasks textvqa_val,ocrbench,docvqa_val \
    --batch_size 1 \
    --log_samples \
    --output_path ./log/llm_sinks


export CUSTOM_CONFIG='{"target_ids":[26673,1141]}'

accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager,mode=token_pruning,custom_config="$CUSTOM_CONFIG" \
    --tasks textvqa_val,ocrbench,docvqa_val \
    --batch_size 1 \
    --log_samples \
    --output_path ./log/vit_sinks_llm_sinks










