
# export CUSTOM_CONFIG='{"target_ids":[30296]}'

# accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
#     --model llava \
#     --model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager,mode=token_pruning,custom_config="$CUSTOM_CONFIG" \
#     --tasks mme,gqa,mmstar,mmbench_en_dev \
#     --batch_size 1 \
#     --log_samples \
#     --output_path ./log/dead


export CUSTOM_CONFIG='{"target_ids":[30296,26673,1141]}'
# export CUSTOM_CONFIG='{"target_ids":[-1,30296,26673,1141]}'
accelerate launch --num_processes 2 --main_process_port 12346 -m lmms_eval \
    --model llava \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager,mode=token_pruning,custom_config="$CUSTOM_CONFIG" \
    --tasks mme,gqa,mmstar,mmbench_en_dev \
    --batch_size 1 \
    --log_samples \
    --output_path ./log/alive







