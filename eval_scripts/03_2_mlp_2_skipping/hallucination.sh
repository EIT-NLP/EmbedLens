
export CUSTOM_CONFIG='{"target_ids":[26673],"skipping_sublayers":[1.5]}'

accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager,mode=skip,custom_config="$CUSTOM_CONFIG" \
    --tasks pope,hallusion_bench_image \
    --batch_size 1 \
    --log_samples \
    --output_path ./log/sink_skip_mlp2


export CUSTOM_CONFIG='{"target_ids":[26673],"skipping_sublayers":[0.5,1]}'

accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager,mode=skip,custom_config="$CUSTOM_CONFIG" \
    --tasks pope,hallusion_bench_image \
    --batch_size 1 \
    --log_samples \
    --output_path ./log/sink_skip_mlp1mha2


export CUSTOM_CONFIG='{"target_ids":[26673],"skipping_sublayers":[0.5,1,1.5]}'

accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager,mode=skip,custom_config="$CUSTOM_CONFIG" \
    --tasks pope,hallusion_bench_image \
    --batch_size 1 \
    --log_samples \
    --output_path ./log/sink_skip_mlp1mha2mlp2


export CUSTOM_CONFIG='{"target_ids":[1141,30296,26673,-1],"skipping_sublayers":[1.5]}'
accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,attn_implementation=eager,mode=skip,custom_config="$CUSTOM_CONFIG" \
    --tasks pope,hallusion_bench_image \
    --batch_size 1 \
    --log_samples \
    --output_path ./log/Nsink_skip_mlp2