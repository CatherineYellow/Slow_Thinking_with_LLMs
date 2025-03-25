#!/bin/bash

# note: benchmark dataset path required in run_eval_tool.py :)
# AIME24: "STILL-3-TOOL/evaluation/dataset/AIME24.jsonl"
# AIME25: "STILL-3-TOOL/evaluation/dataset/AIME25.jsonl"
# hmmt_feb_2025: "STILL-3-TOOL/evaluation/dataset/hmmt_feb_2025.jsonl"

# how to run: cd STILL-3-TOOL/evaluation && bash scripts/eval_tool.sh
PWD=`pwd`

SCRIPT=${PWD}/run_eval_tool.py

# required: model_path_list and prompt_template_path :)
# prompt_template_path: "STILL-3-TOOL/config/prompt_template.json"
model_path_list=(\
    "RUC-AIBOX/STILL-3-TOOL-32B" \
    # "" \
    # "" \
    # "" \
)
prompt_template_path="STILL-3-TOOL/config/prompt_template.json"

num_model=${#model_path_list[@]}

for ((i=0;i<$num_model;i++)) do
{
    MODEL_PATH=${model_path_list[$i]}
    SRC_PATH=${MODEL_PATH}

    # AIME24, greedy, execute code
    # when set decode to greedy, n should be 1
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python ${SCRIPT} \
        --data_name AIME24 \
        --target_path ${SRC_PATH} \
        --model_name_or_path ${MODEL_PATH} \
        --max_tokens 32768 \
        --paralle_size 8 \
        --decode greedy \
        --n 1 \
        --prompt_template ${prompt_template_path} \
        --prompt r1_code \
        --exe_code &

    wait

    # AIME24, sample, execute code
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python ${SCRIPT} \
        --data_name AIME24 \
        --target_path ${SRC_PATH} \
        --model_name_or_path ${MODEL_PATH} \
        --max_tokens 32768 \
        --paralle_size 8 \
        --decode sample \
        --n 8 \
        --prompt_template ${prompt_template_path} \
        --prompt r1_code \
        --exe_code &

    wait

    # AIME25, greedy, execute code
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python ${SCRIPT} \
        --data_name AIME25 \
        --target_path ${SRC_PATH} \
        --model_name_or_path ${MODEL_PATH} \
        --max_tokens 32768 \
        --paralle_size 8 \
        --decode greedy \
        --n 1 \
        --prompt_template ${prompt_template_path} \
        --prompt r1_code \
        --exe_code &

    wait

    # AIME25, sample, execute code
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python ${SCRIPT} \
        --data_name AIME25 \
        --target_path ${SRC_PATH} \
        --model_name_or_path ${MODEL_PATH} \
        --max_tokens 32768 \
        --paralle_size 8 \
        --decode sample \
        --n 8 \
        --prompt_template ${prompt_template_path} \
        --prompt r1_code \
        --exe_code &

    wait

    # hmmt_feb_2025, greedy, execute code
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python ${SCRIPT} \
        --data_name hmmt_feb_2025 \
        --target_path ${SRC_PATH} \
        --model_name_or_path ${MODEL_PATH} \
        --max_tokens 32768 \
        --paralle_size 8 \
        --decode greedy \
        --n 1 \
        --prompt_template ${prompt_template_path} \
        --prompt r1_code \
        --exe_code &

    wait

    # hmmt_feb_2025, sample, execute code
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python ${SCRIPT} \
        --data_name hmmt_feb_2025 \
        --target_path ${SRC_PATH} \
        --model_name_or_path ${MODEL_PATH} \
        --max_tokens 32768 \
        --paralle_size 8 \
        --decode sample \
        --n 8 \
        --prompt_template ${prompt_template_path} \
        --prompt r1_code \
        --exe_code &

    wait

    # AIME24, sample, don't execute code
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python ${SCRIPT} \
        --data_name AIME24 \
        --target_path ${SRC_PATH} \
        --model_name_or_path ${MODEL_PATH} \
        --max_tokens 32768 \
        --paralle_size 8 \
        --decode sample \
        --n 8 \
        --prompt_template ${prompt_template_path} \
        --prompt r1_code &

    wait

    # deepseek-ai/DeepSeek-R1-Distill-Qwen-32B: AIME24, sample
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python ${SCRIPT} \
        --data_name AIME24 \
        --target_path ${SRC_PATH} \
        --model_name_or_path ${MODEL_PATH} \
        --max_tokens 32768 \
        --paralle_size 8 \
        --decode sample \
        --n 8 \
        --prompt_template deepseek &

    wait

    # if you wanna enable mp, set use_slice to True
    # here is an example of splitting the dataset into 2 slices, with 4 gpus each
    {
        # slice 0
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        python ${SCRIPT} \
            --data_name AIME24 \
            --target_path ${SRC_PATH} \
            --model_name_or_path ${MODEL_PATH} \
            --max_tokens 32768 \
            --paralle_size 4 \
            --decode sample \
            --n 8 \
            --prompt_template deepseek \
            --use_slice \
            --slice_id 0 &

        # slice 1
        export CUDA_VISIBLE_DEVICES=4,5,6,7
        python ${SCRIPT} \
            --data_name AIME24 \
            --target_path ${SRC_PATH} \
            --model_name_or_path ${MODEL_PATH} \
            --max_tokens 32768 \
            --paralle_size 4 \
            --decode sample \
            --n 8 \
            --prompt_template deepseek \
            --use_slice \
            --slice_id 1 &
    }
    # remember to merge the results of all slices after they finish :)

    wait
}
done
