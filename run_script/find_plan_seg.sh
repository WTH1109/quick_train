#!/bin/bash

# 配置参数
p_values=(0.6)
lr_values=(0.001)
b_values=(16)
model_name=(resnet18 resnet34 resnet50 efficientnet_b0 efficientnet_b4 inception_v3)
config_name=(configs/base_config/tongue_merge/select_plan/ResNet_tongue_seg_plan1.yaml configs/base_config/tongue_merge/select_plan/ResNet_tongue_seg_plan2.yaml configs/base_config/tongue_merge/select_plan/ResNet_tongue_seg_plan3.yaml configs/base_config/tongue_merge/select_plan/ResNet_tongue_seg_plan4.yaml configs/base_config/tongue_merge/select_plan/ResNet_tongue_seg_plan5.yaml)
GPUS=(1 2 3)          # 可用GPU列表
MAX_JOBS_PER_GPU=2    # 每张GPU上最大并行任务数

rm -rf tmp_seg && mkdir tmp_seg

# 生成参数队列
param_queue=()
for p in "${p_values[@]}"; do
  for lr in "${lr_values[@]}"; do
    for b in "${b_values[@]}"; do
      for m in "${model_name[@]}"; do
        for c in "${config_name[@]}"; do
          param_queue+=("$p $lr $b $m $c")
        done
      done
    done
  done
done

# 任务执行函数
run_job() {
  local gpu_id=$1
  local p=$2
  local lr=$3
  local b=$4
  local m=$5
  local c=$6
  local exp_name="merge_resnet_seg"

  CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --config "$c" \
    -n "$exp_name" \
    -m "$m"\
    -p "$p" \
    -lr "$lr" \
    -b "$b" \
    -e "1000" \
    -d /media/ps/data-ssd/home-ssd/wengtaohan/Dataset/tongue/tongue_IgA \
    --log "True"
}

# 初始化GPU任务计数器文件
for gpu in "${GPUS[@]}"; do
  counter_file="tmp_seg/gpu_${gpu}_jobs"
  echo 0 > "$counter_file"
  # 设置退出时清理计数器文件
  trap "rm -f $counter_file" EXIT
done

# 已提交任务数
job_idx=0
total_jobs=${#param_queue[@]}

# 主循环：动态分配任务
while (( job_idx < total_jobs )); do
  for gpu in "${GPUS[@]}"; do
    # 原子读取当前任务数
    counter_file="tmp_seg/gpu_${gpu}_jobs"
    current_jobs=$(flock -x "$counter_file" -c "cat $counter_file")

    if (( current_jobs < MAX_JOBS_PER_GPU )); then
      # 获取当前参数组合
      params="${param_queue[$job_idx]}"
      set -- $params
      p=$1
      lr=$2
      b=$3
      m=$4
      c=$5

      # 原子更新计数器
      flock -x "$counter_file" -c "echo $((current_jobs + 1)) > $counter_file"

      # 在子shell中启动任务并处理计数器
      (
        run_job "$gpu" "$p" "$lr" "$b" "$m" "$c"

        # 任务完成后原子减少计数器
        flock -x "$counter_file" -c "echo \$((\$(cat $counter_file) - 1)) > $counter_file"
      ) &

      echo "[Job $((job_idx + 1))/$total_jobs] Started on GPU $gpu (p=$p, lr=$lr, b=$b, m=$m)"
      ((job_idx++))

      # 处理完一个参数后立即检查下一个GPU
      break
    fi
  done
  # 避免CPU空转，适当调整等待时间
  sleep 0.5
done

# 等待所有后台任务完成
wait
echo "All jobs completed!"
