#!/usr/bin/env bash
new_dataset="penism"
config_file="config/MTAD_GAT/config.yaml"
sed -i "s#dataset:.*#dataset: $new_dataset#" "$config_file"
# 更新yaml文件的变量
new_epoch=100
sed -i "s#epoch:.*#epoch: $new_epoch#" "$config_file"

new_train_data=data/penism/train_data_3.csv
sed -i "s#train_data_path:.*#train_data_path: $new_train_data#" "$config_file"
# 注释和取消注释训练的代码
comment_lines() {
  sed -i "$1s/^/#/" main.py
}

uncomment_lines() {
  sed -i "$1s/^#//" main.py
}

# 循环运行测试数据
run_test_data() {
  local test_data_path=$1
  sed -i "s#test_data_path:.*#test_data_path: $test_data_path #" "$config_file"
  python main.py --model_config_path "$config_file"
}

# 循环注释和取消注释训练代码
toggle_training_code() {
  if [[ $1 -eq 1 ]]; then
    comment_lines 552
  elif [[ $1 -eq 15 ]]; then
    uncomment_lines 552
  fi
}

# 执行测试数据和训练代码的循环
execute_iterations() {
  local start_index=$1
  local end_index=$2
  local toggle_flag=$3

  for ((i = start_index; i <= end_index; i++)); do
    new_test_data_path="'data/penism/test_data$i\_3.csv'"
    run_test_data "$new_test_data_path"
    toggle_training_code $toggle_flag
  done
}

# 第一次运行
execute_iterations 1 1 1

# 运行15次测试数据
execute_iterations 2 15 0