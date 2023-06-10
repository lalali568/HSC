#!/usr/bin/env bash

new_dataset="SMD"
sed -i "s#dataset:.*#dataset: $new_dataset#" config/TranAD/config.yaml

config_file="config/TranAD/config.yaml"
# 更新yaml文件的变量
new_epoch=80
sed -i "s#epoch:.*#epoch: $new_epoch#" config/TranAD/config.yaml

# 定义需要重复的数据路径列表
test_data_paths=(
  "machine-1-1"
  "machine-1-2"
  "machine-1-3"
  "machine-1-4"
  "machine-1-5"
  "machine-1-6"
  "machine-1-7"
  "machine-1-8"
  "machine-2-1"
  "machine-2-2"
  "machine-2-3"
  "machine-2-4"
  "machine-2-5"
  "machine-2-6"
  "machine-2-7"
  "machine-2-8"
  "machine-2-9"
  "machine-3-1"
  "machine-3-2"
  "machine-3-3"
  "machine-3-4"
  "machine-3-5"
  "machine-3-6"
  "machine-3-7"
  "machine-3-8"
  "machine-3-9"
  "machine-3-10"
  "machine-3-11"
)

# 循环处理每个数据路径
for path in "${test_data_paths[@]}"; do
  new_test_data_path="data/SMD/processed_data/test/$path.csv"
  new_train_data_path="data/SMD/processed_data/train/$path.csv"
  new_label_path="data/SMD/processed_data/test_label/$path.csv"

  sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" "$config_file"
  sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" "$config_file"
  sed -i "s#label_path:.*#label_path: $new_label_path#" "$config_file"

  python main.py --model_config_path config/TranAD/config.yaml
done
