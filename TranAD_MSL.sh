#!/usr/bin/env bash

new_dataset="MSL"
sed -i "s#dataset:.*#dataset: $new_dataset#" config/TranAD/config.yaml

config_file="config/TranAD/config.yaml"

# 更新yaml文件的变量
new_epoch=700
sed -i "s#epoch:.*#epoch: $new_epoch#" "$config_file"
#注释下面的列表

# 数据路径列表
data_paths=(
  'C-1'
  'C-2'
  'D-14'
  'D-15'
  'D-16'
  'F-4'
  'F-5'
  'F-7'
  'F-8'
  'M-1'
  'M-2'
  'M-3'
  'M-4'
  'M-5'
  'M-6'
  'M-7'
  'P-10'
  'P-11'
  'P-14'
  'P-15'
  'S-2'
  'T-4'
  'T-5'
  'T-8'
  'T-9'
  'T-12'
  'T-13'
)
  data_paths=(
  'T-5'
  'T-8'
  'T-9'
  'T-12'
  'T-13'
)


# 循环处理数据
for path in "${data_paths[@]}"; do
  new_test_data_path="data/MSL/$path"_test.npy
  new_train_data_path="data/MSL/$path"_train.npy
  new_label_path="data/MSL/$path"_labels.npy

  sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" "$config_file"
  sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" "$config_file"
  sed -i "s#label_path:.*#label_path: $new_label_path#" "$config_file"

  python main.py --model_config_path "$config_file"
done