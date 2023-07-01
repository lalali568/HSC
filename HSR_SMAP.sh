#!/usr/bin/env bash

new_dataset="SMAP"
sed -i "s#dataset:.*#dataset: $new_dataset#" config/HSR/config.yaml

config_file="config/HSR/config.yaml"

# 更新yaml文件的变量
new_epoch=700
sed -i "s#epoch:.*#epoch: $new_epoch#" "$config_file"
#注释下面的列表

# 数据路径列表
data_paths=(
 'A-1'
 'A-2'
  'A-3'
  'A-4'
  'A-5'
  'A-6'
  'A-7'
  'A-8'
  'A-9'
  'B-1'
  'D-1'
  'D-2'
  'D-3'
  'D-4'
  'D-5'
  'D-6'
  'D-7'
  'D-8'
  'D-9'
  'D-11'
  'D-12'
  'D-13'
  'E-1'
  'E-2'
  'E-3'
  'E-4'
  'E-5'
  'E-6'
  'E-7'
  'E-8'
  'E-9'
  'E-10'
  'E-11'
  'E-12'
  'E-13'
  'F-1'
  'F-2'
  'F-3'
  'G-1'
  'G-2'
  'G-3'
  'G-4'
  'G-6'
  'G-7'
  'P-1'
  'P-2'
  'P-3'
  'P-4'
  'P-7'
  'R-1'
  'S-1'
  'T-1'
  'T-2'
  'T-3'
)



# 循环处理数据
for path in "${data_paths[@]}"; do
  new_test_data_path="data/SMAP/$path"_test.npy
  new_train_data_path="data/SMAP/$path"_train.npy
  new_label_path="data/SMAP/$path"_labels.npy

  sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" "$config_file"
  sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" "$config_file"
  sed -i "s#label_path:.*#label_path: $new_label_path#" "$config_file"

  python main.py --model_config_path "$config_file"
done