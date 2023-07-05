#!/usr/bin/env bash

new_dataset="SWAT"
sed -i "s#dataset:.*#dataset: $new_dataset#" config/GDN/config.yaml

config_file="config/GDN/config.yaml"
# 更新yaml文件的变量
new_epoch=20
sed -i "s#epoch:.*#epoch: $new_epoch#" c"$config_file"


  new_test_data_path="data/SWAT/test_data.csv"
  new_train_data_path="data/SWAT/train_data.csv"
  new_label_path="data/SWAT/labels.csv"

  sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" "$config_file"
  sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" "$config_file"
  sed -i "s#label_path:.*#label_path: $new_label_path#" "$config_file"

  python main.py
