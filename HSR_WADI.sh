#!/usr/bin/env bash

new_dataset="WADI"
sed -i "s#dataset:.*#dataset: $new_dataset#" config/HSR/config.yaml

config_file="config/HSR/config.yaml"
# 更新yaml文件的变量
new_epoch=80
sed -i "s#epoch:.*#epoch: $new_epoch#" config/HSR/config.yaml


  new_test_data_path="data/WADI/test_data.csv"
  new_train_data_path="data/WADI/train_data.csv"
  new_label_path="data/WADI/label.csv"

  sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" "$config_file"
  sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" "$config_file"
  sed -i "s#label_path:.*#label_path: $new_label_path#" "$config_file"

  python main.py --model_config_path config/COUTA/config.yaml
