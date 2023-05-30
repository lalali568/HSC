#!/usr/bin/env bash

#更新yaml文件的变量

new_epoch=15027
sed -i "s#epoch:.*#epoch: $new_epoch#" config/COUTA/config.yaml

#第一次

new_test_data_path='data/MSL/C-1_test.npy'
new_train_data_path='data/MSL/C-1_train.npy'
new_label_path='data/MSL/C-1_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml

python main.py --model_config_path config/COUTA/config.yaml

#第二次
new_test_data_path='data/MSL/C-2_test.npy'
new_train_data_path='data/MSL/C-2_train.npy'
new_label_path='data/MSL/C-2_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml
#第3次
new_test_data_path='data/MSL/D-14_test.npy'
new_train_data_path='data/MSL/D-14_train.npy'
new_label_path='data/MSL/D-14_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml
#第4次
new_test_data_path='data/MSL/D-15_test.npy'
new_train_data_path='data/MSL/D-15_train.npy'
new_label_path='data/MSL/D-15_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml
#第5次
new_test_data_path='data/MSL/D-16_test.npy'
new_train_data_path='data/MSL/D-16_train.npy'
new_label_path='data/MSL/D-16_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第6次
new_test_data_path='data/MSL/F-4_test.npy'
new_train_data_path='data/MSL/F-4_train.npy'
new_label_path='data/MSL/F-4_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第7次
new_test_data_path='data/MSL/F-5_test.npy'
new_train_data_path='data/MSL/F-5_train.npy'
new_label_path='data/MSL/F-5_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第8次
new_test_data_path='data/MSL/F-7_test.npy'
new_train_data_path='data/MSL/F-7_train.npy'
new_label_path='data/MSL/F-7_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第9次
new_test_data_path='data/MSL/F-8_test.npy'
new_train_data_path='data/MSL/F-8_train.npy'
new_label_path='data/MSL/F-8_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第10次
new_test_data_path='data/MSL/M-1_test.npy'
new_train_data_path='data/MSL/M-1_train.npy'
new_label_path='data/MSL/M-1_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml
#第11次
new_test_data_path='data/MSL/M-2_test.npy'
new_train_data_path='data/MSL/M-2_train.npy'
new_label_path='data/MSL/M-2_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第12次
new_test_data_path='data/MSL/M-3_test.npy'
new_train_data_path='data/MSL/M-3_train.npy'
new_label_path='data/MSL/M-3_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第13次

new_test_data_path='data/MSL/M-4_test.npy'
new_train_data_path='data/MSL/M-4_train.npy'
new_label_path='data/MSL/M-4_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第14次
new_test_data_path='data/MSL/M-5_test.npy'
new_train_data_path='data/MSL/M-5_train.npy'
new_label_path='data/MSL/M-5_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#
#第15次
new_test_data_path='data/MSL/M-6_test.npy'
new_train_data_path='data/MSL/M-6_train.npy'
new_label_path='data/MSL/M-6_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第16次
new_test_data_path='data/MSL/M-7_test.npy'
new_train_data_path='data/MSL/M-7_train.npy'
new_label_path='data/MSL/M-7_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第17次
new_test_data_path='data/MSL/P-10_test.npy'
new_train_data_path='data/MSL/P-10_train.npy'
new_label_path='data/MSL/P-10_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第18次
new_test_data_path='data/MSL/P-11_test.npy'
new_train_data_path='data/MSL/P-11_train.npy'
new_label_path='data/MSL/P-11_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第19次
new_test_data_path='data/MSL/P-14_test.npy'
new_train_data_path='data/MSL/P-14_train.npy'
new_label_path='data/MSL/P-14_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第20次
new_test_data_path='data/MSL/P-15_test.npy'
new_train_data_path='data/MSL/P-15_train.npy'
new_label_path='data/MSL/P-15_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#21
new_test_data_path='data/MSL/S-2_test.npy'
new_train_data_path='data/MSL/S-2_train.npy'
new_label_path='data/MSL/S-2_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#22
new_test_data_path='data/MSL/T-4_test.npy'
new_train_data_path='data/MSL/T-4_train.npy'
new_label_path='data/MSL/T-4_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#23
new_test_data_path='data/MSL/T-5_test.npy'
new_train_data_path='data/MSL/T-5_train.npy'
new_label_path='data/MSL/T-5_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#24
new_test_data_path='data/MSL/T-8_test.npy'
new_train_data_path='data/MSL/T-8_train.npy'
new_label_path='data/MSL/T-8_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#25
new_test_data_path='data/MSL/T-9_test.npy'
new_train_data_path='data/MSL/T-9_train.npy'
new_label_path='data/MSL/T-9_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#26
new_test_data_path='data/MSL/T-12_test.npy'
new_train_data_path='data/MSL/T-12_train.npy'
new_label_path='data/MSL/T-12_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#27
new_test_data_path='data/MSL/T-13_test.npy'
new_train_data_path='data/MSL/T-13_train.npy'
new_label_path='data/MSL/T-13_labels.npy'
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml
