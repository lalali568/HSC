#!/usr/bin/env bash

#更新yaml文件的变量

new_epoch=20
sed -i "s#epoch:.*#epoch: $new_epoch#" config/COUTA/config.yaml

#第一次

new_test_data_path="data/SMD/processed_data/test/machine-1-1.csv"
new_train_data_path="data/SMD/processed_data/train/machine-1-1.csv"
new_label_path="data/SMD/processed_data/test_label/machine-1-1.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml

python main.py --model_config_path config/COUTA/config.yaml

#第二次
new_test_data_path="data/SMD/processed_data/test/machine-1-2.csv"
new_train_data_path="data/SMD/processed_data/train/machine-1-2.csv"
new_label_path="data/SMD/processed_data/test_label/machine-1-2.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml
#第3次
new_test_data_path="data/SMD/processed_data/test/machine-1-3.csv"
new_train_data_path="data/SMD/processed_data/train/machine-1-3.csv"
new_label_path="data/SMD/processed_data/test_label/machine-1-3.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml
#第4次
new_test_data_path="data/SMD/processed_data/test/machine-1-4.csv"
new_train_data_path="data/SMD/processed_data/train/machine-1-4.csv"
new_label_path="data/SMD/processed_data/test_label/machine-1-4.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml
#第5次
new_test_data_path="data/SMD/processed_data/test/machine-1-5.csv"
new_train_data_path="data/SMD/processed_data/train/machine-1-5.csv"
new_label_path="data/SMD/processed_data/test_label/machine-1-5.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第6次
new_test_data_path="data/SMD/processed_data/test/machine-1-6.csv"
new_train_data_path="data/SMD/processed_data/train/machine-1-6.csv"
new_label_path="data/SMD/processed_data/test_label/machine-1-6.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第7次
new_test_data_path="data/SMD/processed_data/test/machine-1-7.csv"
new_train_data_path="data/SMD/processed_data/train/machine-1-7.csv"
new_label_path="data/SMD/processed_data/test_label/machine-1-7.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第8次
new_test_data_path="data/SMD/processed_data/test/machine-1-8.csv"
new_train_data_path="data/SMD/processed_data/train/machine-1-8.csv"
new_label_path="data/SMD/processed_data/test_label/machine-1-8.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第9次
new_test_data_path="data/SMD/processed_data/test/machine-2-1.csv"
new_train_data_path="data/SMD/processed_data/train/machine-2-1.csv"
new_label_path="data/SMD/processed_data/test_label/machine-2-1.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第10次
new_test_data_path="data/SMD/processed_data/test/machine-2-2.csv"
new_train_data_path="data/SMD/processed_data/train/machine-2-2.csv"
new_label_path="data/SMD/processed_data/test_label/machine-2-2.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第11次
new_test_data_path="data/SMD/processed_data/test/machine-2-3.csv"
new_train_data_path="data/SMD/processed_data/train/machine-2-3.csv"
new_label_path="data/SMD/processed_data/test_label/machine-2-3.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第12次
new_test_data_path="data/SMD/processed_data/test/machine-2-4.csv"
new_train_data_path="data/SMD/processed_data/train/machine-2-4.csv"
new_label_path="data/SMD/processed_data/test_label/machine-2-4.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第13次

new_test_data_path="data/SMD/processed_data/test/machine-2-5.csv"
new_train_data_path="data/SMD/processed_data/train/machine-2-5.csv"
new_label_path="data/SMD/processed_data/test_label/machine-2-5.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第14次
new_test_data_path="data/SMD/processed_data/test/machine-2-6.csv"
new_train_data_path="data/SMD/processed_data/train/machine-2-6.csv"
new_label_path="data/SMD/processed_data/test_label/machine-2-6.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#
#第15次
new_test_data_path="data/SMD/processed_data/test/machine-2-7.csv"
new_train_data_path="data/SMD/processed_data/train/machine-2-7.csv"
new_label_path="data/SMD/processed_data/test_label/machine-2-7.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第16次
new_test_data_path="data/SMD/processed_data/test/machine-2-8.csv"
new_train_data_path="data/SMD/processed_data/train/machine-2-8.csv"
new_label_path="data/SMD/processed_data/test_label/machine-2-8.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第17次
new_test_data_path="data/SMD/processed_data/test/machine-2-9.csv"
new_train_data_path="data/SMD/processed_data/train/machine-2-9.csv"
new_label_path="data/SMD/processed_data/test_label/machine-2-9.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第18次
new_test_data_path="data/SMD/processed_data/test/machine-3-1.csv"
new_train_data_path="data/SMD/processed_data/train/machine-3-1.csv"
new_label_path="data/SMD/processed_data/test_label/machine-3-1.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第19次
new_test_data_path="data/SMD/processed_data/test/machine-3-2.csv"
new_train_data_path="data/SMD/processed_data/train/machine-3-2.csv"
new_label_path="data/SMD/processed_data/test_label/machine-3-2.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#第20次
new_test_data_path="data/SMD/processed_data/test/machine-3-3.csv"
new_train_data_path="data/SMD/processed_data/train/machine-3-3.csv"
new_label_path="data/SMD/processed_data/test_label/machine-3-3.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#21
new_test_data_path="data/SMD/processed_data/test/machine-3-4.csv"
new_train_data_path="data/SMD/processed_data/train/machine-3-4.csv"
new_label_path="data/SMD/processed_data/test_label/machine-3-4.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#22
new_test_data_path="data/SMD/processed_data/test/machine-3-5.csv"
new_train_data_path="data/SMD/processed_data/train/machine-3-5.csv"
new_label_path="data/SMD/processed_data/test_label/machine-3-5.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#23

new_test_data_path="data/SMD/processed_data/test/machine-3-6.csv"
new_train_data_path="data/SMD/processed_data/train/machine-3-6.csv"
new_label_path="data/SMD/processed_data/test_label/machine-3-6.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#24
new_test_data_path="data/SMD/processed_data/test/machine-3-7.csv"
new_train_data_path="data/SMD/processed_data/train/machine-3-7.csv"
new_label_path="data/SMD/processed_data/test_label/machine-3-7.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#25
new_test_data_path="data/SMD/processed_data/test/machine-3-8.csv"
new_train_data_path="data/SMD/processed_data/train/machine-3-8.csv"
new_label_path="data/SMD/processed_data/test_label/machine-3-8.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#26
new_test_data_path="data/SMD/processed_data/test/machine-3-9.csv"
new_train_data_path="data/SMD/processed_data/train/machine-3-9.csv"
new_label_path="data/SMD/processed_data/test_label/machine-3-9.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#27
new_test_data_path="data/SMD/processed_data/test/machine-3-10.csv"
new_train_data_path="data/SMD/processed_data/train/machine-3-10.csv"
new_label_path="data/SMD/processed_data/test_label/machine-3-10.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml

#28
new_test_data_path="data/SMD/processed_data/test/machine-3-11.csv"
new_train_data_path="data/SMD/processed_data/train/machine-3-11.csv"
new_label_path="data/SMD/processed_data/test_label/machine-3-11.csv"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/COUTA/config.yaml
sed -i "s#train_data_path:.*#train_data_path: $new_train_data_path#" config/COUTA/config.yaml
sed -i "s#label_path:.*#label_path: $new_label_path#" config/COUTA/config.yaml
python main.py --model_config_path config/COUTA/config.yaml