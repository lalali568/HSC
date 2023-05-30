#!/usr/bin/env bash

#更新yaml文件的变量

new_epoch=100
sed -i "s#epoch:.*#epoch: $new_epoch#" config/TranAD/config.yaml

#第一次

new_test_data_path="'data/penism/test_data1.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml

python main.py --model_config_path config/TranAD/config.yaml

#注释掉main.py文件中训练的代码,在第64行
sed -i "83s/^/#/" main.py
#因为val数据集只需要跑一次，所以再把它注释掉
sed -i "103s/^/#/" main.py
#第二次
new_test_data_path="'data/penism/test_data2.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第3次
new_test_data_path="'data/penism/test_data3.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第4次
new_test_data_path="'data/penism/test_data4.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第5次
new_test_data_path="'data/penism/test_data5.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第6次
new_test_data_path="'data/penism/test_data6.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第7次
new_test_data_path="'data/penism/test_data7.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第8次
new_test_data_path="'data/penism/test_data8.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第9次
new_test_data_path="'data/penism/test_data9.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第10次
new_test_data_path="'data/penism/test_data10.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第11次
new_test_data_path="'data/penism/test_data11.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第12次
new_test_data_path="'data/penism/test_data12.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第13次
new_test_data_path="'data/penism/test_data13.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第14次
new_test_data_path="'data/penism/test_data14.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#第15次
new_test_data_path="'data/penism/test_data15.csv'"
sed -i "s#test_data_path:.*#test_data_path: $new_test_data_path#" config/TranAD/config.yaml
python main.py --model_config_path config/TranAD/config.yaml
#再改回来train的代码
sed -i "83s/^#//" main.py
sed -i "103s/^#//" main.py

