# HSC
这个项目是论文HSC: One-Class Classification Constraint in Reconstruction Networks for Multivariate Time Series 
Anomaly Detection的代码实现
<img src="./fig/model.png" />

## Datasets
文中使用的数据集可由下面的文章中获取其相关内容
1. **MSL (Mars Science Laboratory rover)** [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/pdf/1802.04431).

2. **SMD (Server Machine Dataset)** [Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network](https://netman.aiops.org/wp-content/uploads/2019/08/OmniAnomaly_camera-ready.pdf).

3. **SWaT (Secure Water Treatment)** [SWaT: a water treatment testbed for research and training on ICS security](https://ieeexplore.ieee.org/abstract/document/7469060).

## Requirements
代码使用的python版本为3.12，依赖包如下：
```text
scikit-learn==1.4.2
numpy==1.26.4
torch==2.4.0
torch_geometric==2.5.3
tqdm==4.66.4
```
## Usage
代码中的各项参数，通过config.yaml文件进行配置
代码运行的方式如下：
```text
python main.py
```

## Directory Structure

```text
├── checkpoints       # Directory for model checkpoints
│
├── config            # Configuration file for parameters
│
├── data              # Directory for data files
│
├── dataset           # Dataset loader
│
├── models            # Model architecture for HSC
│
├── tester            # Script for testing the model
│
├── trainer           # Script for training the model
│
├── util              # Utility functions directory
│   
├── main.py           # Main script to run the project
│
└── README.md         # Project documentation
```
## Results
<img src="./fig/result1.png" />
<img src="./fig/result2.png" />

## 联系方式
如果有任何问题，请联系lijiazhen_lalali@163.com
