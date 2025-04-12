# README

本项目实现了一个简单的三层全连接神经网络，在CIFAR-10数据集上的测试精度可以达到0.4-0.5之间。这篇文档主要介绍了如何训练及测试该模型，包含前期准备，训练，测试及可视化模型权重四个部分。

## Set up

首先需要将模型代码下载到本地：

```cmd
git clone https://github.com/cydai999/cv-homework1.git
```

之后从 https://drive.google.com/drive/folders/1o9RS29zpJ1bYAaT2TJ2hgZOFu_YXL77V?usp=drive_link 下载模型权重与数据集文件夹，解压缩后将其放在项目根目录下。相对路径位置应如下所示：

```plaintext
cv-homework1
├── dataset 
│   └── cifar-10-python
│   	└── cifar-10-batches-py
├── saved_models
│   └── best_model
│   	└── models
├── mynn
├── train.py
├── test.py
├── hyperparam_search.py
├── visual_weight.py
└── README.md
```



## Train

训练模型的过程非常简单，只需在项目文件夹路径下打开终端，输入如下指令，即可以以默认配置进行训练：

```
python train.py
```

除此之外，支持自定义模型超参数，以下列出一些可选参数及说明：

```
--hidden_size(-hs): 隐藏层维度，默认值1000
--act_func(-a): 激活函数类型，可在'Sigmoid', 'ReLU', 'LeakyReLU'中选择，默认值'LeakyReLU'
--weight_decay_param(-wd): 权重衰减系数，默认值1e-5
--init_lr(-lr): 初始学习率，默认值1e-2
--step_size(-s): 学习率衰减周期，默认值5（5个epoch）
--gamma(-g): 学习率衰减系数，默认值0.1
--batch_size(-bs): 每个批次的样本量，默认值32
--epoch(-e): 遍历轮数，默认值10
--log_iter(-l): 打印loss和accuracy周期，默认值100
```

示例：

比如，想要以1e-3的学习率训练5个epoch，可以输入如下指令：

```
python train.py -lr 1e-3 -e 5
```



## Test

若想要测试训练好的模型在测试集上的表现，可以在终端中输入：

```
python test.py
```

假如saved_models不在要求路径下，也可以指定模型路径：

```
python test.py -p ./PATH/TO/MODEL/DIR/best_model.pickle
```



## Visual

若想要可视化模型参数，可以在终端输入：

```
python visual_weight.py
```

同样，假如saved_models不在要求路径下，也可以指定模型路径：

```
python visual_weight.py -p ./PATH/TO/MODEL/DIR/best_model.pickle
```



