实验环境为：

python	   python 3.6

Anaconda	Anaconda3-5.3.0 Pycharm		20.03

Pytorch-GPU	1.7.0+cu110

torchvision	Torchvision 0.8.1

1. 本实验设计为国家一级保护动物（一级一等、一级二等、一级三等）的识别系统，本实验搭建了VGG-16、ResNet-50、DenseNet-121三个网络模型。
2. 本实验设置了ResNet-50和DenseNet-121网络使用迁移学习的方法进行训练，并在ResNet-50网络模型上加入CBAM注意力机制来对网络进行优化。
3. 本实验为手动手机的数据集数据集包含46类动物6024张图片数据集，数据集链接地址为：https://aistudio.baidu.com/aistudio/datasetdetail/127411
4. 迁移学习ResNet-50官方权重地址为：https://download.pytorch.org/models/resnet50-19c8e357.pth
5. 迁移学习DenseNet-121官方权重地址为：https://download.pytorch.org/models/densenet121-a639ec97.pth