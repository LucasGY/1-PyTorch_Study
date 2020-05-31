今天的主题是：**Pytorch的预处理transforms**

**相关代码已经上传至Github：https://github.com/LucasGY/1-PyTorch_Study**

目录如下

[TOC]

## 一、torchvision计算机视觉工具包
* torchvision.transforms：常用的图像预处理方法，数据反转等；
* torchvision.datasets：常用数据集的dataset实现，如MNIST,CIFAR-IO,ImageNet等
* torchvision.model：常用的模型预训练，AlexNet,VGG,ResNet,GoogLeNet等

## 二、torchvision.transforms
[官方文档](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision-transforms)
[transforms的二十二个方法](https://blog.csdn.net/u011995719/article/details/85107009)
[transforms效果可视化](https://zhuanlan.zhihu.com/p/91477545)
### 2.1 transforms的功能
* 数据中心化
* 数据标准化
* 缩放
* 裁剪
* 旋转
* 翻转
* 填充
* 噪声添加
* 灰度变换
* 线性变换
* 仿射变换
* 常用的图像预处理方法
* 亮度、饱和度及对比度变换

利用`transforms.Compose`将各种变换组成类似pipeline的管道
```python
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```

### 2.2 transforms的执行过程
**2.2.1 在构建Dataset实例时传入transforms对象**
![20200530210422](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200530210422.png)

**2.2.2 在Dataset的__getitem__()中执行transform**
![](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200530211119.png)

**2.2.3 transforms.Compose管道顺序执行每个transform**
![20200530211418](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200530211418.png)



### 2.3 transforms的各种对图像的处理方法
官方文档只是将方法陈列，没有归纳总结，顺序很乱，这里总结一共有四大类，方便大家索引：
**2.3.1 裁剪——Crop**
中心裁剪：`transforms.CenterCrop`
随机裁剪：`transforms.RandomCrop`(以随机点为中心)
随机长宽比裁剪：`transforms.RandomResizedCrop`
上下左右中心裁剪：`transforms.FiveCrop`
上下左右中心裁剪后翻转，`transforms.TenCrop`

**2.3.2 翻转和旋转——Flip and Rotation**
依概率p水平翻转：`transforms.RandomHorizontalFlip(p=0.5)`
依概率p垂直翻转：`transforms.RandomVerticalFlip(p=0.5)`
随机旋转：`transforms.RandomRotation`

**2.3.3 图像变换**
resize：`transforms.Resize`
标准化：`transforms.Normalize`(对每个通道进行标准化)
转为tensor，**并归一化至[0-1]**：`transforms.ToTensor`
填充：`transforms.Pad`
修改亮度、对比度和饱和度：`transforms.ColorJitter`
转灰度图：`transforms.Grayscale`
线性变换：`transforms.LinearTransformation()`
仿射变换：`transforms.RandomAffine`
依概率p转为灰度图：`transforms.RandomGrayscale`
将数据转换为PILImage：`transforms.ToPILImage`
`transforms.Lambda`：Apply a user-defined lambda as a transform.

**2.3.4 对transforms操作，使数据增强更灵活**
`transforms.RandomChoice(transforms)`， 从给定的一系列transforms中选一个进行操作
`transforms.RandomApply(transforms, p=0.5)`，给一个transform加上概率，依概率进行操作
`transforms.RandomOrder`，将transforms中的操作随机打乱


### 2.4 transforms.RandomCrop

### 2.5 transforms.RandomCrop

### 2.6 transforms.RandomCrop

### 2.7 transforms.RandomCrop
