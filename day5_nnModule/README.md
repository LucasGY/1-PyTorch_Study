今天的主题是：**Pytorch的模型的构建**

**相关代码已经上传至Github：https://github.com/LucasGY/1-PyTorch_Study**

目录如下

[TOC]

## 一、PyTorch搭建模型概览
![20200603142050](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603142050.png)
每个网络层都对应一个运算，如卷积、池化等，也属于nn.Module.

![20200603151827](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603151827.png)
### 1.1 建立继承nn.Module的子类

### 1.2 构建网络层
构建**模型子模块**，在`__init__()`：
![20200603150940](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603150940.png)

### 1.3 重写nn.Module的forward函数——构建前向传播
在实例化模型，并进行调用时，触发nn.Module的`__call__()`函数，里面有一个`self.forward`，执行复写的forward函数开始前向传播：
![20200603151314](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603151314.png)

`__call__()`函数表示类的实例被当作函数调用时，执行`__call__()`里面的内容。

## 二、nn.Module
### 2.1 torch.nn Overview
![20200603151951](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603151951.png)

### 2.2 nn.Module Overview
![20200603152024](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603152024.png)

![20200603153110](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603153110.png)

8个字典初始化过程：
1、实例化自定义模型时，进入`__init__()`，先执行父类(nn.Module)的`__init__()`:
![20200603204749](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603204749.png)

2、初始化8个字典：
![20200603204845](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603204845.png)

### 2.3 nn.Conv2d等也属于nn.Module
 ![20200603210714](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603210714.png)
![20200603210846](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603210846.png)
![20200603210926](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603210926.png)

因此也有8个字典(_modules子网络没有东西；_parameters有可学习参数)：
![20200603211129](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603211129.png)
_parameters有weight和bias，属于**Parameter类，特殊的张量：**
![20200603211523](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603211523.png)
具有张量的属性：
![20200603212818](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603212818.png)

### 2.4 将子模块(nn.Module子类)赋值给变量
![20200603213005](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603213005.png)

触发Module类的`__setattr__`函数：
![20200603213253](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603213253.png)
value：nn.Conv2d子模块
name：类属性conv2






## 三、搭建网络模型常用的容器containers，如Sequential，ModuleList, ModuleDict
### 3.1 模型容器containers
![20200603160819](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603160819.png)
![20200603163036](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603163036.png)

**3.1.1 nn.Sequential**
nn.Sequential是nn.Module的容器，用于按顺序包装一组网络层。

用nn.Sequential将LeNet包装为features和classifier两个容器：
![20200603161140](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603161140.png)

![20200603161556](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603161556.png)

![20200603162658](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603162658.png)

nn.Sequential继承于nn.Module，因此也有8个字典属性：
![20200604080920](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200604080920.png)

#### nn.Sequential的两种构建方式

```python
# Example of using Sequential
model = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
        )

# Example of using Sequential with OrderedDict
model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1,20,5)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,64,5)),
            ('relu2', nn.ReLU())
        ]))
```
区别：在执行Sequential的实例化时，跳入__init__()：
* Sequential：_modules有序字典的键是数字；
* Sequential with OrderedDict：_modules有序字典的键是你自己定义的名字，如；'conv1'。
![20200604081240](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200604081240.png)
![20200604081542](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200604081542.png)

![20200604082103](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200604082103.png)





**3.1.2 nn.ModuleList**
![20200603162726](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603162726.png)


**3.1.3 nn.ModuleDict**
![20200603162905](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603162905.png)


## 四、Alexnet以及pytorch`视觉CV`常用模型
![20200603163404](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603163404.png)
![20200603163425](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603163425.png)

在torchvision中有许多已经搭好的现成的模型：
![20200603163642](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200603163642.png)