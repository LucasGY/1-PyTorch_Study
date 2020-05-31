今天的主题是：**Pytorch的数据读取机制DataLoader&Dataset**

**相关代码已经上传至Github：https://github.com/LucasGY/1-PyTorch_Study**

目录如下

[TOC]

## 一、数据读取机制

![20200529134134](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200529134134.png)

我们以前手动加载数据的方式，在数据量小的时候，并没有太大问题，但是到了大数据量，我们需要使用 shuffle, 分割成mini-batch 等操作的时候，我们可以使用PyTorch的API快速地完成这些操作。

* Sampler：生成索引，也就是样本的序号；
* DataSet：根据索引读取图片和标签，决定数据从哪里读取，以及如何读取



### 1.1 DataLoader
构建可迭代的数据读取器
```python
torch.utils.data.DataLoader

DataLoader(dataset,             # Dataset类的对象，负责数据的抽象，一次调用getitem只返回一个样本，决定数据从哪里读取如何读取 
           batch_size=1,        # 每个batch的样本数
           shuffle=False,       # 设置为True，以便在每个epoch重新洗牌数据(默认:False)
           sampler=None,
           batch_sampler=None, 
           num_workers=0,       # 要使用多少子进程来加载数据。0表示将在主进程中加载数据。(默认值:0)
           collate_fn=None,
           pin_memory=False, 
           drop_last=False,     # 当最后一个batch样本数不够batch_size时，是否舍弃这一批数据
           timeout=0,
           worker_init_fn=None)

```
[参考PyTorch官方文档](https://pytorch.org/docs/stable/data.html)

-------------------------------------------------------------------

### 1.2 Dataset
从哪里加载数据

一共有两种形式：
* map-style datasets
    * 需要实现`__getitem__()` and `__len__()`，收到Sampler传来的索引，返回一个样本。
    * 例如，当使用dataset[idx]访问这样的数据集时，可以从磁盘上的文件夹中读取idx-th图像及其对应的标签。

* iterable-style datasets
    * iterable风格的dataset是`IterableDataset`子类的一个实例，它实现了`__iter__()`协议，并表示数据样本上的一个可迭代。这种类型的数据集特别适合于这样的情况，即随机读取非常昂贵，甚至不太可能，并且批大小取决于获取的数据。
    * 例如，它可以返回从数据库、远程服务器甚至实时生成的日志中读取的数据流。

比较常用的是第一种map-style datasets。需要实现`__getitem__()` and `__len__()`。
```python
torch.utils.data.Dataset

class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

```
[参考PyTorch官方文档](https://pytorch.org/docs/stable/data.html)





## 二、数据读取过程
![20200529140526](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200529140526.png)
**2.1 从DataLoader对象中取数据**
![20200529105911](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200529105911.png)

**2.2 跳入DataLoader对象的迭代器`__iter__()`中**
![20200529110252](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200529110252.png)
根据DataLoader对象中的`num_workers`参数，选择单/多进程读取数据；

**2.3 在实例化`_SingleProcessDataLoaderIter`，并执行初始化`__init__()`后，进入`__next__()`,从而进入`_next_data`函数：**
![20200529111626](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200529111626.png)

**2.4 进入`Sampler`的实例里面的`__iter__()`**
![20200529112043](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200529112043.png)
生成一个batch里面batch_size大小的索引列表返回到index上
![20200529112340](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200529112340.png)
后一行就是开始获取数据。

**2.5 fetch数据**
![20200529112603](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200529112603.png)
调用我们的Dataset对象，通过索引读取数据

**2.6 Dataset对象的`__getitem__()`函数包含了针对index索引（单个样本）数据的读取、转换、输出单个样本的X,y**
![20200529112910](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200529112910.png)
其实，这里面的代码也得并不好，应该在Dataset对象的`__init__()`中就完成对数据的读取，`__getitem__()`只进行索引（当然，如果内存够的话）

**2.7 得到data**
![20200529114053](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200529114053.png)
possibly_batched_index：一个batch的索引；
data形式：
type:list
`[(data1,label1),()...(data16,label16)] # batch_size=16`

**2.8 执行 `self.collate_fn(data)`将样本合并进一个batch**
data形式：
`data: [data Tensor, label Tensor]`
```python
data[0].shape
Out[5]: torch.Size([16, 3, 32, 32])

data[1].shape
Out[6]: torch.Size([16])
```
![20200529132733](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200529132733.png)

## 三、代码实例

