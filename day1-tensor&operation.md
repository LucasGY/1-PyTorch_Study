今天的主题是：**张量以及张量的相关操作**

**相关代码已经上传至Github：https://github.com/LucasGY/1-PyTorch_Study**

目录如下
[TOC]

## 一、Tensor&Variable
**pyTorch0.4.0以后，Variable已经并入张量:**

![20200526101800](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526101800.png)

![20200526101853](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526101853.png)

## 二、张量创建

* 直接创建
* 依据数值创建
* 依据概率创建

### 2.1 torch.tensor()直接创建
![20200526102234](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526102234.png)

```python
arr = np.ones((3,3))
print("ndarray的数据类型：", arr.dtype)
t = torch.tensor(arr)
t
```
out:
```python
ndarray的数据类型： float64
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
```
### 2.2 torch.from_numpy(ndarray)创建
![20200526102513](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526102513.png)
```python
arr = np.ones((3,3))
t = torch.from_numpy(arr)
arr[0,0] = -1
arr
t

t[0,1] = -2
arr
t
```
out:
```python
array([[-1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
tensor([[-1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]], dtype=torch.float64)
array([[-1., -2.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
tensor([[-1., -2.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]], dtype=torch.float64)
```

### 2.3 torch.zeros()
![20200526112826](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526112826.png)

### 2.4 torch.zeros_like()
![20200526112856](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526112856.png)

### 2.5 torch.ones()/torch.ones_like()
![20200526113040](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526113040.png)

### 2.6 torch.full()/torch.full_like()
![20200526113108](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526113108.png)

### 2.7 torch.arange()
![20200526113138](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526113138.png)

### 2.8 torch.linspace()
步长 = (start-end)/(steps-1)
![20200526113218](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526113218.png)

### 2.9 torch.logspace()
![20200526113258](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526113258.png)

### 2.10 torch.eye()
![20200526113337](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526113337.png)

### 2.11 torch.normal()
![20200526113409](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526113409.png)
![20200526113420](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526113420.png)
```python
# mean 张量，std:张量
mean = torch.arange(1, 5, dtype=torch.float)
std = torch.arange(1,5,dtype = torch.float)
t_normal = torch.normal(mean, std) 
mean
std
t_normal
# mean 标量，std:标量
t_normal = torch.normal(0.,1.,size=(4,)) #这里需要指定size
t_normal
```
out:
```python
tensor([1., 2., 3., 4.])
tensor([1., 2., 3., 4.])
tensor([0.6955, 2.5137, 2.6518, 3.3169])
tensor([-0.3655, -1.2086, -1.4962,  2.1676])
```

### 2.12 torch.randn()/torch.randn_like()标准正太分布
![20200526114111](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526114111.png)
![20200526114146](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526114146.png)
![20200526114217](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526114217.png)

## 三、张量属性
![20200526102800](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526102800.png)

* grad_fn：记录创建该张量时所用的方法；
* is_leaf：可以查看该张量是否为叶子结点
* requires_grad: 如果将tensor属性.requires_grad设置为True，它将开始追踪(track)在其上的所有操作（这样就可以利用链式法则进行梯度传播了）。完成计算后，可以调用.backward()来完成所有梯度计算。此Tensor的梯度将累积到.grad属性中。



## 四、张量操作

1. 张量的索引与切片
2. 张量的合并与拆分；
3. 张量的形状/维度变换；
4. 张量的数学基础运算——加/减/乘/除等；
5. 张量的线性代数运算
6. 广播机制
7. Tensor和NumPy相互转换
8. 运算的内存开销
9. Tensor on GPU

**PyTorch中的Tensor支持超过一百种操作，包括转置、索引、切片、数学运算、线性代数、随机数等等，可参考[官方文档](https://pytorch.org/docs/stable/tensors.html)。**

### 4.1 张量的索引与切片

我们可以使用**类似NumPy的索引**操作来访问Tensor的一部分，需要注意的是：**索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。**

这里不详细介绍，用到了再查官方文档：

函数|	功能
 :-: | :-: |
index_select(input, dim, index)	|在指定维度dim上选取，比如选取某些行、某些列
masked_select(input, mask)|	例子如上，a[a>0]，使用ByteTensor进行选取
nonzero(input) | 非0元素的下标
gather(input, dim, index)|根据index，在dim维度上选取数据，输出的size与index一样

#### 4.1.1 torch.index_select()
![20200526103639](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526103639.png)
```python
t = torch.randint(0,9,size=(3,3))
idx = torch.tensor([0,2],dtype=torch.long) #float error
t_select = torch.index_select(t,dim=0,index=idx)
t
t_select
```
out:
```python
tensor([[8, 5, 2],
        [7, 8, 3],
        [0, 0, 0]])
tensor([[8, 5, 2],
        [0, 0, 0]])
```

#### 4.1.2 torch.masked_select()
**返回一维张量**
![20200526103845](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526103845.png)
```python
t = torch.randint(0,9,size=(3,3))
mask = t.ge(5) # ge:greater&equal大于等于；gt大于；le小于等于；lt小于
t_select = torch.masked_select(t,mask)
t
mask
t_select
```
out:
```python
tensor([[2, 8, 2],
        [1, 0, 1],
        [2, 0, 1]])
tensor([[False,  True, False],
        [False, False, False],
        [False, False, False]])
tensor([8])
```

### 4.2 张量的合并与拆分
#### 4.2.1 torch.cat()/torch.stack()张量拼接
![20200526104109](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526104109.png)
```python
t = torch.ones((2,3))

t_0 = torch.cat([t,t],dim=0)
t_1 = torch.cat([t,t],dim=1)
t_0
t_1
```
out:
```python
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
tensor([[1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.]])
```
-----------------------------------------
```python
torch.stack([t,t,t],dim=0)
torch.stack([t,t,t],dim=0).shape
```
out:
```python
tensor([[[1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.]]])
torch.Size([3, 2, 3])
```

#### 4.2.2 张量切分——torch.chunk()/torch.split()
![20200526114553](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526114553.png)
![20200526114616](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526114616.png)
**torch.chunk()**
```python
a = torch.ones((2,5))
list_of_tensors = torch.chunk(a, dim=1,chunks=2) #chunck（）向上取整 chunk_size = 5/2=2.5 => 3
for idx,t in enumerate(list_of_tensors):
    print("第{}个张量：{}，shape is {}".format(idx+1,t,t.shape))
```
out:
```python
第1个张量：tensor([[1., 1., 1.],
        [1., 1., 1.]])，shape is torch.Size([2, 3])
第2个张量：tensor([[1., 1.],
        [1., 1.]])，shape is torch.Size([2, 2])
```

**torch.split()**
```python
T = torch.ones((2,5))
list_of_tensors = torch.split(T,2,dim=1)
for idx,t in enumerate(list_of_tensors):
    print("第{}个张量：{}，shape is {}".format(idx+1,t,t.shape))
    
print("\n")    
list_of_tensors = torch.split(T,[2,1,2],dim=1)
for idx,t in enumerate(list_of_tensors):
    print("第{}个张量：{}，shape is {}".format(idx+1,t,t.shape))
```
out:
```python
第1个张量：tensor([[1., 1.],
        [1., 1.]])，shape is torch.Size([2, 2])
第2个张量：tensor([[1., 1.],
        [1., 1.]])，shape is torch.Size([2, 2])
第3个张量：tensor([[1.],
        [1.]])，shape is torch.Size([2, 1])


第1个张量：tensor([[1., 1.],
        [1., 1.]])，shape is torch.Size([2, 2])
第2个张量：tensor([[1.],
        [1.]])，shape is torch.Size([2, 1])
第3个张量：tensor([[1., 1.],
        [1., 1.]])，shape is torch.Size([2, 2])
```

### 4.3 张量的形状/维度变换
* 形状变换
    *  torch.reshape()
    *  tensor.view()
注意view()返回的新Tensor与源Tensor虽然可能有不同的size，但是是共享data的，也即更改其中的一个，另外一个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)
所以如果我们想返回一个真正新的副本（即不共享data内存）该怎么办呢？Pytorch还提供了一个reshape()可以改变形状，但是此函数并不能保证返回的是其拷贝，所以不推荐使用。推荐先用clone创造一个副本然后再使用view。

使用clone还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源Tensor。

* 维度变/交换
    * 交换张量的两个轴：torch.transpose()/torch.t()
    * 压缩/拓展张量维度：torch.squeeze()/torch.unsqueeze() 
    
* 它可以将一个标量Tensor转换成一个Python number：item(), 

注：虽然view返回的Tensor与源Tensor是共享data的，但是依然是一个新的Tensor（因为Tensor除了包含data外还有一些其他属性），二者id（内存地址）并不一致。

#### 4.3.1 torch.reshape()
如果input张量在内存中是连续的，新张量与input共享数据内存
![20200526131002](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526131002.png)
```python
t = torch.randperm(8)
t
t_reshape = torch.reshape(t,(-1,4)) # -1不需要关心，通过其他维度计算后得来
t_reshape

t[0] = 1024
t_reshape
######################
id(t.data)
id(t_reshape.data)
```
out:
```python
tensor([1, 5, 7, 2, 3, 6, 4, 0])
tensor([[1, 5, 7, 2],
        [3, 6, 4, 0]])
tensor([[1024,    5,    7,    2],
        [   3,    6,    4,    0]])
2476829916840
2476829916840
```

#### 4.3.2 tensor.view()
真正新的副本（即不共享data内存）:
```python
x = torch.randperm(8)
x_cp = x.clone().view(2,4)
x -= 1
print(x)
print(x_cp)
```
out:
```python
tensor([ 6,  4,  0,  3,  1,  5, -1,  2])
tensor([[7, 5, 1, 4],
        [2, 6, 0, 3]])
```

#### 4.3.3 tensor.item()
```python
x = torch.randn(1)
print(x)
print(x.item())
```
out:
```python
tensor([-0.3158])
-0.3157952129840851
```

#### 4.3.4 torch.transpose()/torch.t()
![20200526131359](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526131359.png)
```python
t = torch.rand((2,3,4))
t
t_transpose = torch.transpose(t, dim0=1, dim1=2)
t_transpose
t.shape
t_transpose.shape
```
out:
```python
tensor([[[0.1588, 0.7554, 0.0545, 0.9482],
         [0.2970, 0.7333, 0.9678, 0.0518],
         [0.2337, 0.7122, 0.9966, 0.7976]],

        [[0.9408, 0.5882, 0.5032, 0.2701],
         [0.8410, 0.7756, 0.5663, 0.0511],
         [0.6994, 0.0997, 0.3450, 0.9285]]])
tensor([[[0.1588, 0.2970, 0.2337],
         [0.7554, 0.7333, 0.7122],
         [0.0545, 0.9678, 0.9966],
         [0.9482, 0.0518, 0.7976]],

        [[0.9408, 0.8410, 0.6994],
         [0.5882, 0.7756, 0.0997],
         [0.5032, 0.5663, 0.3450],
         [0.2701, 0.0511, 0.9285]]])
torch.Size([2, 3, 4])
torch.Size([2, 4, 3])
```

#### 4.3.5 torch.squeeze()/torch.unsqueeze()
![20200526131539](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526131539.png)
```python
t = torch.rand((1,2,3,1))
t_tq = torch.squeeze(t)
t_0 = torch.squeeze(t,dim=0)
t_1 = torch.squeeze(t,dim=1)

print('t')
t.shape
t
print("-----"*10)
print('t_tq')
t_tq.shape
t_tq
print("-----"*10)
print('t_0')
t_0.shape
print("-----"*10)
print('t_1')
t_1.shape # dim=1时长度不为1
```
out:
```python
t
torch.Size([1, 2, 3, 1])
tensor([[[[0.0711],
          [0.0122],
          [0.1489]],

         [[0.2756],
          [0.6359],
          [0.3835]]]])
--------------------------------------------------
t_tq
torch.Size([2, 3])
tensor([[0.0711, 0.0122, 0.1489],
        [0.2756, 0.6359, 0.3835]])
--------------------------------------------------
t_0
torch.Size([2, 3, 1])
--------------------------------------------------
t_1
torch.Size([1, 2, 3, 1])
```

### 4.4 张量数学运算
![20200526131856](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526131856.png)
![20200526131908](https://raw.githubusercontent.com/LucasGY/TempImage/master/img/20200526131908.png)

### 4.5 张量的线性代数运算
函数	|功能
 :-: | :-: |
trace	|对角线元素之和(矩阵的迹)
diag	|对角线元素
triu/tril	|矩阵的上三角/下三角，可指定偏移量
mm/bmm	|矩阵乘法，batch的矩阵乘法
addmm/addbmm/addmv/addr/baddbmm..	|矩阵运算
t	|转置
dot/cross	|内积/外积
inverse	|求逆矩阵
svd	|奇异值分解























