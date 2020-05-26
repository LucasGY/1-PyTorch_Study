今天的主题是：**张量以及张量的相关操作**

**相关代码已经上传至Github：https://github.com/LucasGY/**

目录如下
[TOC]

## 一、 XGBoost参数
XGB的参数类型又可以被分为：XGBoost框架参数（General parameters）、XGBoost 弱学习器参数（Booster parameters ）、命令行版本的参数（Command line parameters）以及其他参数

### 1.1 XGBoost框架参数（General parameters）
**1. booster [default= gbtree ]**
>booster决定了XGBoost使用的弱学习器类型，可以是默认的gbtree, 也就是CART决策树，还可以是线性弱学习器gblinear以及DART。一般来说，我们使用gbtree就可以了，不需要调参。
Which booster to use. Can be gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.