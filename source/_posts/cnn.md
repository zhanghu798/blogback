layout: post
title: "卷积神经网络简介"
comments: true
tags:
	- CNN
date:  2017-05-19 23:39:25
updated: 
categories:
    - 机器学习
    - DL
---


{% cq %} <font size=4>Convolutional Neural Networks</font>{% endcq %}


<iframe src='http://cs231n.github.io/assets/conv-demo/index.html' width='100%' height='700px' style="border:none;">"http://cs231n.github.io/convolutional-networks/#overview"</iframe>
<center>[gif 1.卷积层卷积过程的动态示意图](http://cs231n.github.io/convolutional-networks/#overview)</center>

本文以图片问题中的CNN为例

# CNN基本思想
CNN是DNN的一种特殊形式

考虑使用DNN做图像分类问题，可以将图片的每个像素上每个通道上的数据看成是一个神经元

问题一：
假设对于$n \times n$大小的图片，每个像素3个颜色通道，则数据输入层为$n \times n \times 3$
假设有M层隐层，每层有N个神经元则一共有$3 n^2 N^M + MN$个参数。参数量较大

问题二：
人眼观察图片上时对单独的像素点是不敏感的，但是多个像素点一起观察的时候就会好很多


针对以上两种情况有了卷积神经网络，卷积神经网络主要的特点是使用了卷积核，卷积核观察数据是以区块进行的，相当与对像素点按块的做了特征整理。另外为了解决训练参数庞大的参数量级，使用了共享参数概念，即使用同一个卷积窗口观察通过一小块一小块的方式观察完一整张图片。每个卷积窗口观察图片的角度是有限的，通过多个不同的窗口（尺寸相同，但是处理观察到数据的方式不同）来使的观察更全面。  
以上就CNN中最重要的思想：卷积核及共享权值

# CNN结构基本框架

<img src="/pic/ml/cnn/cnn_struct.png" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[图2，CNN框架示意图](http://cs231n.github.io/convolutional-networks/#overview)</center>  

其中：
RELU为激励层  
FC为全链接层

# CNN结果说明

## 数据输入层  
图片中每个颜色通道代表一个维度的数据，每个通道上为一张含有位置信息的二维表


## 卷积层

卷积层主要作用是降维和聚焦视野

### 滑动窗口
卷积窗口尺寸：f
卷积窗口滑动间隔：s
卷积深度：d

### 数据对齐

对于输入图像为$a$的正方形图片， 使用宽度为$f$的卷积窗口， 以步长为$s$的方式滑动。一共可以得到
$n^2$个数据，其中：
$$n = \frac{a-f}{s} + 1$$

对于最后一个滑动窗口可能不能对齐是$n$为非正数，可以通过调整卷积窗口的尺寸和步长来实现对齐
另一个解决滑动不完美的方法是在周围填充0。

每行最少填充个数
$$q=f+(\lceil n \rceil-1)s - a$$
其中$\lceil n \rceil$为$n$的向上取整

另外<http://cs231n.github.io/convolutional-networks/#overview>中"no zero-padding"的P是指填充圈数，则$q$与$P$的关系为
$$P=\frac{1}{2}q$$


在保证卷机核能卷积操作与数据尺寸对齐的情况下，可以继续使用0填充来扩充数据边界使的原来数据边界作为数据中心达到边界中心化的目的，对于已经通过调整超参数或0填充的方法使的数据可以滑动对齐的长度为$a$的矩阵数据，在固定超参数$f$和$s$的情况下，有效填充个数$q$满足如下：

$$
\left\{
\begin{aligned}
& 0 \leqslant q \leqslant 2(f-1) \\
& (q+a-f) \% s = 0
\end{aligned}
\right.
$$

'$\%$'：取余符号

### 卷积计算过程  
给定超参数是滑动窗口大小参数，滑动步长，

如gif（1） 

## pooling层  
pooling层主要是降低维度，pool是给定观察窗口大小和滑动长度，每次滑动一个位置，对该位置所有点做运算，如下：

- Max pooling
  	求最大值

- Average pooling
	求平均

## 全链接 

作用：特征处理

# CNN的反向传播  

<http://0.0.0.0:4000/2017/05/19/dnn/>

# ImageNet冠军模型


# 参考 
\[1\] <http://cs231n.github.io/convolutional-networks/#overview>   
\[2\] 2017.03.15《Deep Learning翻译》<https://exacity.github.io/deeplearningbook-chinese/>  
\[3\] 2015.03.02 Google Inc [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf)   
\[4\] [CNN浅析和历年ImageNet冠军模型解析](https://mp.weixin.qq.com/s/fhJbE6V7r0nhVU3Vu0EGbw)  
\[5\] <http://cs231n.github.io/convolutional-networks/#overview>  


