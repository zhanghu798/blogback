layout: post
title: "卷积神经网络简介"
comments: true
tags:
	- CNN
date:  2017-05-23 20:03:33
updated: 
categories:
    - 机器学习
    - DL
---



{% cq %} <font size=4>Convolutional Neural Networks</font>{% endcq %}

卷积神经网络简介

<!-- more -->


<iframe src='http://cs231n.github.io/assets/conv-demo/index.html' width='100%' height='700px' style="border:none;">"http://cs231n.github.io/convolutional-networks/#overview"</iframe>
<center>[动图 1.卷积层卷积过程的动态示意图](http://cs231n.github.io/convolutional-networks/#overview)</center>

图片中"Output"是指<http://reset.pub/2017/05/19/dnn/>中$V_{IN}$，即激励层的输入数据  $\boldsymbol{W} \cdot \boldsymbol{X} + \boldsymbol{b}$

本文以图片问题中的CNN为例

# CNN基本思想
CNN是DNN的一种特殊形式

考虑使用DNN做图像分类问题，可以将图片的每个像素上每个通道上的数据看成是一个神经元  
假设对于$n \times a$大小的图片，每个像素点有RGB共3个颜色通道，则数据输入层为$3 \times a \times a$
假设有M层隐层，每层有N个神经元则一共有$3 a^2 N^M + MN$个参数。参数量较大。   

针对参数量庞大问题有了卷积神经网络，如以上动态图所示，同一个卷积窗口在遍历一张图片时，参数是共享的
 
以上就CNN中最重要的思想：卷积核及共享权值

# CNN结构基本框架

<img src="/pic/ml/cnn/cnn_struct.png" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[图2，CNN框架示意图](http://cs231n.github.io/convolutional-networks/#overview)</center>  

>
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC

其中：  
CONV：卷积层   
RELU：代表激励层  
POOL：池化层   
FC：全链接层    

# CNN结构说明

## 数据输入层  
图片中每个颜色通道代表一个维度的数据，每个通道上为一张含有位置信息的二维表


## 卷积层

卷积层主要作用是降维

另一种理解是，图片信息中相连区域内的像素点比较相关，通过选定位置相关的区域进行处理可以提高网络的泛化能力

### 滑动窗口  
卷积窗口尺寸：f  
卷积窗口滑动间隔：s  
卷积深度：d  

### 数据对齐

对于输入图像为$a$的正方形图片， 使用宽度为$f$的卷积窗口， 以步长为$s$的方式滑动。一共可以得到
$n^2$个数据，其中：
$$n = \frac{a-f}{s} + 1 \tag{1}$$

对于最后一个滑动窗口可能不能对齐是$n$为非正数，可以通过调整卷积窗口的尺寸和步长来实现对齐
另一个解决滑动不完美的方法是在周围填充0。

每行最少填充个数
$$q=f+(\lceil n \rceil-1)s - a \tag{2}$$
其中$\lceil n \rceil$为$n$的向上取整

则经过调整后
$$n = \frac{a + q -f}{s} + 1  \tag{3}$$

另外<http://cs231n.github.io/convolutional-networks/#overview>中"no zero-padding"的P是指填充圈数，则$q$与$P$的关系为
$$P=\frac{1}{2}q \tag{4}$$


在保证卷机核能卷积操作与数据尺寸对齐的情况下，可以继续使用0填充来扩充数据边界使的原来数据边界作为数据中心达到边界中心化的目的，对于已经通过调整超参数或0填充的方法使的数据可以滑动对齐的长度为$a$的矩阵数据，在固定超参数$f$和$s$的情况下，有效填充个数$q$满足如下：

$$
\left\{
\begin{aligned}
& 0 \leqslant q \leqslant 2(f-1) \\
& (q+a-f) \% s = 0
\end{aligned}
\right.
\tag{5}
$$

'$\%$'：取余符号

### 卷积过程及计算 

全部过程如动图（1）所示 

针对每一个固定卷积窗口，按照超参数左右，上下方向$s$移动，最终得到$n^2$（式（1））组数据。针对某颜色通道上$i$上，对于给定的参数$\boldsymbol{W_i}$，假设滑动窗口内的数据为$\boldsymbol{X_i}$，在feture map中相应位置的值为$\boldsymbol{W_i}\cdot \boldsymbol{X_i}$，综合考虑此次滑动是的对应输出为
$$O_{j, k} = \sum_{i=1}^{d} \cdot \boldsymbol{W_i}\cdot \boldsymbol{X_i} \tag{6}$$
其中$d$为输出feture map中二维表的个数为$d$，一般指原始输入的颜色通道个数3，或者是前一层卷积单元的个数（前一层卷积的深度）

对于输入数数据维度$\boldsymbol{X}$的大小为$(K, a, a)$，其对应的下标分别为 $k$，$i$，$j$。 如果是数据输入层，$K$为颜色通道个数3。这个表示方法和动图（1）中的表示方法有点不一样，主要是考虑高纬度矩阵表示习惯，将最高维放在第一维
卷积窗口的权重矩阵$\boldsymbol{W}$的大小为$(K, f, f)$, 同一卷积层有$m$个filter，则卷积过后形成的feture map大小为$m \times n \times n$


## pooling层  


<img src="/pic/ml/cnn/cnn_pooling.png" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[图3，max pooling示意图](http://cs231n.github.io/convolutional-networks/#overview)</center>  

池化层，通过采样的方式对数据进行降维。pool是给定观察窗口大小和滑动长度，每次滑动一个位置，对窗口选择的数据进行一次下采样：

- Max pooling  
  	求二维表中的最大值

- Average pooling  
	求二维表中的平均

## BN层

BN层，<http://reset.pub/2017/05/19/dnn/>


## 全链接层

层级之间的链接是完全的，同[DNN](http://reset.pub/2017/05/19/dnn/)

作用：增强特征表达能力，即普通深度神经网络的最大优势

<!-- [^_^]: -->

# CNN的反向传播  

<http://reset.pub/2017/05/19/dnn/>

可以将动图（1）展开成传统DNN的结构，由于共享权值，为了方便推导，将权值看成输入神经元，输入看成链接权值可以很好结合dnn中式（13）可以得到，

## 数据层+卷积层+激励层的反向传播

卷积深度超过1的情况下每个输入数据要考虑所有深度的情况，所以每个输入数据的导数递推公式和下一层的数据处理方式有关。
以下讨论卷积深度为1的情况

$$
\begin{aligned}
& \frac{\partial{\ell}}{\partial X_{kk\ ,\ i\ ,\ j_{i}}} 
	= \sum_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}} \ 
		f^\prime \big(V_{IN_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}}} \ \ \  \big) \ 
		\cdot \ 
		W_{kk\ ,\ s_{i}\ ,\ t_{j}} \ 
		\cdot \
		\frac{\partial{\ell}}{\partial O_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}}}
\end{aligned}
\tag{7}
$$

$$
\begin{aligned}
& \frac{\partial{\ell}}{\partial W_{kk\ ,\ s\ ,\ t}} 
    = \sum_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}} \
        f^\prime \big(V_{IN_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}}} \ \ \  \big) \ 
        \cdot \ 
        X_{kk\ ,\ i_{s} \ \ , \ i_{t}} \ 
        \cdot \
        \frac{\partial{\ell}}{\partial O_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}}}
\end{aligned}
\tag{8}
$$

$$
V_{IN_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}}}
= \sum_{kk=1}^{k}V_{IN_{kk\ ,\ i+1\ ,\ j_{i+1}}} 
\ \ = \sum_{kk=1}^{k} \Big(
	\boldsymbol{W}_{kk} \cdot \boldsymbol{X}_{kk\ ,\ area=i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}} 
	+ b
	\Big)
\tag{9}
$$

$$
O_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}} = f\Big(V_{IN_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}}}\ \ \Big)
\tag{10}
$$

- 参数说明  
	- $\ell$:  损失函数
	- $k$上一层的卷积窗口个数或图片数据中通道数据的个数3
	- $kk$：$k$的索引
	- $O_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}}$: 数据层的第$kk$层的第$i$行$j_{i}$列与卷积层的第$kk$维的第$s_{i}$行，第$s_{j}$列对齐时内积经过激励函数的后的输出
	- $f$：激励函数

### 数据层(上层结构输出)++Max pooling反向传播  

$$
\begin{aligned}
& \frac{\partial{O_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}}}}{\partial X_{kk\ ,\ i\ ,\  j_{i}}} \\
=  & \Big(X_{kk\ ,\ i\ ,\  j_{i}} == max \big(area_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}}\big) \Big) \\
=  & \Big(X_{kk\ ,\ i\ ,\  j_{i}} == O_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}} \Big)
\end{aligned}
\tag{11}
$$

对于没次pooling层选中

### 数据层(上层结构输出)+Average pooling反向传播

假设$n$为卷积窗口的宽度

$$
\begin{aligned}
& \frac{\partial{O_{i\ ,\  j_{i} \ \ \leftrightarrow \ \ s_{i}\ ,\  s_{j}}}}{\partial X_{kk\ ,\ i\ ,\  j_{i}}} = \frac{1}{n^2}
\end{aligned}
\tag{12}
$$

# 经典CNN模型

说明见：<http://cs231n.github.io/convolutional-networks/#norm> 及相关论文

LeNet: 1990， <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>  
AlexNet: 2012， <http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>  
ZF Net: 2013， <https://arxiv.org/abs/1311.2901>  
GoogLeNet: 2014， <https://arxiv.org/abs/1409.4842>  
VGGNet: 2014， <http://www.robots.ox.ac.uk/~vgg/research/very_deep/>  
ResNet: 2015，残差网络，<https://arxiv.org/abs/1512.03385>  





# 参考 
\[1\] <http://cs231n.github.io/convolutional-networks/#overview>   
\[2\] 2017.03.15《Deep Learning翻译》<https://exacity.github.io/deeplearningbook-chinese/>  
\[3\] 2015.03.02 Google Inc [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf)   
\[4\] [CNN浅析和历年ImageNet冠军模型解析](https://mp.weixin.qq.com/s/fhJbE6V7r0nhVU3Vu0EGbw)  


