layout: post
title: "循环神经网络"
comments: true
tags:
	- RNN
	- Seq2Seq
date: 2017-11-19 23:28:23
updated: 
categories:
    - 机器学习
    - DL
---


{% cq %} <font size=4>Recurrent Neural Network</font>{% endcq %}

本文以下RNN特指循环神经网络

- 深度神经网络极大的提高了浅层网络的模型拟合数据的能力  
- CNN的出现引入了“感受野”的同时共享参数的机制大大减少DNN的网络参数  
- 而RNN的出现则是用“记忆单元”可以使得有时序的数据随着时间改变记忆，这个记忆贯穿整个时间流，从而有了时序处理的能力  

<!-- more -->



# RNN及变种
## RNN


[Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)


<img src="/pic/ml/rnn/rnn.jpg" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[RNN结构及按时序展开图](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</center>  

---
网络结构的公式
$$\left .
\begin{aligned}
& \text{h}_t = f \big(\text{x}_t \text{U}   +  \text{h}_{t-1} \text{W}  + b_h \big  )  \\
& 
\hat{\text{y}_t} = \text{softmax}( \text{h}_t \text{V}  + b_y)  
\end{aligned}
\right.                                \tag{1}
$$

参数说明：

| 参数                     |     说明 | 
| :-----------:  | :-------------------------------------------------------------------------:| 
| $\text{x}_t$  |    第t时刻的输入。假设：$1 \times n_x$。 $n_x$：一般指输入数据纬度， 如文本问题中，embedding词向量长度大小或one-hot中的长度|
| $\text{h}_t$  |   第t时刻的记忆。假设：$1 \times n_h$。 $n_h$：隐层的维度|
| $f$                      |非线形激励。如：tanh |
| $\text{U}$               |   输入系数矩阵, $n_x \times n_h$ |
| $\text{W}$               |   记忆系数矩阵, $n_h \times n_h$|
| $\text{V}$               |   输出系数矩阵, $n_h \times  \text{vocab_size} 。\text{vocab_size}$：非限定 |
| $\hat{\text{y}}_t $      |  第t时刻的输出, $1 \times  \text{vocab_size}。\text{vocab_size}$:  词库大小 |
| $b_h$                    |  偏置，$1 \times n_h$。更确切的说，这个是属于softmax层的参数 |
| $b_y$                    |  偏置, $1 \times \text{vocab_size} 。\text{vocab_size}$：预测词库大小，非限定。softmax层参数  |


对于RNNLM：

 - 损失函数：
对于第$t$时刻的one-hot形式的标准结果$h_t$, 使用交叉熵

$$
\text{loss }_t = \text{-y}_t \log \hat{\text{y}}_t        \tag{2}
$$

 - 整体loss
$$
\text{Loss} = \sum_{t}^{T} \text{loss}_t        \tag{3}
$$


# BPTT
RNN的训练方法： Backpropagation Through Time


<img src="/pic/ml/rnn/BPTT.jpeg" border="0" width="60%" height="60%" style="margin: 0 auto"><center>[BPTT示意图](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)</center>  


$$
\begin{aligned}
 \frac{\partial{\text{Loss}}}{\partial{W}} 
  = &  \sum_{k=0}^{T}  \frac{\partial{\text{loss}_t}}{\partial{\text{W}_{k}}}  \\
  = &  \sum_{k=0}^{T}  \frac{\partial{\text{loss}_t}}{\partial{\hat{\text{y}}_t}} \
        \frac{\partial{\hat{\text{y}}_t}}{\partial{\hat{\text{h}}_{t}}} \
        \frac{\partial{\hat{\text{h}}_t}}{\partial{\hat{\text{h}}_{k}}} \
        \frac{\partial{\hat{\text{h}}_{k}}}{\partial{\text{W}_{k}}} \\
  = & \sum_{k=0}^{T} \
 		\frac{\partial{L_T}}{\partial{\hat{\text{y}}_t}}  \
  		\frac{\partial{\hat{\text{y}}_t}}{\partial{\hat{\text{h}}_{t}}} \
		\Bigg(\prod_{j=k+1}^{t}  \frac{\partial{\hat{\text{h}}_j}}{\partial{\hat{\text{h}}_{j-1}}}   \Bigg) \
 		\frac{\partial{\hat{\text{h}}_{k}}}{\partial{\text{W}_{k}}}
\end{aligned}  \tag{4}
$$

- 特别说明：
	- $\text{W}$是整个过程中共享的
	- 上式中${\text{W}_{k}}$特指在地$k$时刻的对应的$\text{W}$, 表示第$k$时刻的隐藏（记忆）状态$\text{h}_{k}$是由${\text{W}_{k}}$得到的，
	即，在上式中，$\text{W}$看做是$k$个独立${\text{W}_{k}}$，只是具有相同的值$\text{W}$





$$\frac{\partial{\hat{\text{h}}_j}}{\partial{\hat{\text{h}}_{j-1}}}  = \frac{\partial{\hat{f}_j}}{\partial{\text{v}_{f_{in}}}} W  			\tag{5}$$

将式（5）代入（4）导致RNN做BP时，在长时序列数据训练时容易导致梯度爆炸或远处时刻的损失反映不到参数的梯度上（相对梯度消失）。

对于梯度爆炸，可以通过梯度裁剪的方法使得问题被避免：

$$\boldsymbol{g} = \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g} 		\tag{6}$$

这种形式类似BN层，都是把异常数据强制拉回到正常状态（BN层是将参数分布拉到正态分布，梯度剪裁是强制将过大的梯度设置为1）




## LSTM

1997

[**L**ong **S**hort **T**erm **M**emory](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 

LSTM通过引入“遗忘门”及“记忆叠加门”使得记忆更加灵活，而选择记忆+叠加新记忆更新为新的记忆状态的形式也使得LSTM具有相比RNN的梯度消失和梯度爆炸问题得到较大的缓解，从而使得LSTM可以处理较长时序问题

### 内部结构示意图

<img src="/pic/ml/rnn/lstm.jpg" border="0" width="80%" height="80%" style="margin: 0 auto"><center>[单层LSTM内部结构示意图](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</center></center>  
    
 
### LSTM内部示意图及计算方法   
    
| <img src="/pic/ml/rnn/lstm-forget-gate.jpg" > [忘记门](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  |     <img src="/pic/ml/rnn/lstm-update-status.jpg" > [输入和输入门](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 
| :----------------------------------------------------------------------------: | :--------:|
| <img src="/pic/ml/rnn/lstm-update-memory.jpg" > [更新cell状态](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)                                |   <img src="/pic/ml/rnn/lstm-output.jpg" >  [输出门](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) | 

  
  
   
$$
\left .
\begin{aligned}
&f_t = \sigma(W_f \cdot  [h_{t-1}, x_t] + b_f)   &  & 忘记门，忘记系数（选择保留的系数），决定的丢弃记忆\\
&i_t = \sigma(W_i \cdot  [h_{t-1}, x_t] + b_i)    & & 新的记忆选择系数\\ 
& \widetilde{C}_t = \text{tanh}\big(W_C \cdot  [h_{t-1}, x_t] + b_c\big) &  & 新的记忆 ，决定增强的记忆  \\
&C_t = f_t \odot C_{t-1} + i_t \odot \widetilde{C}_t   & & 更新记忆：t-1时刻状态 \times 选择保留系数 + 新记忆 \times 新记忆系数\\  
& o_t = \sigma \big(W_O[h_{t-1}, x_t] + b_O\big)  & & 上一时刻的隐层状态和本次输入决定本次输出\\
& h_t= o_t \cdot \text{tanh} \big(C_t\big) & & 前一时刻输出和当前网络状态决定当前t时刻隐层状态  \\
\end{aligned}
\right. 
$$

## GRU

2014


<img src="/pic/ml/rnn/gru.jpg" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[gru内部结构示意图](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</center> 

## RNN中的DropOut

将输入向量各个维度随机进行Drop操作，而达到增强泛化的作用， 详见<http://reset.pub/2017/05/19/dnn/#more>


<img src="/pic/ml/rnn/dnn_dropout.png" border="0" width="90%" height="90%" style="margin: 0 auto">

- Bagging
- 有性繁殖 VS 无性繁殖
- 共享参数


以p为dropout概率，则

``` python
is_drop = True if random(0, 1) < p else False

if x_i not is_drop:
  x_i = x_i / (1 - p)

```
    



# Seq2Seq

<img src="/pic/ml/rnn/seq2seq.jpg" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[seq2seq基本示意图](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</center> 

\ \ 

<img src="/pic/ml/rnn/encoder-decoder.jpg" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[encoder-decoder基本示意图](https://github.com/nicolas-ivanov/tf_seq2seq_chatbot)</center> 



## Beam-search

用于最大路径概率预测， 主要是降低时间复杂度。通常是计算时序问题的最大概率路径问题。如HMM，CRF中给定模型参数，求最大概率路径问题。是动态规划每次截断第t-1的前K大路径，传播到第t时刻。
Beam-search是一种介于viterbi和贪心算法见的算法。是最优解和最快效率的一个平衡


### Beam-search示意图

<img src="/pic/ml/rnn/beam-search.gif" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[beam-search](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</center> 

### 求最大概率路径的算法对比

假设每一步都有n个状态，一共有s个时间步 

|算法|时间复杂度|是否最优|
|:------:|:---------------------------------------------:|:------------------:|
|  穷举 | $n^s$ | 是 |
| viterbi | $s \times n^2$| 是|
| 贪心|$s \times n \times k$, 即Beam search中的$k=1$的情况 | 否|
| Beam search | $s \times n \times k$。$1 \leq k \leq n$。 当$k=1$时为贪心算法; 当$k=n$时，为viterbi算法| 否|



## Attention 
<https://github.com/mli/gluon-tutorials-zh/blob/master/chapter_natural-language-processing/nmt.md>

> attention主要解决信息过剩（RNN时间步较长后梯度消失或爆炸）的情况下，通常需要关注某些主要因素时，进行注意力分配的问题

attention的本质是对历史状态加权作为输入的一部分，达到注意力的目的
即如下式1中$C_t$是注意力部分

- 注意力计算部分
$$C_t = \overline{h}_s \odot a_t   	           			  \tag{9}$$
注意力计算如下式，其中$\overline{h}_s$是历史数据向量组成的向量，$a_t$对应的权值

- 注意力和当前状态合并作为输入进行预测

$$\tilde{h}_t = \tanh(W_c[C_t; h_t])  	     			\tag{8}$$

### global-attention(soft attention)

\ \ 

<img src="/pic/ml/rnn/attention-global.jpg" border="0" width="60%" height="60%" style="margin: 0 auto"><center>[attention-global基本示意图](https://arxiv.org/abs/1508.04025)</center> 

\ \ 

<img src="/pic/ml/rnn/attention_1.jpeg" border="0" width="40%" height="40%" style="margin: 0 auto"><center>[归一化的权重计算](https://arxiv.org/abs/1508.04025)</center> 

\ \  

<img src="/pic/ml/rnn/attention_2.jpeg" border="0" width="40%" height="40%" style="margin: 0 auto"><center>[权重计算](https://arxiv.org/abs/1508.04025)</center> 




### local-attention(hard attention)
详见论文 [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)


<img src="/pic/ml/rnn/attection-local.jpg" border="0" width="60%" height="30%" style="margin: 0 auto"><center>[attion-local基本示意图](https://arxiv.org/abs/1508.04025)</center> 


<img src="/pic/ml/rnn/attention_5.jpeg" border="0" width="40%" height="40%" style="margin: 0 auto"><center>[权重计算](https://arxiv.org/abs/1508.04025)</center> 

<img src="/pic/ml/rnn/attention_6.jpeg" border="0" width="40%" height="40%" style="margin: 0 auto"><center>[核心位置计算](https://arxiv.org/abs/1508.04025)</center> 

## seq2seq中的其他trick

- unk replace

- reverse

- feed input

<img src="/pic/ml/rnn/feed-intput.jpg" border="0" width="60%" height="30%" style="margin: 0 auto"><center>[feed-input](https://arxiv.org/abs/1508.04025)</center> 




# 参考

[Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393)  
[Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)   
[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)  
[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)   
[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)   
[[译] 理解 LSTM 网络](http://www.jianshu.com/p/9dc9f41f0b29)   
[Sequence-to-Sequence Learning as Beam-Search Optimization](https://arxiv.org/abs/1606.02960)  
[Sequence to Sequence Learning with Neural Networks](Sequence to Sequence Learning with Neural Networks)    
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
[Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329)  
[How to Use Dropout with LSTM Networks for Time Series Forecasting  ](https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/)  
[An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue)  
[Deep Learning for Natural Language Processing : Hang Li](http://www.hangli-hl.com/uploads/3/4/4/6/34465961/deep_learning_for_natural_language_processing.pdf) 

