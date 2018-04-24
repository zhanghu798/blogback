layout: post
title: "RNN及Seq2Seq"
comments: true
tags:
	- RNN
	- Seq2Seq
date: 2017-12-19 23:28:23
updated: 
categories:
    - 机器学习
    - DL
---


{% cq %} <font size=4>Recurrent Neural Network</font>{% endcq %}

介绍从RNN到Seq2Seq的基本组件

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
| $\text{x}_t$  |    第t时刻的输入。假设：$ 1\times n_x$|
| $\text{h}_t$  |   第t时刻的记忆。假设：$1 \times n_h$|
| $f$                      |非线形激励。tanh |
| $\text{U}$               |   输入系数矩阵, $n_x \times n_h$ |
| $\text{W}$               |   记忆系数矩阵, $n_h \times n_h$|
| $\text{V}$               |   输出系数矩阵, $n_h \times  \text{vocab_size} 。\text{vocab_size}$：非限定 |
| $\hat{\text{y}}_t $      |  第t时刻的输出, $1 \times  \text{vocab_size}。\text{vocab_size}$:  词库大小 |
| $b_h$                    |  偏置，$1 \times n_h$ |
| $b_y$                    |  偏置, $1 \times \text{vocab_size} 。\text{vocab_size}$：预测词库大小，非限定  |


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

 - 训练：BPTT,  Backpropagation Through Time


<img src="/pic/ml/rnn/BPTT.jpeg" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[BPTT示意图](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)</center>  


$$
\begin{aligned}
 \frac{\partial{\text{loss}_t}}{\partial{W}} 
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


$$\frac{\partial{\hat{\text{h}}_j}}{\partial{\hat{\text{h}}_{j-1}}}  = \frac{\partial{\hat{f}_j}}{\partial{\text{v}_{f_{in}}}} W  			\tag{5}$$

将式（5）代入（4）导致RNN做BP时，在长时序列数据训练时容易导致梯度爆炸或远处时刻的损失反映不到参数的梯度上（相对梯度消失）。
对于梯度爆炸，可以通过梯度裁剪的方法避免：

$$\boldsymbol{g} = \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g} 		\tag{6}$$


----

## LSTM

**L**ong **S**hort **T**erm **M**emory

1997

>人类并不是每时每刻都从一片空白的大脑开始他们的思考。在你阅读这篇文章时候，你都是基于自己已经拥有的对先前所见词的理解来推断当前词的真实含义。我们不会将所有的东西都全部丢弃，然后用空白的大脑进行思考。我们的思想拥有持久性。


<img src="/pic/ml/rnn/lstm.jpg" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[lstm](./lstm.jpg)<center>[单层LSTM内部结构示意图](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</center></center>  



<img src="/pic/ml/rnn/BPTT.jpeg" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[BPTT示意图](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)</center>   

| <img src="/pic/ml/rnn/lstm-forget-gate.jpg" > [忘记门](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  |     <img src="/pic/ml/rnn/lstm-update-status.jpg" > [输入和输入门](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 
| :----------------------------------------------------------------------------: | :--------:|
| <img src="/pic/ml/rnn/lstm-update-memory.jpg" > [更新cell状态](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)                                |   <img src="/pic/ml/rnn/lstm-output.jpg" >  [输出门](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) | 


$$
\left .
\begin{aligned}
&f_t = \sigma(W_f \cdot  [h_{t-1}, x_t] + b_f)   &  & 忘记门，忘记系数，决定的丢弃记忆\\
&i_t = \sigma(W_i \cdot  [h_{t-1}, x_t] + b_i)    & & 新的记忆 ，决定增强的记忆\\ 
& \widetilde{C}_t = \text{tanh}\big(W_C \cdot  [h_{t-1}, x_t] + b_c\big) &  & 新的记忆选择系数，记忆增强/降低的系数 \\
&C_t = f_t \odot C_{t-1} + i_t \odot \widetilde{C}_t   & & 更新记忆：忘记 \times 忘记系数 + 新记忆 \times 新记忆系数\\  
& o_t = \sigma \big(W_O[h_{t-1}, x_t] + b_O\big)  & & 上一时刻的输出和本次输入决定本次的主要结果\\
& h_t= o_t \cdot \text{tanh} \big(C_t\big) & & 前一时刻状态和输入，通过记忆的输出单元的转化  \\
\end{aligned}
\right. \tag{7}
$$

## GRU

2014


<img src="/pic/ml/rnn/gru.jpg" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[gru内部结构示意图](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</center> 

## RNN中的DropOut


<img src="/pic/ml/rnn/dnn_dropout.png" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[gru内部结构示意图](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</center> 

- Bagging
- 有性繁殖 VS 无性繁殖
- 共享参数


以p为dropout概率，则

	# p: define dropout rate. 0 <= p < 1
    is_drop = True if random(0, 1) < p else False

	if x_i not is_drop:
		x_i = x_i / (1 - p)
    



# Seq2Seq

<img src="/pic/ml/rnn/seq2seq.jpg" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[seq2seq基本示意图](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</center> 

<img src="/pic/ml/rnn/encoder-decoder.jpg" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[encoder-decoder基本示意图](https://github.com/nicolas-ivanov/tf_seq2seq_chatbot)</center> 



## Beam-search


<img src="/pic/ml/rnn/beam-search.gif" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[beam-search](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</center> 

**穷举** vs **Viterbi** vs **Beam search** vs **贪心**

## Attention 

>注意力（Attention）是大脑神经系统解决信息超载问题的主要手段，是在计算能力有限情况下的一种资源分配方案，将计算资源分配给更重要的任务

$$\tilde{h}_t = \tanh(W_c[C_t; h_t])  	     			\tag{8}$$
$$C_t = \overline{h}_s \odot a_t   	           			  \tag{9}$$

### global-attention(soft attention)


<img src="/pic/ml/rnn/attention-global.jpg" border="0" width="60%" height="60%" style="margin: 0 auto"><center>[attention-global基本示意图](https://arxiv.org/abs/1508.04025)</center> 


<img src="/pic/ml/rnn/attention_1.jpeg" border="0" width="40%" height="40%" style="margin: 0 auto"><center>[得分指数化、归一化](https://arxiv.org/abs/1508.04025)</center> 

  

<img src="/pic/ml/rnn/attention_2.jpeg" border="0" width="40%" height="40%" style="margin: 0 auto"><center>[得分](https://arxiv.org/abs/1508.04025)</center> 




### local-attention(hard attention)

<img src="/pic/ml/rnn/attection-local.jpg" border="0" width="60%" height="30%" style="margin: 0 auto"><center>[attion-local基本示意图](https://arxiv.org/abs/1508.04025)</center> 

<img src="/pic/ml/rnn/attention_3.jpeg" border="0" width="40%" height="40%" style="margin: 0 auto"><center>[attention](https://arxiv.org/abs/1508.04025)</center> 

<img src="/pic/ml/rnn/attention_5.jpeg" border="0" width="40%" height="40%" style="margin: 0 auto"><center>[attention](https://arxiv.org/abs/1508.04025)</center> 

<img src="/pic/ml/rnn/attention_6.jpeg" border="0" width="40%" height="40%" style="margin: 0 auto"><center>[attention](https://arxiv.org/abs/1508.04025)</center> 

## seq2seq中的其他trick


<img src="/pic/ml/rnn/seq2seq_trick.jpg" border="0" width="80%" height="80%" style="margin: 0 auto"><center>[seq2seq的trick效果-结果来自论文](https://arxiv.org/abs/1508.04025)</center> 



<img src="/pic/ml/rnn/feed-intput.jpg" border="0" width="60%" height="30%" style="margin: 0 auto"><center>[feed-input](https://arxiv.org/abs/1508.04025)</center> 



seq2seq及attention可参考 <https://github.com/mli/gluon-tutorials-zh/blob/master/chapter_natural-language-processing/nmt.md>


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

