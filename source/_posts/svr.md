layout: post
title: "SVR"
comments: true
tags:
	- SVR
date: 2017-11-24 23:52
categories:
    - 机器学习
---

{% cq %} <font size=4>SVR，Support Vector Regression，支持向量回归</font>{% endcq %}

<!-- more -->

假设样本结果为y, 预测结果为$\widehat{y}$，定义epsilon不敏感损失为：
$$
L_\epsilon(y, \widehat{y}) = 
\left\{
\begin{aligned}
& 0 & if & \|y -  \widehat{y}\| < \epsilon \\
& \|y -  \widehat{y}\| - \epsilon &  & otherwise
\end{aligned}
\right. \tag{1}
$$  
上式 $\epsilon > 0$，epsilon的意义：在离回归线上下$\epsilon$都不计入损失，其回归线可以看成回归线上下$\epsilon$区域边界组成的回归带，落在在回归带上的点x损失为0，预测$\widehat{y}$为固定x后回归带的中点

利用误差＋L2正则 形式：
$$J = C \sum_{i=1}^m  L_\epsilon(y, \widehat{y}) + \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 \tag{2}$$ 

对于固定的$\epsilon$，所有点均在回归带上的情况：
$$
\left\{
\begin{aligned}
obj\ :\ \ & \min_{\boldsymbol{w},b,\epsilon} \frac{1}{2}\lVert\boldsymbol{w}\rVert^2  \\
st\ :\ \ & \lvert  \boldsymbol{w} \cdot \boldsymbol{x_i} + b - y_i \rvert  < \epsilon
\end{aligned}
\right. \tag{3}
$$  

对于宽度为$2\epsilon$的回归带不能覆盖所有点的情况下：
参照SVM的线性不可分的情况，为每个样本引入松弛因子$\xi_i$使得不在回归带的点满足约束，并且在目标函数中加入对松弛因子的惩罚项$C$，则有

$$
\left\{
\begin{aligned}
obj\ :\ \ & \min_{\boldsymbol{w},b,\epsilon} \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 ＋ C \xi_i  \\
st\ :\ \ & \lvert  \boldsymbol{w} \cdot \boldsymbol{x_i} + b - y_i \rvert < \epsilon + \xi_i \\
& \xi_i > 0
\end{aligned}
\right. \tag{4}
$$  

考虑上式约束不连续可导，去掉绝对值：
$$
\left\{
\begin{aligned}
obj\ :\ \ & \min_{\boldsymbol{w},b,\epsilon} \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 ＋ C \xi_i  \\
st\ :\ \ &  \boldsymbol{w} \cdot \boldsymbol{x_i} + b - y_i  < \epsilon + \xi_i  \\
&  \boldsymbol{w} \cdot \boldsymbol{x_i} + b - y_i  > -\epsilon - \xi_i \\
& \xi_i > 0
\end{aligned}
\right. \tag{5}
$$  
