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
& 0 & if \ \  & \lvert y -  \widehat{y} \rvert < \epsilon \\
& \lvert y -  \widehat{y} \rvert - \epsilon &  & otherwise
\end{aligned}
\right. \tag{1}
$$  
上式 $\epsilon > 0$，epsilon的意义：在离回归线上下$\epsilon$都不计入损失，其回归线可以看成回归线上下$\epsilon$区域边界组成的回归带，落在在回归带上的点x损失为0，预测$\widehat{y}$为固定x后回归带的中点

利用误差＋L2正则 形式：
$$J = C \sum_{i=1}^m  L_\epsilon(y, \widehat{y}) + \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 \tag{2}$$ 

设所求回归线为$f(x) = \boldsymbol{w} \cdot \boldsymbol{x} + b$  
对于固定的$\epsilon$，所有点均在回归带上的情况：
$$
\left\{
\begin{aligned}
obj\ :\ \ & \min_{\boldsymbol{w},b,\epsilon} \frac{1}{2}\lVert\boldsymbol{w}\rVert^2  \\
st\ :\ \ & \lvert  f(\boldsymbol{x_i})- y_i \rvert  \leqslant \epsilon
\end{aligned}
\right. \tag{3}
$$  

对于宽度为$2\epsilon$的回归带不能覆盖所有点的情况下：
参照SVM的线性不可分的情况，为每个样本引入松弛因子$\xi_i$使得不在回归带的点满足约束，并且在目标函数中加入对松弛因子的惩罚项$C$，则有

$$
\left\{
\begin{aligned}
obj\ :\ \ & \min_{\boldsymbol{w},b,\epsilon} \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 ＋ C \xi_i  \\
st\ :\ \ & \lvert  f(\boldsymbol{x_i})  - y_i \rvert \leqslant \epsilon + \xi_i \\
& \xi_i \geqslant 0
\end{aligned}
\right. \tag{4}
$$  

考虑上式约束不连续可导，分情况去掉绝对值： 
$$
\begin{eqnarray}  
隔离带上方的样本：& f(x_i) &            &            & <          & y_i     <        f(x_i) + \epsilon + \xi^+_i  \\
隔离带下方的样本：& f(x_i) & - \epsilon &  - \xi^-_i &  <         & y_i     <        f(x_i)  \\
隔离带上的样本：  & f(x_i) & - \epsilon &            &  \leqslant & y_i  \leqslant   f(x_i) + \epsilon 
\end{eqnarray}  
$$

综上约束条件可以化为：  $$f(x_i) - \epsilon   - \xi^-_i \leqslant y_i  \leqslant f(x_i) + \epsilon + \xi^+_i$$

式(4)中将$f(x) = \boldsymbol{w} \cdot \boldsymbol{x} + b$，替换等价约束可以转化为：
$$
\left.
\begin{aligned}
obj\ :\ \ & \min_{\boldsymbol{w},b,\boldsymbol{\xi}} \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 ＋ C\sum_{i=1}^m(\xi^+_i ＋ \xi^-_i ) \\
st\ :\ \ &  \boldsymbol{w} \cdot \boldsymbol{x_i} + b - \epsilon - \xi^-_i - y_i \leqslant 0\\
&  y_i - \boldsymbol{w} \cdot \boldsymbol{x_i} - b - \epsilon  - \xi^+_i   \leqslant 0   \\
& -\xi^+_i \leqslant 0 \\
& -\xi^-_i \leqslant 0 \\
& C，\epsilon 为超参数，C >0，\epsilon \geqslant 0 
\end{aligned}
\right. \tag{5}
$$  
对于一个样本点$x_i$其松弛因子$\xi^+_i$和$\xi^-_i$$至多只有一个大于0  

式(5)为不带等式约束的[凸优化](http://0.0.0.0:4000/2017/03/18/convex-optimization/)，

引入拉格朗日乘子$a_i^＋ \geqslant 0$， $a_i^－ \geqslant 0$，$mu^+_i \geqslant 0$和$mu^-_i \geqslant 0$，则拉格朗日函数为：
$$
\begin{aligned}
L(\boldsymbol{w},b,\boldsymbol{\xi^+},\boldsymbol{\xi^-},\boldsymbol{\mu^+},\boldsymbol{\mu^-})= 
& \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 ＋ C\sum_{i=1}^m(\xi^+_i ＋ \xi^-_i ) \\ 
& + \sum_{i=1}^m a_i^-(\boldsymbol{w} \cdot \boldsymbol{x_i} + b - \epsilon - \xi^-_i - y_i)\\
& + \sum_{i=1}^m a_i^+(y_i - \boldsymbol{w} \cdot \boldsymbol{x_i} - b - \epsilon  - \xi^+_i ) \\
& - \sum_{i=1}^m \mu_i^+ \xi_i^+ \\
& - \sum_{i=1}^m \mu_i^- \xi_i^-  
\end{aligned}
$$

则：

$$
\begin{eqnarray}
\nabla_{\boldsymbol{w}}L = 0 & \Longrightarrow & \boldsymbol{w} + \sum_{i=1}^m a_i^- \boldsymbol{x_i} - \sum_{i=1}^m a_i^+ \boldsymbol{x_i} = 0 \tag{8} \\
\nabla_{\boldsymbol{b}}L = 0 & \Longrightarrow & \sum_{i=1}^m a_i^+   - \sum_{i=1}^m a_i^- ＝ 0 \tag{8}\\
\nabla_{\boldsymbol{\xi_i^+}}L = 0 & \Longrightarrow  & C - a_i^+ - \mu_i^+ = 0 \tag{8} \\
\nabla_{\boldsymbol{\xi_i^-}}L = 0 & \Longrightarrow & C - a_i^- - \mu_i^- = 0 \tag{8}\\
\end{eqnarray}
$$

将上式代入$L$中，整理相关参数的KKT条件，得到拉格朗日对偶函数如下：

。。。



