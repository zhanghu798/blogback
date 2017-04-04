layout: post
title: "AdaBoost"
comments: true
tags:
	- AdaBoost
	- 集成方法
date:  2017-03-30 22:44:27
updated: 2017-04-02 20:00:27
categories:
    - 机器学习
    - 集成方法
---

{% cq %} <font size=4>Adaptive Boosting</font>{% endcq %}
AdaBoost是多个分类器组合算法，[维基百科AdaBoost算法过程](https://zh.wikipedia.org/wiki/AdaBoost)：

<!-- more -->

# 算法描述
+ 输入：
	- 训练数据集：$T＝{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)}$
	- 基函数模型
+ 输出：
	训练M个模型$G_m(x), m \in [1, M]$，按模型权重$\alpha_m, m \in [1, N]$相加得到最红加法模型，如下：
	$$
	\sum_{i=1}^M \alpha_m G_m(x)
	$$

# 算法流程
* 初始化训练数据权重 $w_{1i}=\frac{1}{N}，i= 1,2,\ldots,N$
* for m in 1, 2, $\ldots$, M  #(训练M个基分类器):
	- 按样本权重分布$w_{m}$采样$D_m$训练集
	- 在$D_m$上训练基本分类器$G_m(x)$
	- 计算$G_m(x)$在训练数据集上的分类误差率：
	$$
	e_m = P\big(G_m(x_i) \neq y_i\big)
	= \sum_{i=1}^N w_{mi}I \big(G_m(x_i) \neq y_i\big)
	$$  
	- 计算基本分类器 $G_m(x)$的权重：
	$$\alpha_m = \frac{1}{2}log \frac{1 - e_m}{e_m}$$
	- 更新样本权重
	$$
	w_{m+1,i} = \frac{w_{m,i}}{Z_m} \times 
	\left\{
	\begin{aligned}
	& e^{-\alpha_m}，& if G_m(x_i) = y_i \\
	& e^{\alpha_m}，& if G_m(x_i) \neq y_i \\
	\end{aligned}
	\right.
	$$
	
	$$
	Z_m = \sum_{i=1}^N
	w_{m,i} \times \left\{
	\begin{aligned}
	& e^{-\alpha_m}，& if G_m(x_i) = y_i \\
	& e^{\alpha_m}，& if G_m(x_i) \neq y_i 
	\end{aligned}
	\right.
	$$

* 输出分类器
$$
\boxed{
G(x) = sign(f(x)) = sign \Big(\sum_{m=1}^M \alpha_m G_m(x)\Big)
}
$$

# 样本权重调整方法
- 对于样本按样本权重抽样  
优点：样本集可以直接用现成的分类模型去拟合  
缺点：对于样本权重 0.3333这种需要抽样凑成整数时个样本量较大 ，或者样本权重有损失  
- 修改分类器损失或特征选择时的条件(决策树)  
优点：可以较好的保持样本权重  
缺点：需要造基本分类器轮子  

# AdaBoost与向前分布算法的关系
AdaBoost是[向前分步算法]<http://reset.pub/2017/03/31/forward-stagewise-algorith/> 算法的特例，是由基本分类器组成的加法模型，损失函数为指数函数


# 参考资料  
[1]《统计学习方法》，李航著，2012  
	

