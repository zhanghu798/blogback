layout: post
title: "前向分步算法"
comments: true
tags:
	- 前向分步算法
	- 集成方法
date: 2017-03-31 01:53:56
updated: 2017-04-04 19:49:11
categories:
    - 机器学习
    - 集成方法
---



{% cq %} <font size=4>Forward Stagewise Algorithm</font>{% endcq %}

<!-- more -->


# 算法描述  
+ 输入：
	- 训练数据集：$T＝{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)}$
	- 损失函数：L(y, f(x))。$y$：样本标签向量，$f(x)$：预测结果向量
	- 基函数集：${b(x, \gamma)}$。$\gamma$：模型参数向量， 一组$\gamma$对应一个子模型
+ 输出：
	训练M个模型$b(x, \gamma_m)$，按模型权重$\beta_m$相加得到最红加法模型，如下：
	$$
	\boxed{
	f(x) = \sum_{i=1}^M \beta_m b(x, \gamma_m)
	}
	$$

# 算法流程:
- $f_0(x) = 0$  
- for m in 1, 2, $\ldots$, M:
	- $$
	\begin{multline}
	(\beta_m, \gamma) = arg \min_{\beta, \gamma} \sum_{i=1}^m(y_i, f_{m-1}+\beta b(x_i;\gamma)
	\end{multline}
	$$
	- $$
	\begin{multline}
	f_m(x) = f_{m-1}(x) + \beta_m b(x; \gamma_m)
	\end{multline}
	$$
- $f(x) = f_m(x)$

# 参考资料  
[1]《统计学习方法》，李航著，2012  