layout: post
title: "Gradient Boosting"
comments: true
tags:
	- Boosting
	- GB
	- GBDT
	- 集成方法
date:  2017-04-03 21:20:54
updated: 2017-04-04 16:30:07
categories:
    - 机器学习
    - 集成方法
---

 
 {% cq %} <font size=4>Gradient Boosting，梯度提升</font>{% endcq %}
 
 提升，梯度提升，梯度提升树
<!-- more -->

# Boosting，提升  

梯度提升树是有多棵树共同决策而成的
# 模型
$$F_m(x) = \sum_{i=1}^m f_m(x) = F_{m-1}(x) + f_m(x) \tag{1}$$

- $F_m(x)$：由m个子模型构成的模型
- $f_m(x)$：第m个子模型

构建过程：

* 1，初始化
$$F_0(x) = \mathop{\arg\min}\limits_{\gamma} \sum_{i=1}^n L(y_i, \gamma) \tag{2}$$   
	* $L(y_i, \gamma)$：损失函数
    * $y_i$：回归目标
- 2，训练第m棵树$$
F_m(x) = F_{m-1}(x) + \mathop{\arg\min}\limits_{f} \sum_{i=1}^n L\big(y_i, F_{m-1}(x_i) + f(x_i)\big)  \tag{3}
$$

## 平方损失的提升
对于损失函数$L(y_i, \hat{y}_i) = (y_i - \hat{y}_i) ^2时$

第m棵树构建时，第i个样本取得最小值时，须使得损失函数一阶导在$F_{m-1}(x_i) + f_m(x_i)$处等于零，即：
$$2 \cdot \big(y_i - F_{m-1}(x_i) - f_m(x_i)\big)= 0$$
即，$$f_m(x_i) = y_i - F_{m-1}(x_i)$$

算法步骤

- 初始化$f_0(x) = \mathop{\arg\min}\limits_{\gamma} \sum_{i=1}^n (y_i, \gamma)^2 = \frac{1}{n} x$
- for m in 1，2，$ldots$, M: # 建立第m棵树
	- $r_{mi} = y_i  - f_{m-1}(x_i), i = 1,2, \cdots, N$
  	- 拟合残差向量$r_{m}$学习一个回归树，得到$f_m(x)$
  	- $F_m(x) = F_{m-1}(x) + f_m(x)$
- $F_M(x) = \sum_{m=1}^M f_m(x)$




# Gradient boosting，梯度提升  

## 问题表达

梯度提升是提升在损失函数不方便直接求极值时的扩展。梯度提升使用梯度下降求极值

考虑   

>
$f = \min l(\theta)$，$l(\theta)$是凸的。
利用梯度下降，步长为$\gamma$的梯度下降为：
$$
\theta^{(t)}= \theta^{(t-1)} - \gamma \bigg[\frac{ \partial l(\theta)} {\partial \theta}\bigg]_{\theta = \theta^{(t-1)} }
$$

原问题可以重新表达为在已知$F_{m-1}(x)$的情况下，更新参数$F_{m}(x)$替代$F_{m-1}(x)$，使得$L \big(y_i, F(x)\big)$最小化，则，可以利用梯度下降的方式：
$$
F_m(x) = F_{m-1}(x) - \bigg( \gamma_m \sum_{i=1}^n \frac{\partial L(y_i,Z_i)}{\partial Z_i)} \bigg)_{Z_i = F_{m-1} \ \ \ (x_i)} \tag{4}
$$

$$
\gamma_m = \mathop{\arg\min}\limits_{\gamma} \sum_{i=1}^n \Bigg(
 F_{m-1}(x_i) - \gamma \bigg[ \frac{\partial L(y_i,Z_i)}{\partial Z_i} \bigg]_{Z_i =  F_{m-1} \ \ (x_i) }
 \Bigg) \tag{5}
$$

$\gamma_m$ 可以通过[线性搜索](https://en.wikipedia.org/wiki/Line_search)得到

## [伪代码](https://en.wikipedia.org/wiki/Gradient_boosting)


<img src="/pic/ml/gbdt/gbdt_Gradient-boosting.png" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[图1 梯度提升伪代码](https://en.wikipedia.org/wiki/Gradient_boosting)</center>

# Gradient Boosting Decision Tree， 梯度提升树  
梯度提升树是梯度提升的一个特例，基模型由决策回归树构建

给定一棵树，任何样本样本在树的规则下映射到叶子结点，叶子结点的值即为该样本的预测值，假设有$J_m$个叶子结点，每个叶子的值为$b_{jm}$，则树的输出$h_m(x)$可以写成
$$
h_m(x) = \sum_{j=1}^{J_m} b_{jm} I (x \in R_{jm}) \tag{6}
$$

如果将$h_m(x)$看成梯度提升中拟合的伪残差$r_{mi}$， 则梯度提升树可以表示为：
$$
F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)
$$
将$h_m(x)$带入上式得：
$$
\gamma_m = \mathop{\arg\min}\limits_{\gamma} \sum_{i = 1}^n L(y_i, F_{m-1(x_i) + \gamma h_m(x_i)})
$$
	

$$
F_m(x) = F_{m-1}(x) + \gamma_m \sum_{j=1}^{J_m} b_{jm} I (x \in R_{jm}) 
$$

令：$$\gamma_{jm} = \gamma_m \cdot b_{jm}$$
则：
$$
F_m(x) =  F_{m-1}(x) + \sum_{j=1}^{J_m} \gamma_{jm} I (x \in R_{jm}) \tag{7}
$$

$$
\gamma_m = \mathop{\arg\min}\limits_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)
\tag{8}
$$
	
# 模型改进
1，修改模型权重
$$F_m(x) = F_{m-1} + \nu \cdot \gamma_m h_m(x)， 0 < \nu \leqslant 1 \tag{9}$$
引入后需要更多的子模型个数

2，使用随机梯度下降代替梯度下降  
3，加入bagging采样构建每个子模型  
4，增加了模型复杂度  

# 参考资料  
[1]《统计学习方法》，李航著，2012  
[2] https://en.wikipedia.org/wiki/Gradient_boosting  
[3] XGboost的GitHub地址：<https://github.com/dmlc/xgboost>    

