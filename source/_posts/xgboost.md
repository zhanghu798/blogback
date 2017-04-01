layout: post
title: "XGBoost"
comments: true
tags:
	- XGBoost
	- 集成方法
date:  2017-04-01 15:18:51
categories:
    - 机器学习  
    - 集成方法
    - GBDT
---

{% cq %} <font size=4>eXtreme Gradient Boosting</font>{% endcq %}

XGBoost是GBDT的一种实现的代码库，同样也是集成方法的一种

相对原始GBDT主要有以下改进  

- 在目标函数中加入模型复杂度  
- 主要利用泰勒展开式的前两项作为目标函数的近似

<!-- more -->

XGboost快于GBDT的说明：  
1，XGboost使用二阶泰勒展开近似目标，使用函数的极值处导数为零，可以一步得到全局极值的近似。  
2，GBDT是给定一个当前收敛最快的方向，每次走一步调整一步，需要多个完成。  
基于以上两点XGBoost的收敛速度快于GBDT。  
这也是也是牛顿法收敛速度快于梯度下降的原因  

XGboost 是由多个回归树boosting而成的结果

# 算法说明  
GBDT算法的目标：损失+正则
$$Obj(\Theta) = L(\theta) + \Omega(\Theta)$$  

* 损失函数：
	- 对于回归问题： 
	$$L(\theta) = \sum_i (y_i-\hat{y}_i)^2$$

	- 对于二分类问题  
	$$L(\theta) = \sum_i[ y_i\ln (1+e^{-\hat{y}_i}) + (1-y_i)\ln (1+e^{\hat{y}_i})]$$

		- $\hat{y}_i = 0,1$，$x_i$样本的预测结果


	- 多分类  
	softmax的负似然对数函数

* 正则项
	$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

	- $T$：叶子节点个数
	- $w_j$：叶子节点代表的回归值，例：CART回归树中落在某一叶子节点$y_i$的平均值
l	- $\gamma$：超参数，叶子节点个数的惩罚系数
	- $\lambda$：超参数，L2-norm平方的系数  

boosting算法的一般形式：
$$\begin{split}\hat{y}_i^{(0)} &= 0\\
\hat{y}_i^{(1)} &= f_1(x_i) = \hat{y}_i^{(0)} + f_1(x_i)\\
\hat{y}_i^{(2)} &= f_1(x_i) + f_2(x_i)= \hat{y}_i^{(1)} + f_2(x_i)\\
&\dots\\
\hat{y}_i^{(t)} &= \sum_{k=1}^t f_k(x_i)= \hat{y}_i^{(t-1)} + f_t(x_i)
\end{split}$$

使用最小二乘法做为损失：
$$
\begin{split}\text{obj}^{(t)} & = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \sum_{i=1}^t\Omega(f_i) \\
          & = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) + constant \\
          & = \sum_{i=1}^n (y_i - (\hat{y}_i^{(t-1)} + f_t(x_i)))^2 + \sum_{i=1}^t\Omega(f_i) + constant \\
          & = \sum_{i=1}^n [2(\hat{y}_i^{(t-1)} - y_i)f_t(x_i) + f_t(x_i)^2] + \Omega(f_t) + constant
\end{split}
$$



# 参考资料
[1]《机器学习》，周志华著，2016  


XGBoost的github地址：<https://github.com/dmlc/xgboost>  
<https://xgboost.readthedocs.io/en/latest/>  

XGBoost的介绍 <https://xgboost.readthedocs.io/en/latest/model.html>