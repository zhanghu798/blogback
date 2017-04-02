layout: post
title: "XGBoost"
comments: true
tags:
	- XGBoost
	- 集成方法
date:  2017-04-01 15:18:51
updated: 2017-04-01 21:46:35
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

## 决策树的性能度量
GBDT算法的目标：损失+正则
$$Obj(\Theta) = L(\theta) + \Omega(\Theta)$$  


* 损失函数：
以下描述中$\hat{y}_i$为预测结果
	- 对于回归问题： 
	$$L(\theta) = \sum_i (y_i-\hat{y}_i)^2$$

	- 对于二分类问题  
	$$L(\theta) = \sum_i[ y_i\ln (1+e^{-\hat{y}_i}) + (1-y_i)\ln (1+e^{\hat{y}_i})]$$

	- 多分类  
	softmax的负似然对数函数

* 正则项
	$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

	- $T$：叶子节点个数
	- $w_j$：叶子节点代表的回归值，例：CART回归树中落在某一叶子节点$y_i$的平均值
	- $\gamma$：超参数，叶子节点个数的惩罚系数
	- $\lambda$：超参数，L2-norm平方的系数  

## XGBoost的一般形式
boosting算法的一般形式：
$$\begin{split}\hat{y}_i^{(0)} &= 0\\
\hat{y}_i^{(1)} &= f_1(x_i) = \hat{y}_i^{(0)} + f_1(x_i)\\
\hat{y}_i^{(2)} &= f_1(x_i) + f_2(x_i)= \hat{y}_i^{(1)} + f_2(x_i)\\
&\dots\\
\hat{y}_i^{(t)} &= \sum_{k=1}^t f_k(x_i)= \hat{y}_i^{(t-1)} + f_t(x_i)
\end{split}$$


$$
\begin{split}\text{obj}^{(t)} & = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \sum_{i=1}^t\Omega(f_i) \\
          & = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) + constant
\end{split}
$$

## 决策树决策规则
>
泰勒展式：
$$
f(x) = f(a) + \frac{f'(a)}{1!}(x-a)+ \frac{f^{2}(a)}{2!}(x-a)^2 + \cdots + \frac{f^{(n)}(a)}{n!}(x-a)^n + R_n(x)
$$

$\begin{aligned}
令， &f(x) = l\big(y_i， \hat{y}_i^{(t-1)} + f_t(x_i)\big)，f(a) = l(y_i， \hat{y}_i^{(t-1)}) \\
则，& x - a = f_t(x_i)
\end{aligned}$




则目标函数的二阶泰勒展开为：
$$
\text{obj}^{(t)} \approx 
 \sum_{i=1}^n \big[ l(y_i， \hat{y}_i^{(t-1)}) + g_i \cdot f_t(x_i) + \frac{1}{2} \cdot h_i \cdot f_t^2(x_i)\big] +\sum_{i=1}^t\Omega(f_i)
$$

$$
g_i = \frac{\partial l(y_i， \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}  \\
h_i = \frac{\partial^2 l(y_i， \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}} 
$$

定义好损失函数，前$t-1$棵树训练好后，$g_i$ 和 $h_i$就确定了。$g_i$ 和 $h_i$是前$t-1$棵树和第$t$棵树之间的纽带，也体现Boosting思想

考虑到是对$\text{obj}^{(t)}$求最小值，前$t-1$棵确定下来后$l(y_i， \hat{y}_i^{(t-1)})$为定值，另外前$t-1$棵树的正则项也为常数，即对于目标而言 $\sum_{i=1}^t\Omega(f_i) = constant + \Omega(f_t)$

所以目标可以表示为：
$$
\begin{split}
 \text{obj}^{(t)} 
& \approx  \sum_{i=1}^n \big[g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)\big] + \Omega(f_t) \\
& = \sum_{i=1}^n [g_i f_t(x_i)  + \frac{1}{2} h_i f_t^2(x_i) ] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2\\
\end{split}
$$

其中$f_t(x_i)$表示样本$x_i$在第$t$棵树上的预测结果，假设设n个样本在第$t$棵树上的预测结果分布在$T$个叶子节点上，则某一叶子节点$I_j$上有必相同的回归值$w_j$，则有
$$
\begin{split}
\sum_{i\in I_j} g_i  f_t(x_i) = (\sum_{i\in I_j} g_i) \cdot w_j \\
\end{split}
$$

则：
$$
\begin{split}
\text{obj}^{(t)} 
\approx \sum^T_{j=1} [(\sum_{i\in I_j} g_i) w_j + \frac{1}{2} (\sum_{i\in I_j} h_i + \lambda) w_j^2 ] + \gamma T
\end{split}
$$

$$
令：G_j = \sum_{i\in I_j} g_i，H_j = \sum_{i\in I_j} h_i
$$

$$
\text{obj}^{(t)} \approx  \sum^T_{j=1} \big[G_jw_j + \frac{1}{2} (H_j+\lambda) w_j^2\big] +\gamma T
$$

当$\text{obj}^{(t)}$取得极小值时， $\frac{\partial \text{obj}^{(t)}}{\partial w_j} = 0$，则：
$$
w_j^\ast = -\frac{G_j}{H_j+\lambda}
$$

则决策树的损失为：
$$\text{obj}^\ast = -\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j+\lambda} + \gamma T
$$

决策树损失越小越好，类别成CART的基尼系数。则决策树某个节点分裂前后的的增益的增益为：父节点点的损失 - 左子树的损失 - 右子树的损失。不分裂：全部样本子一个叶子结点；分裂：左子树（叶）和右子树（叶）都先看成叶子，则$T=1$，决策树分裂前后的增益为：
$$
Gain = \frac{1}{2} \left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma
$$

决策树建立过程是寻找使得增益$Gain$最大的特征及特征上的值过程

# 伪代码



# 参考资料
[1]《机器学习》，周志华著，2016  


XGBoost的github地址：<https://github.com/dmlc/xgboost>  
<https://xgboost.readthedocs.io/en/latest/>  

XGBoost的介绍 <https://xgboost.readthedocs.io/en/latest/model.html>

