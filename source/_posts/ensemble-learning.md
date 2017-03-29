layout: post
title: "集成方法"
comments: true
tags:
	- 集成学习
date:  2017-03-27 14:23:37
categories:
    - 机器学习
---

这个题目取得有点大，就当先挖个坑  

# 两种基本集成学习思路

## Bagging  
Bootstrap aggregating  

>
给定一个大小为$n$的训练集$D$，Bagging算法从中均匀、有放回地选出$m$个大小为$n'$的子集$D_{i}$，作为新的训练集。在这$m$个训练集上使用分类、回归等算法，则可得到 $m$个模型，再通过取平均值、投票等方法，即可得到Bagging的结果  

对于样本总量为m，有放回的的随机抽样m次， 某一样本至少抽到一次的概率：
$$
\begin{aligned}
p = & 1 - (\frac{m - 1}{m}) ^ m \\
= & 1 - \frac{1}{(1 + \frac{1}{-m})^{-m}} \\
\geqslant & 1 - \frac{1}{e} \ \ \ \ (当m \rightarrow \infty时，取等号)\\
\geqslant & 63.2％ 
\end{aligned}
$$


上式中e为[自然常数](https://zh.wikipedia.org/wiki/E_(数学常数)) 。将m看做未知数，则$1 - \frac{1}{m}$ 为增函数，则$(1 - \frac{1}{m}) ^ m$为增函数， 则$p$为减函数，所以当$m \rightarrow \infty$时，取得极大值

能够减小训练方差  
简单说明，考虑以下极端情况

* 假设n个模型是完全独立的，即各模型任意组合的协方差为0
	假设各个模型的方差均为$var$， 则
	$$
	\begin{aligned}
	& VAR\big[\frac{1}{n}(X_1 + X_2 + \ldots + X_n)\big]  \\
	= & \sum_{i=1}^n var(\frac{1}{n}X_1) \\
	= & n \cdot \frac{1}{n^2} var \\
	= & \frac{1}{n} var \\
	\end{aligned}
	$$

* 假设各模型完全相同，则：
$$
VAR\big[\frac{1}{n}(X_1 + X_2 + \ldots + X_n)\big] = VAR(\frac{1}{n} \cdot n X_i) = var(x)
$$  

现实情况介于两者之间，所以Bagging可以降低方差


## Boosting  
> 通过多个弱分类器集成为强分类器方法的统称 

根据以上定义，Bagging是boosting的一个子集，但是很多资料上Boosting特指，各个模型是训练基于整体模型的误差训练的。如AdaBost是串行训练模型，第n＋1
个模型是针对前n个模型集成后的误差来训练的。下文也是将Boosting方法看成是多个模型基于误差协作组成强模型的的方法

Boosting的主要特点：弱分类器之间有依赖关系

因为Boosting的误差相关联性，所以Boosting是偏向于降低误差

# 随机森林  
Random Forests，<https://zh.wikipedia.org/wiki/随机森林> ， 是Bagging的一种实现

>训练n不剪枝，没棵树是有部分样本中的部分特征组成，对结果进行投票或取平均  
>分类问题：多个ID3、C4.5、C5.0或CART分类树的方法结果进行投票  
>回归问题：多个不剪枝CART回归树结果求平均  




# AdaBost  
Adaptive Boosting

# GBDT  
Gradient Boosting Decision Tree  

## XGBoost  
Extreme Gradient Boosting  
<https://xgboost.readthedocs.io/en/latest/model.html>

## LightGBM  
Light Gradient Boosting Machine  
<http://www.msra.cn/zh-cn/news/blogs/2017/01/lightgbm-20170105.aspx>

知乎地址：<https://github.com/Microsoft/LightGBM>

## gcFrest  
multi-Grained Cascade forest  
<https://arxiv.org/pdf/1702.08835.pdf>



# 参考

<http://wenda.chinahadoop.cn/question/4155>






