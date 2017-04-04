layout: post
title: "集成方法"
comments: true
tags:
	- 集成方法
date:  2017-04-04 23:55:28
updated: 2017-04-05 02:53:06
categories:
    - 机器学习
    - 集成方法
---

{% cq %} <font size=4>Ensemble methods</font>{% endcq %}

集成方法是将多个基模型集成起来以提高最终模型的预测准确率或泛化能力的方法
<!-- more --> 

# 两种基本集成学习思路

## Bagging  
Bootstrap aggregating  

### Bagging思想  

>
给定一个大小为$m$的样本集$D$，Bagging算法从中均匀、有放回地选出$n$个大小为$s$的子集$D_{i}$，作为新的训练集。在这$t$个训练集上使用分类、回归等算法，则可得到 $t$个模型，再通过取平均值、投票等方法，即可得到Bagging的结果

### 样本被抽到的概率  

对于样本总量为m，有放回的的随机抽样$t(t =n \cdot s)$次， 任一样本至少抽到一次的概率：
$$
P = 1 - (1 - \frac{1}{m})^{t}
$$

特别的，当$m \rightarrow +\infty$且$t＝m$时，有如下：
$$
\begin{aligned}
P = & 1 - (1 - \frac{1}{m})^m \\
= & 1 - \frac{1}{(1 + \frac{1}{-m})^{-m}} \\
＝ & 1 - \frac{1}{e} \\
\approx & 63.2％ 
\end{aligned}
$$  
上式中$e$为[自然常数](https://zh.wikipedia.org/wiki/E_(数学常数)) 。

### 减小方差的说明
Bagging可以减小预测结果方差

简单说明，考虑以下极端情况

* 假设n个模型是完全独立的，即各模型任意组合的协方差为0
	假设各个模型的方差均为$Var(x)$， 则
	$$
	\begin{aligned}
	& Var\big[\frac{1}{n}(X_1 + X_2 + \ldots + X_n)\big]  \\
	= & \sum_{i=1}^n var(\frac{1}{n}X_1) \\
	= & n \cdot \frac{1}{n^2} Var(x) \\
	= & \frac{1}{n} Var(x) \\
	\end{aligned}
	$$

* 假设各模型完全相同，则：
$$
Var\big[\frac{1}{n}(X_1 + X_2 + \ldots + X_n)\big] = Var(\frac{1}{n} \cdot n X_i) = Var(x)
$$  

现实情况介于两者之间，所以Bagging可以降低方差


## Boosting  
> 通过多个弱分类器集成为强分类器方法的统称 

根据以上定义，Bagging是boosting的一个子集，但是很多资料上Boosting特指，各个模型是训练基于整体模型的误差训练的。如AdaBost是串行训练模型，第n＋1
个模型是针对前n个模型集成后的误差来训练的。下文也是将Boosting方法看成是多个模型基于误差协作组成强模型的的方法

Boosting的主要特点：弱分类器之间有依赖关系

因为Boosting的误差相关联性，所以Boosting是偏向于降低误差


# Bagging算法
随机森林:  
Random Forests，<https://zh.wikipedia.org/wiki/随机森林> ， 是Bagging的一种实现 

> 训练n不剪枝，参与构建树的特征为抽样特征，对多对棵树进行投票或取平均  
> 分类问题：多个ID3、C4.5、C5.0或CART分类树结果投票 
> 回归问题：多个CART回归树结果求平均  

# Boosting算法  
向前分布算法:  
 <http://reset.pub/2017/03/31/forward-stagewise-algorith/>  

AdaBoost:  
<http://reset.pub/2017/03/30/adaboost/>   

提升，梯度提升，梯度提升树:  
<http://reset.pub/2017/04/03/gbdt/>   

xgboost:  
<http://reset.pub/2017/04/01/xgboost/>  

LightGBM:  
微软开源的梯度提升库
Light Gradient Boosting Machine  
<http://www.msra.cn/zh-cn/news/blogs/2017/01/lightgbm-20170105.aspx>

gcForest:  
周志华博士和其学生提出的深度森林模型。
<http://reset.pub/2017/03/31/gcForest/>  

Boosting模型在构建构成中可以对每基模型使用样本、特征抽样。所以以上算法不一定是纯粹的Boosting



# 参考 
[1]《统计学习方法》，李航著，2012 
[2]《机器学习》，周志华著，2016  
[3] https://en.wikipedia.org/wiki/Gradient_boosting  
[4] XGboost的GitHub地址：<https://github.com/dmlc/xgboost>   
\[5\] [Deep Forest: Towards An Alternative to Deep Neural Networks](https://arxiv.org/pdf/1702.08835.pdf)，Zhi-Hua Zhou and Ji Feng，2017.02.28  
[6] XGBoost官网：<https://xgboost.readthedocs.io/en/latest/model.html>  
[7] XGboost的GitHub地址：<https://github.com/dmlc/xgboost>  
[8] <http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf>  
[9] <https://www.zhihu.com/question/26760839>  
[10] <https://zh.wikipedia.org/wiki/E_(数学常数)>  


