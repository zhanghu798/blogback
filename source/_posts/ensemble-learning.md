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

一个是boosting派系，它的特点是各个弱学习器之间有依赖关系。另一种是bagging流派，它的特点是各个弱学习器之间没有依赖关系，可以并行拟合。本文就对集成学习中Bagging与随机森林算法做一个总结

# Bagging  

能够减小训练方差

Bootstrap aggregating

>
给定一个大小为$n$的训练集$D$，Bagging算法从中均匀、有放回地选出$m$个大小为$n'$的子集$D_{i}$，作为新的训练集。在这$m$个训练集上使用分类、回归等算法，则可得到 $m$个模型，再通过取平均值、取多数票等方法，即可得到Bagging的结果  
>
提高其准确率、稳定性的同时，通过降低结果的方差，避免过拟合的发生

# Boosting  
由多个弱分类器组合训练为强分类器的方法

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






