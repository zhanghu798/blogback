layout: post
title: "GBDT"
comments: true
tags:
	- GBDT
date:  2017-03-26 15:16:38
categories:
    - 机器学习
---


{% cq %} <font size=4>GBDT，Gradient Boosting Decision Tree，梯度提升树</font>{% endcq %}

多个决策树加法模型

提升树：训练第n+1棵树基于前n可棵树组成模型的偏差来训练

梯度提升树：
第n+1棵树是基于前n棵树残差的剃度来训练，可以理解为梯度下降，此时的梯度是第n+1棵树

考虑梯度下降训练参数：
$$\theta_{t+1} = \theta_{t} + r\nabla_{\theta}$$




LightGBM github 地址 https://github.com/Microsoft/LightGBM

如何看待微软新开源的LightGBM? https://www.zhihu.com/question/51644470

Melt/LightGBM中GBDT的实现: http://files.cnblogs.com/files/rocketfan/LightGBM%E4%B8%AD%E7%9A%84GBDT%E5%AE%9E%E7%8E%B0.pdf

GBDT的两个版本 http://suanfazu.com/t/gbdt-mart-gai-nian-jian-jie/133

GBDT中分类与回归 https://zhuanlan.zhihu.com/p/25257856?refer=data-miner

