layout: post
title: "GBDT"
comments: true
tags:
	- GBDT
	- 集成方法
date:  2017-04-03 21:20:54
categories:
    - 机器学习
    - 集成方法
    - GBDT
---

 
 {% cq %} <font size=4>Gradient Boosting Decision Tree</font>{% endcq %}
 
 梯度提升树，利用梯度进行多棵树集成的方法
<!-- more -->

# Boosting，提升

$$f_M(x) = \sum_{m=1}^M T(x;\Theta_m) \tag{1}$$
$$
f_m(x) = f_{m-1}(x) + T(x; \Theta_m) \tag{2}
$$

$$
\hat{\Theta}_m = arg \min_{\Theta_m} \sum_{i=1}^N L\big(y_i, f_{m-1}(x_i) + T(x_i; \Theta_m) \big)  \tag{3}
$$

算法步骤

- 初始化$f_0(x) = 0$
- for m in range[1, M+1]: # 建立第m棵树
	- $r_{mi} = y_i  - f_{m-1}(x_i), i = 1,2, \cdots, N$
  	- 拟合残差$r_{mi}$学习一个回归树，得到$T(x； \Theta_m)$
  	- $f_m(x) = f_{m-1} + T(x；\Theta_m)$  
- $f_M(x) = \sum_{m=1}^M T(x;\Theta_m)$




# Gradient boosting，梯度提升

梯度提升树是有多棵树共同决策而成的
$$F_m(x) = \sum_{i=1}^m f_m(x) = F_{m-1}(x) + f_m(x)$$

多棵树是串行训练而成的，初始时，令
$$F_0(x) = \mathop{\arg\min}\limits_{\gamma} \sum_{i=1}^n L(y_i, \gamma)$$
则：
$$
F_m(x) = F_{m-1}(x) + \mathop{\arg\min}\limits_{f} \sum_{i=1}^n L\big(y_i, F_{m-1}(x_i) + f(x_i)\big) 
$$

考虑   

>
$f = \min l(\theta)$，$l(\theta)$是凸的。
利用梯度下降，步长为$\gamma$的梯度下降为：
$$
\theta^{(t)}= \theta^{(t-1)} - \gamma \bigg[\frac{ \partial l(\theta)} {\partial \theta}\bigg]_{\theta = \theta^{(t-1)} }
$$

原问题可以重新表达为在已知$F_{m-1}(x)$的情况下，更新参数$F_{m}(x)$替代$F_{m-1}(x)$，使得$L \big(y_i, F(x)\big)$最小化，则，可以利用梯度下降的方式：
$$
F_m(x) = F_{m-1}(x) - \bigg( \gamma_m \sum_{i=1}^n \frac{\partial L(y_i,Z_i)}{\partial Z_i)} \bigg)_{Z_i = F_{m-1} \ \ \ (x_i)}
$$

$$
\gamma_m = \mathop{\arg\min}\limits_{\gamma} \sum_{i=1}^n \Bigg(
 F_{m-1}(x_i) - \gamma \bigg[ \frac{\partial L(y_i,Z_i)}{\partial Z_i} \bigg]_{Z_i =  F_{m-1} \ \ (x_i) }
 \Bigg)
$$

$\gamma_m$ 可以通过[线性搜索](https://en.wikipedia.org/wiki/Line_search)得到

[算法步骤](https://en.wikipedia.org/wiki/Gradient_boosting)


<img src="/pic/ml/gbdt/gbdt_Gradient-boosting.png" border="0" width="90%" height="90%" style="margin: 0 auto"><center>[图1 梯度提升伪代码](https://en.wikipedia.org/wiki/Gradient_boosting)</center>

# Gradient Boosting Decision Tree， 梯度提升树

	
	


# 模型结构

# 算法流程

# 参考资料  
[1]《统计学习方法》，李航著，2012  
[2] https://en.wikipedia.org/wiki/Gradient_boosting  
[3] XGboost的GitHub地址：<https://github.com/dmlc/xgboost>    

LightGBM github 地址 <https://github.com/Microsoft/LightGBM>

如何看待微软新开源的LightGBM? <https://www.zhihu.com/question/51644470>

Melt/LightGBM中GBDT的实现: <http://files.cnblogs.com/files/rocketfan/LightGBM%E4%B8%AD%E7%9A%84GBDT%E5%AE%9E%E7%8E%B0.pdf>

GBDT的两个版本 <http://suanfazu.com/t/gbdt-mart-gai-nian-jian-jie/133>

GBDT中分类与回归 <https://zhuanlan.zhihu.com/p/25257856?refer=data-miner>

