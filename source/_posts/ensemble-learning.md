layout: post
title: "集成方法"
comments: true
tags:
	- 集成学习
date:  2017-03-29 16:55:28
categories:
    - 机器学习
---

这个题目取得有点大，就当先挖个坑  

# 两种基本集成学习思路

## Bagging  
Bootstrap aggregating  

>
给定一个大小为$m$的样本集$D$，Bagging算法从中均匀、有放回地选出$n$个大小为$s$的子集$D_{i}$，作为新的训练集。在这$t$个训练集上使用分类、回归等算法，则可得到 $t$个模型，再通过取平均值、投票等方法，即可得到Bagging的结果  

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

能够减小训练方差  
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

# 随机森林  
Random Forests，<https://zh.wikipedia.org/wiki/随机森林> ， 是Bagging的一种实现

>训练n不剪枝，没棵树是有部分样本中的部分特征组成，对结果进行投票或取平均  
>分类问题：多个ID3、C4.5、C5.0或CART分类树结果投票 
>回归问题：多个CART回归树结果求平均  

# 向前分布算法  
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

+ 算法流程:
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


# AdaBoost  
Adaptive Boosting，<https://zh.wikipedia.org/wiki/AdaBoost>  
AdaBoost是多个分类器组合算法，[维基百科AdaBoost算法过程](https://zh.wikipedia.org/wiki/AdaBoost)：  
<img src="/pic/ml/down/AdaBoost_process.png" width="100%" height="100%" style="margin: 0 auto">

- 最终分类器为：
$$
G(x) = sign\big(f(x)\big) = sign\big(\sum_{i=1}^{k_{max}} \alpha_i C_i(x)\big)
$$  

* 参数说明：
	- $k$：循环次数
	- $W_k(i)$：训练地k个分类器时，第i个样本的权重
	- $E_k$：在训练集上的误差率。实际上是预测错误的样本按样本编号去重后，样本权重$W_k(i)$求和
	- $h_k(x^i)$：第K个分类器$C_k$给出的对任一样本点xi类别的预测
	- $Z_k$：归一化因子。$Z_k = \sum_{j=1}^{k} W_j(i)exp\big(-\alpha_j h_j(x^i)y_i\big)$。j：分类器标号

* 算法说明：
	- 如果一个算法对于某个样本预测错误的，其样本权重变高，下一步重点训练该样本。反之，预测正确的样本权重变高
	- 单一模型的分类训练误差高，该模型的权重低。反之，权重变高


# GBDT  
Gradient Boosting Decision Tree  

# XGBoost  
eXtreme Gradient Boosting  
XGBoost的github地址：<https://github.com/dmlc/xgboost>  
<https://xgboost.readthedocs.io/en/latest/>
  
XGBoost的介绍 <https://xgboost.readthedocs.io/en/latest/model.html>

$$
\begin{split}\hat{y}_i^{(0)} &= 0\\
\hat{y}_i^{(1)} &= f_1(x_i) = \hat{y}_i^{(0)} + f_1(x_i)\\
\hat{y}_i^{(2)} &= f_1(x_i) + f_2(x_i)= \hat{y}_i^{(1)} + f_2(x_i)\\
&\dots\\
\hat{y}_i^{(t)} &= \sum_{k=1}^t f_k(x_i)= \hat{y}_i^{(t-1)} + f_t(x_i)
\end{split}
$$

# LightGBM  
Light Gradient Boosting Machine  
<http://www.msra.cn/zh-cn/news/blogs/2017/01/lightgbm-20170105.aspx>

知乎地址：<https://github.com/Microsoft/LightGBM>

# gcFrest  
multi-Grained Cascade forest  

<https://arxiv.org/pdf/1702.08835.pdf>
<http://it.sohu.com/20170302/n482153688.shtml>



# 参考

<http://wenda.chinahadoop.cn/question/4155>  
<https://www.zhihu.com/question/26760839>  
<chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=http%3A%2F%2Fnew.sis-statistica.org%2Fwp-content%2Fuploads%2F2013%2F10%2FCO09-Variable-Selection-using-Random-Forests.pdf>

<chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fhal.inria.fr%2Ffile%2Findex%2Fdocid%2F755489%2Ffilename%2FPRLv4.pdf>


