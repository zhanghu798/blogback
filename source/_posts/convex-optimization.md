layout: post
title: "凸优化"
comments: true
tags:
	- 凸优化
date: 2016-04-23 23:52
categories:
    - 机器学习
---

{% cq %} <font size=4>Convex Optimization，凸优化</font>{% endcq %}

偏支持向量机问题解中用到的凸优化
<!-- more -->

# 凸优化相关知识
## 凸集
---

如果集合$C$中任意两点间的**线段**仍然在$C$中, 即满足:
$$\\left.
\begin{aligned}
  \forall x\_1, x\_2 \in C &  \\\\
\forall\ \theta\ ,\ 0 \leqslant \theta \leqslant 1 &
\end{aligned}
\\right\\} \Longrightarrow \theta x\_1 + (1-\theta)x\_2 \in C
$$

多点推广:
$$\\left.
\begin{aligned}
  \forall x\_1, x\_2, \ldots, x\_k \in C &  \\\\
  \theta\_i \geqslant 0, \ i = 1, \ldots, k & \\\\
  \sum\_{i=1}^k \theta\_i = 1 &
\end{aligned}
\\right\\} \Longrightarrow \theta_i x\_i + \ldots + \theta_k x\_k \in C
$$

多点推广的证明:
1, 三点成立证明
$$\begin{aligned}
 \\left.
\begin{aligned}
    \\left.
        \begin{aligned}
            \\left.
                \begin{aligned}
                    \forall x\_1, x\_2 \in C &  \\\\
                    \forall\ \theta\_1\ ,\ 0 \leqslant \theta\_1 \leqslant 1 &
                \end{aligned}
            \\right\\} \Longrightarrow \theta\_1 x\_1 + (1-\theta\_1)x\_2 \in C \\\\
            \forall\ \theta\_2\ ,\ 0 \leqslant \theta\_2 \leqslant 1 \\\\\
            \forall\ x\_3 \in C \\\\
        \end{aligned}
    \\right\\} \Longrightarrow \theta\_2 (\theta\_1 x\_1 + (1-\theta\_1)x\_2) + (1-\theta\_2)x\_3 \in C \\\\
   \\left.
        \begin{aligned}
            &0 \leqslant \theta\_1, \theta\_2, \theta\_3 \leqslant 1 \\\\
            令:\ &\theta\_1'=\theta\_2 \theta\_1 \\\\
            &\theta\_2'=\theta\_2(1-\theta\_1) \\\\
            &\theta\_3'=(1-\theta\_2)
        \end{aligned}
    \\right\\} \Longrightarrow \\left\\{
    \begin{aligned}
        \theta\_1' + \theta\_2' + \theta\_3' = 1 \\\\
        0 \leqslant \theta\_1',\theta\_2', \theta\_3' \leqslant 1 &
    \end{aligned}
    \\right.
\end{aligned}
\\right\\} &\\\\
\\\\
\Longrightarrow \theta\_1' x\_1 + \theta\_2' x\_2 + \theta\_3' x\_3 \in C  &
\end{aligned}
$$

整理上式得:
$$\\left.
\begin{aligned}
  \forall \  x\_1, x\_2, x\_3 \in C &  \\\\
  \forall \ \theta\_i, 0 \leqslant \theta\_i \leqslant 1 & \ \ i \in [1, 2, 3]\\\\
  \theta\_1' + \theta\_2' + \theta\_3' = 1&
\end{aligned}
\\right\\} \Longrightarrow \theta_1' x\_1 + \theta_2' x\_2  + \theta_3' x\_3  \in C
$$

2, k点成立正证明:
用同样的方法可以证明k-1个点到k个点的情况, 即通过归纳法证得多点也成立


## 仿射集
如果集合$C$中任意两点决定的**直线**仍然在$C$中, 即满足:
$$\\left.
\begin{aligned}
  \forall x\_1, x\_2 \in C &  \\\\
\forall\ \theta\ &
\end{aligned}
\\right\\} \Longrightarrow \theta x\_i + (1-\theta)x\_2 \in C
$$
类似凸集可推广到多点

## 凸函数
如果函数$\ f\ $定义域是凸集,对于定义域内的任意$\ x, y \in \boldsymbol{dom}\ f\ $ 和任意$\ 0 \leqslant \theta \leqslant 1\ $, 有 $f\Bigg\(\theta x + (1-\theta)y\Bigg) \leqslant \theta f(x) + (1-\theta)f(y)$, 即:

$$\\left.
\begin{aligned}
\ 函数f\ 的定义域\boldsymbol{dom}\ f\ 是凸集 &  \\\\
\ \forall x, y \in \boldsymbol{dom}\ f\  &  \\\\
\forall \theta\ ,\ 0 \leqslant \theta \leqslant 1 &
\end{aligned}
\\right\\} \Longrightarrow f(\theta x + (1-\theta)y) \leqslant \theta f(x) + (1-\theta)f(y)
$$

* 即,定义域内,两点的期望的函数值小于分别求函数值的期望, 亦即$\ f(E(x)) \leqslant E(f(x))$
* 当以上不等式存在'$=$'成立时,是**非严格凸**, 否则为**严格凸**


### 凸函数的判断
#### 根据定义判断
直接利用定义证明

#### 一阶条件判断
假设函数$\ f\ $可微, 则:
$$f是凸函数 \Longleftrightarrow
\\left\\{
\begin{aligned}
& \boldsymbol{dom} \ f\  是凸集 \\\\
& \forall \ x, y \in \boldsymbol{dom}\ f \\\\
& f(y) \geqslant f(x) + \nabla f(x)^\mathsf{T}(y-x)
\end{aligned}
\\right.
$$

#### 二阶条件判断
如果函数$\ f \ $二阶可微, 则$\ f \ $为凸函数的**充要条件**是, 二阶导数$Hessian$矩阵是正定阵

## 仿射函数
如果具有 $f(x)\ =\ \boldsymbol{A}\boldsymbol{x}+\boldsymbol{b}$的形式的函数$\ f\ $称为仿射函数

## 凸优化

### 优化问题的一般形式
$$\begin{aligned}
obj\ :&\ \ \min\ f\_0(x)  \\\\
st\ :&\ \ f\_i(x) \leqslant 0, \ i = 1,2, \ldots, m \  \\\\
&\ \ h_i(x) = 0, i = 1, 2, \ldots, p
\end{aligned}
$$

### 凸优化问题的一般形式
$$\begin{aligned}
obj\ :&\ \ \min\ f\_0(x) & \\\\
st\ :&\ \ f\_i(x) \leqslant 0\ , \ i = 1,2, \ldots, m &\\\\
&\ \ h_i(x) = 0, i = 1, 2, \ldots, p & \\\\
&\ \ f\_0(x)\ 是凸函数 \\\\
&\ \ f\_i(x)\ 是凸函数 \ , i \in [1, m] & \\\\
&\ \ h\_i(x)\ 是仿射函数\, 即h\_i(x)为\boldsymbol{A}\boldsymbol{x}+\boldsymbol{b} 的形式\ , i \in [1, p] &
\end{aligned}
$$


## 广义拉格朗日函数:

$$\begin{aligned}
& L(x, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f\_0(x) + \sum\_{i=1}^m\lambda\_i f\_i(x) + \sum\_{i=1}^p \nu\_i h\_i(x) \\\\
& \boldsymbol{\nu} \in \boldsymbol{R}^p , \  \boldsymbol{\lambda} \in \boldsymbol{R}^m \\\\
& \lambda\_i \geqslant 0, \ i \in [1, m]
\end{aligned}$$


## 拉格朗日函数:
$$\theta\_p(x) = \max\_{\boldsymbol{\lambda}, \boldsymbol{\nu}}{L(x, \boldsymbol{\lambda}, \boldsymbol{\nu})}
\\left\\{
\begin{aligned}
 f\_0(x), & \ \  x满足原始问题的约束  \\\\
 +\infty, & \ \ 其他\\\\
\end{aligned}
\\right.
$$

## 拉格朗日函数的理解
<img src="/pic/ml/down/LagrangeMultipliers2D.svg.png" width="50%" height="50%" style="margin: 0 auto"> <center>图1([来自 wikipedia](https://en.wikipedia.org/wiki/Lagrange_multiplier)), $\ f\_0(x)$等高线示与约束条件示意图,蓝色箭头方向为等高线对应值降低的方向,<br>当$g(x, y) - c = 0$时红色线表示等式约束;<br>当$g(x, y) - c \leqslant 0$ 红色线箭头方向为不等式约束成立的方向 </center >


- 等高线:
考虑三维情况, $z=f\_0(x, y)$的情况, 等高面$z = d$与函数$f\_0(x, y)$相交的部分为等高线, 等高线在$(x, y)平面的投影可表示为f\_0(x, y)=d$


### 一个等式的约束问题

如图假设等式约束为 $h\_1(x, y)=g(x, y) - c=0$
拉格朗日函数表示为: $L=f\_(x, y) + \nu h\_1(x, y)$
如果函数$f\_0(x, y)$ 的值域连续
则其等高线投影与g(x, y) -c 相切处取得等式约束条件下的极值, 此时

$$\\left.
\begin{aligned}
    & \\left.
    \begin{aligned}
         f\_0(x, y)与h\_1(x, y)相切\Rightarrow  \nabla_{x, y} f\_0(x,  y) = \nu' (\nabla_{x, y} h\_1(x, y)) \Rightarrow \nabla_{x, y}[f\_0 + \nu(h\_1)] =0 & \\\\
          h\_1(x,y)=0 \Rightarrow \nabla_{\nu}[f\_0 + \nu(h\_1)]  = 0 &
    \end{aligned}
    \\right\\} \\\\
    \\\\
    & \Longrightarrow  \nabla_{x, y, \nu} \Bigg\[f\_0(x) + \nu h\_1(x) \Bigg\]  = 0
    \\\\
    & \Longrightarrow 目标函数f(x,y)在一个等式约束条件下取得最值处的解与L=f\_(x, y) + \nu h\_1(x,y)极值的解等价
\end{aligned}
\right.
$$
- *注*:
$\nu'$ 可以看作是两个等价切线(超平面)的系数, 同分割超平面的等价超平面的系数$k$
$\nu = -\nu'$

### 多个等式约束问题
<img src="/pic/ml/down/600px-As_wiki_lgm_parab.svg.png" width="45%" height="45%" style="margin: 0 auto"><center>图2([来自 wikipedia](https://en.wikipedia.org/wiki/Lagrange_multiplier)), $\ f\_0(x)$等高线, 及含有两个约束条件的情况示意图</center>

极值多个等式约束看作在满足约束条件下集合处, 定义极值处的梯度方向为多个约束加权后得到的梯度和目标函数$f\_0(极点)$处的梯度平行
即极值处满足:
$$\\left.
\begin{aligned}
    & \nabla\_{\boldsymbol{x}} f(x) = \sum\_{i=1}^p \lambda\_i' \nabla_{\boldsymbol{\boldsymbol{x}}}  f\_i(\boldsymbol{x})  \\\\
    & f\_i(x) = 0, \ \ i \in [1, m]
\end{aligned}
\\right\\} \Longrightarrow \nabla\_{\boldsymbol{x}, \boldsymbol{\lambda}} \Bigg\[f\_0(x) + \sum\_{i=1}^p \lambda\_i f\_i(x)\Bigg\] = 0, \ \ \ \ 当\lambda\_i \neq 0时, \ \ i \in [1, m]
$$

### 一个不等式等式约束
假设目标函数$\min f\_0(x)$, 有不等式约束$f\_1(x) < 0$
如图1表示, 当不等式约束有效时,须使得目标函数的函数值减小的方向与不等式约束成立的方向相反**且**目标函数与不等式约束的边界相切时取得极值. 即如果约束有效的情况下, 目标函数取得极值时一定在约束的边界处, 问题可以简化为等式约束的情况

边界有效时: 目标函数的函数值减小的方向与不等式约束成立的方向相反
则:
$$\\left.
\begin{aligned}
    f\_0(x)降低方向的梯度为: -\nabla f_0(x)  & \\\\
    f\_1(x) < 0所表示定义域方向梯度方向为: -\nabla f_1(x) & \\\\
    f\_0(x)与f\_1(x)相切,且降低方向的梯度和定义域方向梯度相反 &
\end{aligned}
\\right\\} \Longrightarrow
\\left\\{
\begin{aligned}
    & \nabla f_0(x) = -\lambda \nabla f_1(x) \\\\
    & \lambda > 0
\end{aligned}
\\right.
$$

考虑不等式约束无效的情况, 目标函数降低方向的梯度和定义域方向梯度相同, 即假设不等式约束成立的方向为图1中红色箭头相反的方向, 此时仍然可以有目标函数与不等式约束边界相切, 但是所求$\lambda < 0$, 且此时切点对应的函数值显然不是极值处

另外,只考虑不等式约束时, 不等式自身约束恒成立力时(或着当有多个约束时, 其他约束的定义域为该约束的子集. 即该约束对所有约束的交集无贡献), 该约束项的乘子可为0.

综上: 考虑约束的有效性综, 不等式约束的朗格朗日乘$\lambda \leqslant 0$, 等号拉格朗日乘子等于0时,代表约束对取得最值时无贡献

### 多个不等式约束
多个不等式约束同多个不等式约束的情况. 其中每个不等式约束的朗格朗日乘子均大于等于0

### 多个等式约束和多个不等式约束的情况
同多个等式约束情况, 取得极值处可看作是在可行域内有效约束内**且**由权值为拉格朗日乘子加权作为梯度平行于原函数的梯度


### 原问题的拉格朗日函数
$$L\_P= \min\_{\boldsymbol{x}}\max\_{\boldsymbol{\lambda}, \boldsymbol{\nu}} {L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})}$$

### 原问题的拉格朗日对偶函数
对偶函数
$$令, \ g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf\_{x \in D}L(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf\_{x \in D\}\(f\_0(x) + \sum\_{i=1}^m\lambda\_i f\_i(x) + \sum\_{i=1}^p \nu\_i  h\_i(x)\)$$

$$L\_D = \max\_{\boldsymbol{\lambda}, \boldsymbol{\nu}}\min\_{x} {L(x, \lambda, \nu)} = \max\_{\boldsymbol{\lambda}, \boldsymbol{\nu}}g(\boldsymbol{\lambda}, \boldsymbol{\nu}) $$

$g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ 一簇是关于($\boldsymbol{\lambda}, \boldsymbol{\nu})$仿射函数逐点求下界, 又因为仿射函数的下界求交集是凹的. 所以$g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ 是凹的
对于原问题的最优值$p^\*$, 任意 $\lambda \succeq 0$, 有$g(\boldsymbol{\lambda}) \geqslant p^\*$,即$g(\boldsymbol{\lambda})$ 为原始问题的下界


### 原问题拉格朗日函数与对偶函数的关系
假设原问题和其对偶问题均有最优值

$$\begin{aligned}
& \min\_{x} {L(x, \lambda, \nu)} \leqslant L(x, \lambda, \nu) \leqslant \max\_{\lambda, \nu} {L(x, \lambda, \nu)} \\\\
\Longrightarrow \ & \max\_{\lambda, \nu}\min\_{x} {L(x, \lambda, \nu)} \leqslant L(x, \lambda, \nu) \leqslant \min\_{x}\max\_{\lambda, \nu} {L(x, \lambda, \nu)} \\\\
\Longrightarrow \ & L\_D \leqslant L\_P
\end{aligned}
$$

当上式子取得等号时, 称为强对偶.

### Slater 准则
用于凸优化问题中, 强对偶条件成立是否存在

证明见:
《凸优化》－ 清华出版社 Stephen Boyd ...著， 王书宁...译, $P\_{226} - P\_{228}$


### KKT条件
原拉格朗日函数和拉格朗日对偶函数最优值可以达到并且相等是的**条件**

以上问题可看作是凸二次优化问题
引入拉格朗日函数:


$$L(\boldsymbol{w}, b, \boldsymbol{\alpha})=\frac{1}{2}\lVert\boldsymbol{w}\rVert^2 \- \sum\_{i=1}^m\alpha\_iy\_i(\boldsymbol{w}^\mathsf{T}\boldsymbol{x}+b)  \+ \sum\_{i=1}^m\alpha\_i$$