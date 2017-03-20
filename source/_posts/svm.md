layout: post
title: "SVM"
comments: true
tags:
	- SVM
date: 2017-11-03 23:52
categories:
    - 机器学习
---

{% cq %} <font size=4>SVM，Support Vector Mechines，支持向量机</font>{% endcq %}



# 线性可分SVM

<img src="/pic/ml/down/440px-Svm_max_sep_hyperplane_with_margin.png" border="0" width="40%" height="40%" style="margin: 0 auto">

## 问题引出


有样本集 $D=\\{(\boldsymbol{x_1}，y_1)，(\boldsymbol{x_2}，y_2)，\ldots，(\boldsymbol{x_m}，y_m)\\}$。其中， $x_i$是一个样本，列向量；标签$y_i\in\\{-1， 1\\}$且$D$是线性可分的。 找到一个最优超平面$ \boldsymbol{w} ^ \mathsf{T}\boldsymbol{x} + b = 0$($\boldsymbol{w}$为是超平面系数，列向量；$b$为截距；$\boldsymbol{x}$为列向量)，使得在保证正确分类的情况下，样本点到超平面的*最小*距离*最大*化

---
* *最小*距离：所以样本点到超平面的距离
* 支持向量：每个类别中到超平面距离最小的点
* *最大*化：使得两类支持向量到超平面最小距离最大化，即两类支持向量点到超平面的距离相等
---

<!-- more -->


## 问题的数学表达


$$假设条件(正确分类条件)\\left\\{
\begin{aligned}
\frac{\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_i} + b}{\lVert\boldsymbol{w}\rVert} > 0， & \ y_i = +1  \\\\
\frac{\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_i} + b}{\lVert\boldsymbol{w}\rVert} < 0， & \ y_i = -1  \\\\
\end{aligned}
\\right.
$$

$$目标函数与约束条件(最大间隔条件)\\left\\{
\begin{aligned}
obj: &  \max_{\boldsymbol{w}，b} \frac{\left|\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_0} + b\right|}{\lVert\boldsymbol{w}\rVert} & \\\\
st: & \frac{\left|\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_i} + b\right|}{\lVert\boldsymbol{w}\rVert} \geqslant  \frac{\left|\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_0}+ b\right|}{\lVert \boldsymbol{w}\rVert} ，\ i\in[1， m]& \\\\
& \boldsymbol{x_0}代表所有支撑向量 &
\end{aligned}
\right.
$$

---

点到直线的距离:
二维空间中，点$(x_0， y_0)$到直线$Ax+By+C=0$的距离$d=\frac{\left|Ax_0+By_0+C\right|}{\sqrt{A^{2}+B^{2}}}$.
多纬空间中，点$\boldsymbol{x\_0}$到$\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x} + b = 0$的距离$d=\frac{\left|\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x\_0} + b\right|}{\lVert\boldsymbol{w}\rVert} $

---

1，$\ $由于$\lVert\boldsymbol{w}\rVert > 0$， 则假设条件可以写成 $y_i(\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_i} + b) > 0$

2，$\ $由于超平面$ \boldsymbol{w} ^ \mathsf{T}\boldsymbol{x} + b = 0$ 有无穷多个等价超平面 $ \kappa(\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x} + b) = 0$， 所以存在等价超平面 $\boldsymbol{w'} ^ \mathsf{T}\boldsymbol{x} + b' = 0$使得不在该超平面上的点$x_0$满足$\left|\boldsymbol{w'} ^ \mathsf{T}\boldsymbol{x_0} + b'\right| = 1$

3，$\ $固定点到所有等价超平面的距离相等

基于以上，合并正确分类条件和最大间隔条件
$$\\left\\{
\begin{aligned}
obj: &  \max_{\boldsymbol{w'}，b'} \frac{1}{\lVert\boldsymbol{w'}\rVert} & \\\\
st: & y_i(\boldsymbol{w'} ^ \mathsf{T}\boldsymbol{x_i} + b')\geqslant
 1，\ i\in[1， m]& \\\\
& 满足y_0(\boldsymbol{w'} ^ \mathsf{T}\boldsymbol{x_0} + b') = 1 的点为支撑向量&
\end{aligned}
\right.
$$

对上式变量替换及极值等价为以下:
$$\\left\\{
\begin{aligned}
obj: &  \min_{\boldsymbol{w}，b} \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 & \\\\
st: & y_i(\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_i} + b)\geqslant
 1，\ i\in[1， m]& \\\\
& 满足y_0(\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_0} + b) = 1 的点为支撑向量&
\end{aligned}
\right.
$$

*注*:$\ $此时所求$\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x} + b = 0$ 为原始假设超平面的等价超平面

问题目标：
找到满足上是约束的分离超平面参数$\boldsymbol{w^\*}，b^\*$，求的分离超平面为$\boldsymbol{w^\*} \cdot \boldsymbol{x} + b^\* = 0$, 对于待预测样本$\boldsymbol{x\_i}$分类决策函数为：$$f(x) = sign(\boldsymbol{w^\*} \cdot \boldsymbol{x\_i} + b)$$


## 求解
利用[凸优化简介](http://reset.pub/2017/03/18/convex-optimization)，原问题可以写作：
$$\\left\\{
\begin{aligned}
obj: &  \min_{\boldsymbol{w}，b} \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 & \\\\
st: & 1 - y_i(\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_i} + b)\leqslant
 1，\ i\in[1， m]&
\end{aligned}
\\right.
$$由于目标函数是二次函数是凸的，约束函数是关于$w\_i$ 和 $b$的仿射函数，所以此问题的为凸优化问题

引入拉格朗日乘子$\alpha\_i, i \in [1, ldots, m]$， 其拉格朗日函数为
$$L(\boldsymbol{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 + \sum\_{i=1}^m \alpha\_i \big\[1 - y_i(\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_i} + b) \big\]$$

### 问题的拉格朗日函数
$$\min\_{\boldsymbol{w}, b} \max\_\boldsymbol{\alpha} L(\boldsymbol{w}, b, \boldsymbol{\alpha})
$$

### 对偶函数为
$$\max\_\boldsymbol{\alpha} \min\_{\boldsymbol{w}, b}  L(\boldsymbol{w}, b, \boldsymbol{\alpha})
$$

### KKT条件
对照[凸优化简介](http://reset.pub/2017/03/18/convex-optimization/)，为有若干个不等式约束的凸优化情况
$$\\left\\{
\begin{aligned}
    & \nabla\_{\boldsymbol{x}} L = 0 & \Longrightarrow & \ \  \\left\\{
    \begin{aligned}
        & \nabla\_{\boldsymbol{w}} L(\boldsymbol{w}, b, \boldsymbol{\alpha})\_{\boldsymbol{w}=\boldsymbol{w}^\*, b=b^\*} = \boldsymbol{0}  & \ \ \ \ (1)\\\\
        & \nabla\_{b} L(\boldsymbol{w}, b, \boldsymbol{\alpha})\_{\boldsymbol{w}=\boldsymbol{w}^\*, b=b^\*} = 0 & \ \ \ \ (2)
    \end{aligned}
    \\right. \\\\
    & \lambda\_i^\*f\_i(x^\*) = 0 & \Longrightarrow & \ \  \alpha\_i^\* \big\( y\_i(\boldsymbol{w}^\* \cdot \boldsymbol{x\_i} + b^\*) -1 \big\) = 0 ， i = 1, \ldots, m & \ \ \ \ (3)\\\\
    & f\_i(x) \leqslant 0 & \Longrightarrow & \ \ 1 - y\_i(\boldsymbol{w}^\* \cdot \boldsymbol{x\_i} + b^\*) \leqslant 0， i = 1, \ldots, m & \ \ \ \ (4) \\\\
    & \lambda\_i \geqslant 0 & \Longrightarrow & \ \ \alpha\_i \geqslant 0，  i = 1, \ldots, m & \ \ \ \ (5)
\end{aligned}
\\right.
$$

说明：(1)，(2)式中，参数$\boldsymbol{w}, b$对应[凸优化简介](http://reset.pub/2017/03/18/convex-optimization/)中KKT条件的$\boldsymbol{x}$， 是待求参数。$y_i(\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_i} + b)\geqslant
 1，\ i\in[1， m]$ 约束中的$\boldsymbol{x_i}$和$y\_i$为一个样本及其对应的类别，为已知常数

### $\min\_{\boldsymbol{w}, b}  L(\boldsymbol{w}, b, \boldsymbol{\alpha})$的求解

$$\boldsymbol{w^\*} - \sum\_{i=1}^m \alpha\_i y\_i \boldsymbol{x\_i} = \boldsymbol{0} \Longrightarrow \boldsymbol{w^\*} = \sum\_{i=1}^m \alpha\_i y\_i \boldsymbol{x\_i} \ \ \ \ (6)$$

由KKT条件(2)可得：
$$ \sum\_{i=1}^m \alpha\_i y\_i  = 0 \ \ \ \ (7)$$

由KKT条件(3)，(4)，(5)联立可得：
- 当$a\_i^\* > 0$时，其对应不等式约束为为边界条件，对应样本点为支持向量，有：
$$\begin{aligned}
& y\_j(\boldsymbol{w}^\* \cdot \boldsymbol{x\_j} + b^\*) -1 = 0， i = 1, \ldots, n。\ \ ( 其中n为支撑向量个数) \\\\
& \because (6)式，且 1 = y\_j \cdot y\_j \\\\
& \therefore b^\* = y\_j - \sum\_{i=i}^n \alpha\_i ^ \* y\_i(\boldsymbol{x_i} \cdot \boldsymbol{x_j})
\end{aligned}
$$

- 当$a\_i^\* = 0$时，其对应的不等式约束$y\_i(\boldsymbol{w}^\* \cdot \boldsymbol{x\_i} + b^\*) -1 < 0， i = 1, \ldots, m$为非边界条件，该样本不在支撑向量上

综合考虑$a\_i^\* \geqslant 0$情况：
$$ b^\* = y\_j - \sum\_{i=i}^m \alpha\_i ^ \* y\_i(\boldsymbol{x_i} \cdot \boldsymbol{x_j}) ， \ j = 1, \ldots, m \ \ \ \ (8)$$

$$
\begin{aligned}
    & \min\_{\boldsymbol{w}, b} L(\boldsymbol{w}, b, \boldsymbol{\alpha}) \\\\
    = & \frac{1}{2}\lVert\boldsymbol{w^\*}\rVert^2 + \sum\_{i=1}^m \alpha\_i \big\(1 - y\_i(\boldsymbol{w^\*} \cdot \boldsymbol{x\_i} + b^\*)\big\) \\\\
    = & \frac{1}{2}\lVert\boldsymbol{w^\*} \rVert^2 + \sum\_{i=1}^m \alpha\_i -  \boldsymbol{w^\*} \cdot \sum\_{i=1}^m     \alpha\_i y\_i \boldsymbol{x\_i} - \sum\_{i=1}^m \alpha\_i y\_i b^\* \\\\
  　\ = & \frac{1}{2}\lVert\boldsymbol{w^\*} \rVert^2 + \sum\_{i=1}^m \alpha\_i -  \boldsymbol{w^\*} \cdot \boldsymbol{w}  - \sum\_{i=1}^m \alpha\_i y\_i b^\*  \ \ \ \because 式(6)： \boldsymbol{w^\*} = \sum\_{i=1}^m \alpha\_i y\_i \boldsymbol{x\_i}：\\\\
    = & -\frac{1}{2}\lVert\boldsymbol{w^\*} \rVert^2 + \sum\_{i=1}^m \alpha\_i -  b^\*  \sum\_{i=1}^m \alpha\_i y\_i \\\\
    = & - \frac{1}{2} \sum\_{i=1}^m\sum\_{j=1}^m \alpha\_i \alpha\_j y\_i y\_j(\boldsymbol{x\_i} \cdot \boldsymbol{x\_j}) + \sum\_{i=1}^m \alpha\_i \ \ \ \ \because 式(6)， 式(7)： \sum\_{i=1}^m \alpha\_i y\_i  = 0
\end{aligned}
$$


### $\max\_\boldsymbol{\alpha} \min\_{\boldsymbol{w}, b}  L(\boldsymbol{w}, b, \boldsymbol{\alpha}) $ 函数的解


考虑KKT条件中上式仍有约束的条件，有如下：
$$
\begin{aligned}
 & \max\_\boldsymbol{\alpha} \min\_{\boldsymbol{w}, b}  L(\boldsymbol{w}, b, \boldsymbol{\alpha}) \\\\
= & \\left\\{
\begin{aligned}
    obj\ :\ &\max\_{\boldsymbol{\alpha}} -\frac{1}{2} \sum\_{i=1}^m\sum\_{j=1}^m \alpha\_i \alpha\_j y\_i y\_j(\boldsymbol{x\_i} \cdot \boldsymbol{x\_j}) + \sum\_{i=1}^m \alpha\_i, \ \ \ \  i，j  \in [1,\ldots, m]\\\\
    st\ :\ &\ \sum\_{i=1}^m \alpha^\* y\_i = 0，i  \in [1,\ldots, m] \\\\
    &\ \alpha\_i \geqslant 0， i  \in [1,\ldots, m]
\end{aligned}
\right.
\end{aligned}
$$

求得$a\_i^\*$后，分离超平面为：
$$\sum\_{i=1}^m a\_i^\* y\_i(\boldsymbol{x} \cdot \boldsymbol{x\_i}) + b^\* = 0， (其中(\boldsymbol{x\_i}, y\_i）为训练样本及对应标签)$$

分类决策函数为：
$$f(x) = sign \big\[\sum\_{i=1}^m a\_i^\* y\_i(\boldsymbol{x\_i} \cdot \boldsymbol{x}) + b^\* \big\] \ \ \ (\boldsymbol{x} 为待预测样本)$$

具体解法见SMO

# 线性不可分SVM

## 问题表示及拉格朗日函数
对每个样本点引入一个松弛变量$\xi\_i \geqslant 0$，使得线性不可分的点对应的支持超平面向着平行于其法线且朝着分离超平面的方向移动$\frac{\xi\_i}{\lVert\boldsymbol{w}\rVert^2} $作为伪支撑超平面上为参照（该点在此伪支撑超平面上）， 使得函数距离约束变松。同时对松弛因子加入惩罚系数（超参数C， C > 0）。以上被称为软件个最大化，数学描述为：

$$\\left\\{
\begin{aligned}
obj: \ &  \min\_{\boldsymbol{w}，b} \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 + C \sum\_{i=1}^m \xi\_i， \ C \geqslant 0 , & i\in[1， m] & \\\\
st: \ & y_i(\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_i} + b)\geqslant 1 - \xi\_i, & i\in[1， m]& \\\\
 & \xi\_i \geqslant 0 , &\ i\in[1， m]& \\\\
& 满足y_0(\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x_0} + b) = 1 的点为支撑向量 &
\end{aligned}
\right.
$$

为松弛变量$\boldsymbol{\xi}$引入拉格朗日乘子$\boldsymbol{\mu}$，其拉格朗日函数为：
$$
\begin{aligned}
& L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) \\\\
= &  \frac{1}{2}\lVert\boldsymbol{w}\rVert^2 + C \sum\_{i=1}^m \xi\_i + \sum\_{i=1}^m \alpha\_i \big\[ 1 - \xi\_i - y\_i(\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x\_i} + b) \big\] - \sum\_{i=1}^m \mu\_i \xi\_i
\end{aligned}
$$

## KKT条件

$$\\left\\{
\begin{aligned}
    & \nabla\_{\boldsymbol{w}} L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu})\ = \boldsymbol{0} \Longrightarrow w^\* - \sum\_{i=1}^m \alpha\_i^\* y\_i x\_i =0 & \ \ \ \ (1)\\\\
    & \nabla\_{b} L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) = 0   \Longrightarrow -\sum\_{i=1}^m a\_i^\* y\_i = 0 & \ \ \ \ (2) \\\\
    & \nabla\_{\xi\_i} L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) = 0 \Longrightarrow C - a^\*\_i - u^\*\_i = 0 \ & i\in[1， m]   \ \ \ \ (3) &\\\\
    & y_i(\boldsymbol{w^\*} \cdot \boldsymbol{x_i} + b^\*)\geqslant 1 - \xi\_i^\*, & i\in[1， m]  \ \ \ \ (4)& \\\\
    & \xi\_i^\* \geqslant 0 ， &\ i\in[1， m]  \ \ \ \ (5)& \\\\
    & \alpha_i^\* \geqslant 0 ， &\ i\in[1， m]  \ \ \ \ (6) &  \\\\
    & \mu_i^\* \geqslant 0 ， &\ i\in[1， m]  \ \ \ \ (7) &  \\\\
    & \alpha\_i^\* \big\[ 1 - \xi\_i^\* - y\_i(\boldsymbol{w^\*} \cdot \boldsymbol{x\_i} + b^\*) \big\] = 0 ， &\ i\in[1， m]  \ \ \ \ (8)&  \\\\
    & \mu\_i^\* \xi\_i^\* = 0 ， &\ i\in[1， m]  \ \ \ \ (9) &  \\\\
\end{aligned}
\\right.
$$

## 求解
### $\min\_{\boldsymbol{w}, b, \boldsymbol{\xi}}   L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu})$的求解

将(1),(2),(3)带入拉格朗日函数得，
$$
\begin{aligned}
& \min\_{\boldsymbol{w}, b} L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) \\\\
 = & - \frac{1}{2} \sum\_{i=1}^m\sum\_{j=1}^m \alpha\_i \alpha\_j y\_i y\_j(\boldsymbol{x\_i} \cdot \boldsymbol{x\_j}) + \sum\_{i=1}^m \alpha\_i
\end{aligned}
$$

### $\max\_{\boldsymbol{\alpha}, \boldsymbol{\mu}} \min\_{\boldsymbol{w}, b, \boldsymbol{\xi}}   L(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu})$
考虑对偶问题最大化时求$\boldsymbol{\alpha}$
考虑KKT条件对上式有约束的条件：
由(3),(5),(7)消去$\mu\_i$得：
$$0 \leqslant \alpha\_i \leqslant C$$

则， 对偶函数的可以表示为
$$
\begin{aligned}
 & \max\_\boldsymbol{\alpha} \min\_{\boldsymbol{w}, b}  L(\boldsymbol{w}, b, \boldsymbol{\alpha}) \\\\
= & \\left\\{
\begin{aligned}
    obj\ :\ &\max\_{\boldsymbol{\alpha}} -\frac{1}{2} \sum\_{i=1}^m\sum\_{j=1}^m \alpha\_i \alpha\_j y\_i y\_j(\boldsymbol{x\_i} \cdot \boldsymbol{x\_j}) + \sum\_{i=1}^m \alpha\_i, \ \ \ \  i，j  \in [1,\ldots, m]\\\\
    st\ :\ &\ \sum\_{i=1}^m \alpha^\* y\_i = 0，i  \in [1,\ldots, m] \\\\
    &\ 0 \leqslant \alpha\_i \leqslant C， i  \in [1,\ldots, m]
\end{aligned}
\right.
\end{aligned}
$$

具体解法见SMO


求得$a\_i^\*$后
$$ b^\* = y\_j - \sum\_{i=i}^n \alpha\_i ^ \* y\_i(\boldsymbol{x_i} \cdot \boldsymbol{x_j}) ， 0 < a\_j < C， j = 1, \ldots, n $$


分离超平面为：
$$\sum\_{i=1}^m a\_i^\* y\_i(\boldsymbol{x} \cdot \boldsymbol{x\_i}) + b^\* = 0， (其中(\boldsymbol{x\_i}, y\_i）为训练样本及对应标签)$$

分类决策函数为：
$$f(x) = sign \big\[\sum\_{i=1}^m a\_i^\* y\_i(\boldsymbol{x\_i} \cdot \boldsymbol{x}) + b^\* \big\] \ \ \ (\boldsymbol{x} 为待预测样本)$$

当惩罚系数$C \to +\infty$ 时，退化为线性可分的情况

> $\alpha\_i^\* = 0$：非支持向量
> $\alpha\_i^\* = C$：支持向量，但不在支撑超平面上， 支持向量$\boldsymbol{x\_i}$离对应正确分类支撑超平面的距离为:$\frac{\xi\_i}{\lVert\boldsymbol{w}\rVert^2} $
> - $\xi\_i^\* > 1$： x\_i为误分类点
> - $\xi\_i^\* = 1$： x\_i为在分隔超平面上
> - $0 < \xi^\* < 1$： x\_i在分隔超平面和正确支撑超平面之间
>
> $0 < \alpha\_i^\* < C$： 在支撑超平面上的支持向量




### 非线性可分SVM
#### 问题求解
对于线性不可分的情况，使用核函数将原特征映射到更高维度后进行分类
核函数表示为 $K(x, z) = \phi(x) \cdot  \phi(z)$，其中$\phi(x)$ 为某种映射函数

对于线性不可分的情况
$$
\begin{aligned}
 & \max\_\boldsymbol{\alpha} \min\_{\boldsymbol{w}, b}  L(\boldsymbol{w}, b, \boldsymbol{\alpha}) \\\\
= & \\left\\{
\begin{aligned}
    obj\ :\ &\max\_{\boldsymbol{\alpha}} -\frac{1}{2} \sum\_{i=1}^m\sum\_{j=1}^m \alpha\_i \alpha\_j y\_i y\_j  \kappa(\boldsymbol{x\_i}, \boldsymbol{x\_j}) + \sum\_{i=1}^m \alpha\_i, \ \ \ \  i，j  \in [1,\ldots, m]\\\\
    st\ :\ &\ \sum\_{i=1}^m \alpha^\* y\_i = 0，i  \in [1,\ldots, m] \\\\
    &\ 0 \leqslant \alpha\_i \leqslant C， i  \in [1,\ldots, m]
\end{aligned}
\right.
\end{aligned}
$$

具体解法见SMO

求得$a\_i^\*$后
$$ b^\* = y\_j - \sum\_{i=i}^n \alpha\_i ^ \* y\_i \kappa(\boldsymbol{x\_i}, \boldsymbol{x\_j})  ， 0 < a\_j < C， j = 1, \ldots, n $$



分离超平面为：
$$\sum\_{i=1}^m a\_i^\* y\_i \kappa(\boldsymbol{x\_i}, \boldsymbol{x\_j}) + b^\* = 0， (其中(\boldsymbol{x\_i}, y\_i）为训练样本及对应标签)$$

分类决策函数为：
$$f(x) = sign \big\[\sum\_{i=1}^m a\_i^\* y\_i  \kappa(\boldsymbol{x\_i}, \boldsymbol{x}) + b^\* \big\] \ \ \ (\boldsymbol{x} 为待预测样本)$$


### 核函数
- 线性核
$$\kappa(x, z) = (x \cdot z)$$


- 多项式核
$$\kappa(x, z) = (\gamma x \cdot z + r)^p$$

- rbf核（高斯核）
$$\kappa(x, z) = exp \big\(-\frac{\lVert x - z\rVert^2 }{2 \sigma^2}\big\)$$

- 双曲正切核（sigmoid核）
$$\kappa(x, z) = tanh(\gamma x \cdot z  + r)$$



# 损失函数

# 与感知机的区别

# 支持向量回归

# 多分类SVM

# sklearn中的SVM


<https://www.csie.ntu.edu.tw/~cjlin/papers/multisvm.pdf>

[\[!PDF\] Multi-Class Support Vector Machine - Springer](http://www.springer.com/cda/content/document/cda_downloaddocument/9783319022994-c1.pdf?SGWID=0-0-45-1446422-p175468473)

[机器学习核函数手册](https://my.oschina.net/lfxu/blog/478928)











