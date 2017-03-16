layout: post
title: "SVM"
comments: true
tags:
	- SVM
date: 2016-11-03 23:52
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

2，$\ $由于超平面$ \boldsymbol{w} ^ \mathsf{T}\boldsymbol{x} + b = 0$ 有无穷多个等价超平面 $ k(\boldsymbol{w} ^ \mathsf{T}\boldsymbol{x} + b) = 0$， 所以存在等价超平面 $\boldsymbol{w'} ^ \mathsf{T}\boldsymbol{x} + b' = 0$使得不在该超平面上的点$x_0$满足$\left|\boldsymbol{w'} ^ \mathsf{T}\boldsymbol{x_0} + b'\right| = 1$

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


## 求解

# 线性不可分SVM

## 引入松弛变量
### 问题求解

# 非线性不可分SVM

## 核函数

## SMO
算法描述， 流程图..

### 损失函数
图片

### 与感知机的联系区别

## 几点想法

# 多分类SVM
<https://www.csie.ntu.edu.tw/~cjlin/papers/multisvm.pdf>

[\[!PDF\] Multi-Class Support Vector Machine - Springer](http://www.springer.com/cda/content/document/cda_downloaddocument/9783319022994-c1.pdf?SGWID=0-0-45-1446422-p175468473)

# sklearn中的SVM







