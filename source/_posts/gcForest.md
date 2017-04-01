layout: post
title: "gcForest"
comments: true
tags:
	- 集成方法
	- gcForest
date:  2017-03-31 02:29:40
categories:
    - 机器学习  
    - 集成方法
---

{% cq %} <font size=4>multi-Grained Cascade Forest</font>{% endcq %}


决策树模型具有可解释性强的优点，多个决策树构成随机森林，多层多个森林构成深度森林  

gcForest是西瓜书作者周志华博士和冯霁博士提出的一基于随机森林的深度森林的方法，尝试用深度森林的方法解决深度神经网络中存在的问题： 

- 需要大量样本才能使得深度神经网络有较好的性能
- 调参困难
- 对硬件性能要求较高  

论文中实验了在小数据集上gcForest，取得了不错的效果  

论文地址：<https://arxiv.org/pdf/1702.08835.pdf>  
部分翻译：<http://it.sohu.com/20170302/n482153688.shtml>

<!-- more -->

# 算法基本思路  
gcForest是西瓜书作者周志华博士和冯霁博士提出的一基于随机森林的深度森林的方法，是一种“ensemble of ensembles”的方法。类似深度神经网络，深度森林是每层是由多个随机森林组成，每层的随机森林是由完全随机森林及随机森林组成  

>
- **完全随机**森林的构建：构建1000个(超参数)**完全随机**树。完全决策树的构建过程：对于所有的特征，随机选择特征，随机选择特征下的split值，一直生长，直到每个叶节点包含相同的类别或者不超过10（超参数）个时，决策树训练停止
- 随机森林的构建：构建1000个(超参数）决策树。决策树的构建过程：从$d$个特征，随机抽取$\sqrt{d}$个特证，由gini系数做为特征选择及分裂的标准构建CART决策树
 
每一层经过多个森林处理的输出作为下级的输入，当到达某一层没有明显的性能提升(超参数)时，级连森林停止生长

# 算法流程  
## 类向量的训练
<img src="/pic/ml/gcForest/gcForest_class_vector.png" width="80%" height="80%" style="margin: 0 auto">
<center>（[图1，类向量生成示意图](https://arxiv.org/pdf/1702.08835.pdf))</center>

假设为k分类问题   
1，针对已经训练好的森林中的树时，记录每个叶子节点的样本类别，按类别统计叶子节点的权重得到k维向量，树模型的每个叶子节点都对应一个k维向量(带key的向量，如，label_1：0.5, label_2:0.3, lable_3=0.2)。 


2，给定一个样本经过树的运算到达叶子节点，对应一个k维向量，一个随机森林中对应1000个k维向量， 将1000个k维向量按照类别求平均，平均后的k维向量即为该样本在该森林上的类向量

## 级连森林训练  
<img src="/pic/ml/gcForest/gcForest_struct.png" width="80%" height="80%" style="margin: 0 auto">
<center>（[图2，级连森林模型示意图](https://arxiv.org/pdf/1702.08835.pdf))</center>

伪代码如下：
```python 
def model(input_data_list_list, label_list):
    """
    input_data_list_list: 训练样本列表
    label_list:  标签列表
    """

    i ＝ -1  ＃ 第i层级连森林 
    feture_list_list = []
	
    model_list_list = []
    feture_list_list[0] = input_data
    performance = 0 ＃ 初始化准确率
    while 1:
        i += 1
		
        modle_complete_rf_list, modle_rf_list <- 以feture_list[i]为特征， label_list为样本标签，并行训练完全随机森林和随机森林
        model_list_list[i] = [modle_complete_rf_list, modle_rf_list]
        
        # 统计该层的性能（如：正确率，准确率，召回率等）
        new_performance <- 根据各森林的每个树的叶子节点统计性能
			
        # 对比上层森林群的结果，对比上层性能增加是否达到阈值theta（阈值： 超参数）
        if new_performance - performance > theta:
            return model_list_list ＃ 输出模型
			
        # 更新性能统计
        performance = new_performance

```

## 级连森林预测  
伪代码如下：
```python
def fit(x, model_list_list):
    '''
    x：待预测样本
    model_list_list：深度森林模型，model_list_list[0]：列表，代表第0层的森林模型列表
    '''
    feture_list = x
	
    # 循环每层级连森林
    for model_list in model_list_list:
        class_vector_list <- 获得特征向量在m+n个森林上的m＋n个类向量
        feture_list <- 串行化类向量class_vector_list
	   
    result <- 按标签分组，分别求权重均值。找出最大平均值对应的类别即为预测结果
```
	

# 使用多粒度扫描做特征处理
- 多粒度扫描结构
<img src="/pic/ml/gcForest/gcForest_multi-grained-scanning.png" width="100%" height="100%" style="margin: 0 auto">
<center>（[图3，多粒度扫描示意图](https://arxiv.org/pdf/1702.08835.pdf))</center>
  
- 带多粒度扫描的级连森林结构
<img src="/pic/ml/gcForest/gcForest_struct-with-grained-scanning.png" width="100%" height="100%" style="margin: 0 auto">
<center>（[图4，带多粒度扫描的增强级连森林示意图](https://arxiv.org/pdf/1702.08835.pdf))</center>

- 描述：  
类似卷机神经网络的Pooling，深度森林也引入"滑动窗口"，替代pooling层的方法max-pooling, mean-pooling的计算方式为多个森林（完全随机森林和随机森林）后的类向量串行，具体过程大概如下：
每次滑动窗口选出来的特征经过随机森林和完全随机森林经过多个森林建模后得到类向量，串联类向量作为新的特征作为下一级的输入层
- 假设原始特征长度为m，滑动窗口长度为n(n < m)，滑动窗口个数：[n, m]即共有m-n+1个滑动窗口
- 可以并行接入不同窗口做Pooling操作






