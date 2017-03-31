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

论文地址：<https://arxiv.org/pdf/1702.08835.pdf>  
部分翻译：<http://it.sohu.com/20170302/n482153688.shtml>

<!-- more -->

# 算法基本思路  
gcForest是西瓜书作者周志华博士和冯霁博士提出的一基于随机森林的深度森林的方法，是一种“ensemble of ensembles”的方法。类似深度神经网络，深度森林是每层是由多个随机森林组成，每层的随机森林是由完全随机森林及随机森林组成  

- **完全随机**森林的构建：构建1000个(超参数)**完全随机**树。完全决策树的构建过程：对于所有的特征，随机选择特征，随机选择特征下的split值，一直生长，直到每个叶节点包含相同的类别或者不超过10（超参数）个时，决策树训练停止

- 随机森林的构建：构建1000个(超参数）决策树。决策树的构建过程：从$d$个特征，随机抽取$\sqrt{d}$个特证，由gini系数做为特征选择及分裂的标准构建CART决策树
 
# 算法流程  
## 类向量的训练
```
假设为k分类问题， 
1，针对已经训练好的森林中的树时，记录每个叶子节点的样本类别，按类别统计叶子节点的权重得到k维向量，树模型的每个叶子节点都对应一个k维向量(带key的向量，如，label_1：0.5, label_2:0.3, lable_3=0.2)。
2，给定一个样本经过树的运算到达叶子节点，对应一个k维向量，一个随机森林中对应1000个k维向量， 将1000个k维向量按照类别求平均，平均后的k维向量即为该样本在该森林上的类向量
```
## 模型训练 
```python 
def model(input_data_list_list, label_list):
	"""
	input_data_list_list: 训练样本列表
	label_list:  标签列表
	"""

	i ＝ -1  ＃ Cascade，第i层  
	feture_list_list = []
	
	model_list_list = []
	feture_list_list[0] = input_data
	performance = 0 ＃ 初始化准确率
	while 1:
		i += 1
		
		以feture_list［i］做为样本， label_list为样本标签 训练完全随机森林和随机森林
		并行训练m个完全随机森林存入modle_complete_rf_list, n个随机森林存入modle_rf_list
		model_list_list[i] = [modle_complete_rf_list, modle_rf_list]
		
		
		for 特征向量 in 特征向量的向量：  # 遍历每个样本(特征向量)
			获得特征向量在m+n个森林上的m＋n个类向量
			feture_list_list[i+1][j]  <- 将获得的m＋n个累向量串行化为(m+n)* k个特征
			预测：将m+n个类向量按类别求平均，求最大值对应的类别
			
		new_performance <- 统计所有样本预测结果

		# 对比上层森林群的结果，对比上层性能增加是否达到阈值theta（阈值： 超参数）
		if new_performance - performance > theta:
			return model_list_list ＃ 输出模型
			
		# 更新性能统计
		performance = new_performance

```

## 使用模型预测
```python
def fit(x, model_list_list):
	feture_list = x
	
	# 循环每次层级连森林
	for model_list in model_list_list:
		针对m+n个森林，得到m＋n类向量
		feture_list <- 合并m＋n个类向量
	
	feture_list按标签分组，分别求权重均值。找出最大平均值对应的类别即为预测结果
```
	





