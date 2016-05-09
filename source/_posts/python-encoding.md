---
layout: post
title: "python2.7编码处理小结"
date: 2016-04-23 23:52
comments: true
tags: 
	- python
	- tips
---

python2.7编程中遇到的一些编码问题小结

<!-- more -->
## python code中指定编码方式

**1.指定源文件的的编码方式**
作用:解决python解释器载入py程序文件时编码报错问题
``` python
	#coding=utf8  
```

``` python
	#coding:utf8
```

``` python
	# -*- coding: UTF-8 -*-
```
以上方式效果是等价的,选择一种使用即可

**2.指定中文代码默认解码方式**
作用:解决解释器环境输出文本时乱码的部分问题
``` python
	import sys
	reload(sys)
	sys.setdefaultencoding('utf-8')
```
关于reload(sys)?:
setdefaultencoding函数在第一次系统调用后会被删除,另外由于python的import的"ifndefine"机制,import sys不能保证sys被加载,所以由reload显式加载sys


## 关于未知编码转utf8编码

> 一个实际数据场景:数据来源和存储形式多样,编码多样,需要将不同编码转化为utf8编码进行后续加工处理

``` python
	import chardet
	buf = '这是一个编码'
	str_detect_dict = chardet.detect(buf)
	buf = unicode(buf, str_detect_dict['encoding'], "ignore")
	print buf, str_detect_dict
	>>这是一个编码 {'confidence': 0.99, 'encoding': 'utf-8'}
```

以上代码可以解决遇到的比较多得编码转换问题,但之前有碰到一些调皮些的编码问题
``` python
	import chardet
	buf = '\u8fd9\u662f\u4e00\u4e2a\u7f16\u7801\u95ee\u9898'
	str_detect_dict = chardet.detect(buf)
	buf = unicode(buf, str_detect_dict['encoding'], "ignore")
	print buf, str_detect_dict
	>>\u8fd9\u662f\u4e00\u4e2a\u7f16\u7801\u95ee\u9898 {'confidence': 1.0, 'encoding': 'ascii'}
```
好尴尬,chardet.detect以绝对的自信检测出该编码为ascii,所以unicode函数没有起作用


### 介绍两个函数
>repr(): 返回机器存储方式

``` python
	buf = u'这是一个编码'
	print repr(buf)
	>>u'\u8fd9\u662f\u4e00\u4e2a\u7f16\u7801'
	buf = '这是一个编码'
	print repr(buf) 
	>>'\xe8\xbf\x99\xe6\x98\xaf\xe4\xb8\x80\xe4\xb8\xaa\xe7\xbc\x96\xe7\xa0\x81'
```


>eval(): 将字符串str当成有效的表达式来运行并返回结果

``` python
	result = eval('12+3')
	print type(result), result
	>><type 'int'> 15
```

是不是想到了sql注入攻击?

### 回到调皮编码的问题

``` python
	buf = '\u8fd9\u662f\u4e00\u4e2a\u7f16\u7801\u95ee\u9898' 或 buf = '乱码字符串'
```

以上编码离repr('这是一个编码')在存储上只差一个'u'
个人处理chardet.dect解决后仍有问题的编码,通常按一下方式解决:
>1.透过现象看本质,repr()后查看字符串的机器存储方式
>2.找规律,找这些编码方式和已知编码方式的相似处和区别
>3.手术刀,修改机器存储方式,通过eval()函数整合为新的编码的字符串

解决问题过程大概如下:

``` python
	import chardet
	buf = '\u8fd9\u662f\u4e00\u4e2a\u7f16\u7801\u95ee\u9898' #或 buf = '一堆乱码'
	print repr(buf)
	>>'\\u8fd9\\u662f\\u4e00\\u4e2a\\u7f16\\u7801\\u95ee\\u9898'
	new_buf_repr = 'u"%s"' % buf #或new_buf_repr = 'u"%s"' % repr(buf)
	print new_buf_repr
	>>"\u8fd9\u662f\u4e00\u4e2a\u7f16\u7801\u95ee\u9898"
	new_buf = eval(new_buf_repr)
	print type(new_buf), new_buf
	>><type 'unicode'> 这是一个编码问题
```

### 这种数据虽然有点狗血.如果是成批量的话可参考以上解决方式.另外极少数的编码方式是混合编码,这部分有要的时候需要用re来解决
