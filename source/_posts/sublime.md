layout: post
title: "Sublime使用过程中的tips"
comments: true
tags:
    - Sublime
date: 2017-03-22 15:21
updated: 2017-03-24 01:04:31
categories:
	- Tools
---

测试环境：Mac  

<!-- more --> 

# 为Sublime添加自动输入日期插件  

2017-03-23 18:17:45  
来源：<http://www.phperz.com/article/14/1125/37633.html>     
1. 创建插件：Tools → Developer → New Plugin   
替换代码为：

``` python
import datetime
import sublime_plugin
class AddCurrentTimeCommand(sublime_plugin.TextCommand):
    def run(self, edit):
        self.view.run_command("insert_snippet", 
            {
                "contents": "%s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
            }
        )
```

保存为：addCurrentTime.py

2. 创建快捷键：Preference → Key Bindings - User:  

``` python
[
    {
        "command": "add_current_time",
        "keys": [
            "ctrl+shift+."
        ]
    }
]
```

保存。"ctrl+shift+."为自动插入日期快捷键  


# Sublime标记修改但未提交的行 
2017-03-22 18:18:10  
来源：<https://github.com/gornostal/Modific>  
```
Cmd+Shift+P -> Package Control: Install Package -> Input: "Modific" -> install...
```
