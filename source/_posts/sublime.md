layout: post
title: "Sublime使用过程中的tips"
comments: true
tags:
	- Tips
date: 2017-03-22 15:21
updated: 2017-03-23 15:50:57
categories:
    - Sublime
---

测试环境：Mac  

# 为Sublime添加自动输入日期插件  

<!-- more -->  

参考：<http://www.phperz.com/article/14/1125/37633.html>   

1. 创建插件：Tools → New Plugin   
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

2. 创建快捷键：  

```
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


# 在文件内部显示与git版本库修改和新增记号

```
Cmd+Shift+P -> Package Control: Install Package -> Input: "Modific" -> install...
```

参考：  
<https://github.com/gornostal/Modific>  

