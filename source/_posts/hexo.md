layout: post
title: "Hexo使用过程中遇到的问题"
comments: true
tags:
	- Hexo
date: 2017-03-21 15:56 
updated: 2017-03-29 16:09:56
categories:
    - Tools
---
<!-- more -->

# kill占用4000端口的程序  
2017-03-29 16:09:39
```bash 
lsof -i tcp:4000  | awk '{print $2}' | xargs kill -9
```

# Hexo 为文章添加更新日期    
2017-03-22 18:32:20  
来源: <http://wuchenxu.com/2015/12/13/Static-Blog-hexo-github-7-display-updated-date>

1，增加主题代码  
在hexo/themes/next/layout/_macro/post.swig的theme.post_meta.created_at 的"endif"模块后添加  

```html
{%if post.updated and post.updated > post.date%}
  <span class="post-meta-divider">|</span>
  <span class="post-meta-item-icon">
  <i class="fa fa-calendar-o"></i>
  </span>
  <span class="post-meta-item-text">{{ __('post.updated') }}</span>
  <span class="post-updated">
  <time title="{{ __('post.updated') }}" itemprop="dateCreated dateUpdated" datetime="{{ moment(post.date).format() }}">
    {{ date(post.updated, config.date_format) }}
  </time>
{% endif %}
```

2，配置语言文件 hexo/themes/next/languages/zh-Hans.yml  
在key为post的value中添加"updated: 更新于"  

```
post:
  ...
  ...
  updated: 更新于
```

3，主题文件配置  
在hexo/themes/next/_config.yml添加一行  
```
display_updated: true
```

4，文章开头设置  
```
updated: 2015-12-13 20:18:54
```


# Hexo 数学公式问题  
2017-03-23 18:32:41  
配置主题目录下的_config.yml的开关mathja
```
enable: true
```
问题:
编辑公式是在latex编辑器里编辑正确的公式在hexo不能被正确的渲染

解决:

- 方案2(建议):更换Hexo的markdown渲染引擎    
来源 在Hexo中渲染MathJax数学公式，<http://www.jianshu.com/p/7ab21c7f0674> 

卸载后重新安装:
```
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-pandoc --save
```
执行以上命令配置生效，git clean -> hexo s

- 方案1(麻烦):在不该配置文件的情况下，用二分法在'\_'、 '['、'('、'{'、'*'、'\'等处加转义尝试，如'\_'


# Hexo本地测试与发布到git
本地测试:
```
hexo s
```

必要时
```
hexo clean
```

编译并发布到github:
```
hexo d -g
```
# Hexo正文中标题自动编号  
参考：为Hexo博客标题自动添加序号 hexo-heading-index，<http://www.tuicool.com/articles/7BnIVnI>
1， 安装hexo-heading-index  
```
npm install hexo-heading-index --save
```


2， 修改顶层_config.yml  
```
heading_index:
enable: true
index_styles: "{1} {1} {1} {1} {1} {1}"
connector: "."
global_prefix: ""
global_suffix: ". "
```


3， 修改Hexo主题下的_config.yml， 避免侧边栏重复自动生成编号，禁用侧边栏自动编号  
```
# Table Of Contents in the Sidebar
toc:
  enable: true

  # Automatically add list number to toc.
  number: false
```






