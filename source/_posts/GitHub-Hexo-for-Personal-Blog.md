---
title: GitHub+Hexo for Personal Blog
date: 2019-10-09 00:37:38
categories: Network
tags:
  - GitHub
  - Hexo
  - Markdown
  - NexT
---

## 前言
几个月前准备保研，在网上搜索老师信息时，进了一位老师的个人博客，当时有被“性冷淡”的博客界面风格吸引到（后来发现还是有挺多用这个主题的...）。保研去向尘埃落定，虽然最后没有跟随那位“博主”老师（因为压根就没有回我邮件2333，故事还没开始就结束了），但在这段清闲的日子还是想着自己也弄个Blog打发一下时间。随便在网上搜索了一下，决定了用GitHub + Hexo搭建个人博客，之后顺藤摸瓜从那位老师的GitHub对应仓库找到那款 ~~“性冷淡”~~ 简约风格主题——NexT。  

写这篇的原因一是为了记录，但其实类似的教程多不胜数，我自己也看了许多人写的搭建过程，都是大同小异。所以这里只记录最最基本的流程以及少数自己用到的个性化操作，后续有增加的话再来修改这一篇或者另写就行了。写这篇的另一个原因就是熟悉一下Markdown语法，还是多有趣的一件事:)  

**Hexo**: <https://hexo.io>  
**Markdown Doc**(Simple Chinese): <https://markdown-zh.readthedocs.io/en/latest/>  
**NexT v6 and v7**: <https://github.com/theme-next/hexo-theme-next>  

## Preinstallation
- install Node.js <https://nodejs.org/en/download/>
- install git <https://git-scm.com/download/win>
- install hexo(*Bash* recommended) `$npm install hexo-cli -g`

## SSH keys
-  git bash `$ cd ~/. ssh` to check exisiting ssh keys, if ***'No such file or directory'*** , then  
`ssh-keygen -t rsa -C "your_email@example.com" ` to generate key file  
- `$ cat ~/.ssh/id_rsa.pub` and copy the content in the key file  
- create key in GitHub: account Settings->SSH and GPG keys->New SSH keys  
- check the SSH key `$ ssh -T git@github.com #do not modify the address` and ***'Hi MatthewXY01! You've successfully authenticated, but GitHub does not provide shell access.'*** represents successful configuration  

## Initialization
- create an empty folder called *hexo* (whatever) *eg.* 'M:\hexo'  
- *git bash here* and  
``` bash
$ hexo init  #hexo会自动下载一些文件到这个目录，包括node_modules
$ hexo g     #或者 hexo generate 生成文件
$ hexo s     #或者 hexo server 启动服务，之后hexo 在public 生成相关html，可通过local test 查看本地预览
```
local test: <http://localhost:4000/>  

## Deploy  
- install deployer: git bash `npm install hexo-deployer-git --save`
- in **hexo** folder, modify the **_config.yml** file as follows:
``` yml
deploy:
        type: git  
        repository: git@github.com:MatthewXY01/MatthewXY01.github.io.git #SSH address
        branch: master
```
{% asset_img SSH_address.png SSH_address %}  
- type `$ hexo g` and `$ hexo d` or type `$ hexo d -g`

## Theme---*NexT*
- the simplest way to install is to clone the entire repository:  
```
$ cd hexo  
$ git clone https://github.com/theme-next/hexo-theme-next themes/next-reloaded
```
- in the **site** _config.yml(/_config.yml), set *'theme: next-reloaded'*
- config the **theme** _config.yml in the file '/themes/next-reloaded/_config.yml'. (many subsequent modifications are made here!)

### Scheme
- in the **theme** _config.yml, set *'scheme: Mist'*  

### Local search
- `$npm install hexo-generator-searchdb --save`
- in the **theme** _config.yml, set *'local_search: enable: true'*
- in the **site** _config.yml, paste the following lines anywhere:  
``` yml
search:
  path: search.xml
  field: post #search range, can be 'post', 'page', 'all'
  format: html
  limit: 10000
```

### Social links  
- in the **theme** _config.yml, set as follow:  
``` yml
social:
  GitHub: https://github.com/MatthewXY01 || github
  E-Mail: mailto:mxinyuan@foxmail.com || envelope
```

### Menu
- in the **theme** _config.yml, search *'menu'* and set as follow:  
``` yml
menu:
  home: / || home
  tags: /tags/ || tags
  categories: /categories/ || th
  archives: /archives/ || archive
```
- create new pages and link to the menu `$ hexo new page categories`
- edit the '/source/categories/index.md' file of the page just created. add *'type: categories'* 
- similar method for the creation of the 'tags' page

## Posts
### Create a new article
- in the **theme** _config.yml, modify *'auto_excerpt'*, sothat home page shows only the excerpt and click *Read more* to read the full article :  
``` yml
auto_excerpt:
  enable: true
  length: 150 #length of the excerpt
```
- in the **site** _config.yml, set *'post_asset_folder'* to *'true'*, sothat every time we create  new posts, it will automatically generates folders with the same name(Used to store audio and video resources)  
- in **hexo** folder git bash here and enter the command *hexo new 'postname'*  

### Tags and category
- open '/source/_posts/postname.md'
- in the beginning of the post, set tags and categoty: 
``` yml
categories: categoty_example
tags: #multi-tag is available
  - tag1
  - tag2
  - tag3
  - ...
```

## Tips
### YMAL
- case sensitive
- in 'key-value pair', do not ignore a space after the colon

### Markdown representation
- If something is wrong with the representation and hard to tune, you can try deleting extra spaces or adopting an alternative format  
e.g. code block in  format:  
\`\`\`
code  
\`\`\`
some representation error occurs when there's extra space following the last "\`\`\`" 

### 404
case sensitive: sometimes you switch between capital letter and small letter, which may lead to 404 not found errors  
- set `ignorecase` as false in the file *'.deploy_git/.git/config'*
- `$ hexo clean` and `$ hexo d -g`