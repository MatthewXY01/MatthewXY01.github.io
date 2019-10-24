---
title: Back up Hexo blog source files with GitHub branch
date: 2019-10-24 16:18:21
categories: Network
tags:
  - GitHub
  - Hexo
---
## Goal
It's easy to deploy blog by Hexo. However the `hexo d -g` command can only push `.deploy_git` folder's content to the repo. Just in case the computer is broken or we're going to write a blog on another computer, it's necessary to back up the blog's source files in another place. We can use a new branch of the blog repo to achieve that.(note that the default branch `master` is used for Hexo to deploy static website files).  

## Before pushing
1. enter the blog directory(eg. /hexo ) `git init`  
2. `git add .` Hexo has already made a `.gitignore` file, edit it as needded  
3. `git commit -m "commit first time"`  
4. `git remote add origin git@github.com:MatthewXY01/MatthewXY01.github.io.git`  

## Push source files to the new branch
```bash
git branch source #新建分支 source
git branch #查看分支
git checkout source #切换分支
```
Then, pull codes(which we just commit) and push them in the new branch:  
```bash
git pull origin master
git push -u origin source
```
And we can find the source files in the new branch of the repo.  

## Update blog
Every time we modify source files (eg. new posts, new configuration in _config.yml etc.), we can push them to the new branch:  
```bash
git add .
git commit -m 'modify blog'  
git push --set-upstreams origin source # configure 'push', this command only needs to be executed once
git push
```

## Edit source files in another computer
1. refer the part **Preinstallation** and **SSH keys** in [GitHub+Hexo for Personal Blog](https://matthewxy01.github.io/2019/10/09/GitHub-Hexo-for-Personal-Blog/#more)  
2. clone the repo
3. `git checkout origin/source`
4. enter the blog directory and `npm install hexo-deployer-git --save`
5. write blogs