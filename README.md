# Reinforement-Learning-from-Scratch

学习强化学习，将练手的代码放在这里。

==Note：安装gym库，pip install gym\=\=0.17.2，否则gridworld.py会报错==

## 主要来源：

### 1.周博磊老师录制的公开课

[这套公开课](https://github.com/zhoubolei/introRL)是香港中文大学周博磊老师录制的精品公开课，课件是英文的，讲授是中文的，附带作业和代码。

b站上面有[全套视频资源](https://space.bilibili.com/511221970?from=search&seid=2201320965631291666)



### 2.叶强老师的知乎专栏

[这个专栏](https://www.zhihu.com/column/reinforce)是对David Silver强化学习公开课的中文讲解

Github上面有对应代码

https://github.com/qqiang00/Reinforce/tree/master/reinforce



### 3.一些博客

[赛艇队长的强化学习专栏](https://blog.csdn.net/hhy_csdn/category_8657689.html?spm=1001.2014.3001.5482)

[Stan Fu的强化学习专栏](https://blog.csdn.net/qq_37266917/category_10194288.html)



### 4.PaddlePaddle的公开课

[视频](https://www.bilibili.com/video/BV1yv411i7xd?from=search&seid=15487704916099010843)

[课件](https://aistudio.baidu.com/aistudio/education/group/info/1335)



## 练手代码

### 1.MDP

包括使用**动态规划**策略迭代和价值迭代解决FrozenLake和FrozenLakeNotSlippery环境。学习这一部分时，首先是看完周博磊老师的第一课，弄明白马尔可夫决策过程，看懂贝尔曼方程，然后手敲代码，再自己学着应用到一个新环境中去，基本就能掌握。



### 2.Monte-Carlo

这一节开始进入model-free的部分。包括使用Monte-Carlo采样的方法解决BlackJack、FrozenLake和FrozenLakeNotSlippery环境。要注意首次访问蒙特卡洛方法和每次访问蒙特卡洛方法的区别。手敲课件上的伪代码进行实现。



### 3.Q_Learning

TD之off policy算法实例。包括使用Q_Learning的算法解决FrozenLake和Taxi-v3环境。引入了PARL的gridworld环境，可以将CliffWalk和FrozenLake环境用turtle绘图进行可视化。自己要学会将算法应用到新的环境中。



### 4.Sarsa

TD之on policy算法实例。包括使用Sarsa的算法解决FrozenLake和Taxi-v3环境。引入了PARL的gridworld环境，可以将CliffWalk和FrozenLake环境用turtle绘图进行可视化。自己要学会将算法应用到新的环境中。本算法与Q_Learning算法仅仅有几行区别。

### 5.Function approximation

函数近似。状态空间过于巨大时，需要函数近似，将未知的状态近似到已知的状态。
