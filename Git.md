## Git

#### 1 基本概念

------------------------------------

​		**仓库（Repository）**

​		仓库用来存放项目代码，每个项目对应一个仓库，多个开源项目则有多个仓库。

​		**收藏（Star）**

​		**复制克隆项目（Fork）**

​		**发起请求（Pull Request）**

​		**关注（Watch）**

​		**事物卡片（Issue）**

​		发现代码BUG，但是目前没有成型代码，需要讨论时用；



#### 2 向仓库中添加文件流程

------------------

​		![img](file:///C:/Users/ligang/AppData/Local/Packages/oice_16_974fa576_32c1d314_1013/AC/Temp/msohtmlclip1/01/clip_image002.jpg)



#### 3 Git初始化

------------

##### 3.1 基本信息设置

```
1.设置用户名
git config --global user.name 'ligang-chn'

2.设置用户名邮箱
git config --global user.email '123456789@qq.com'

Note:该设置在GitHub仓库主页显示谁提交了该文件。
```



##### 3.2  初始化一个新的Git仓库

​		1、在本地创建项目文件夹，如Data_Analysis；

​		![1561018299611](assets/1561018299611.png)

​		也可以使用命令mkdir Data_Analysis。

​		2、在文件内初始化git（创建git仓库）

```
git init
```

​		可以在项目文件下看到.git文件夹，默认隐藏。

​		![1561018592279](assets/1561018592279.png)

​		3、向仓库添加文件

```
git remote add origin git@github.com:ligang-chn/Data_Analysis.git

git pull git@github.com:ligang-chn/Data_Analysis.git#在本地同步仓库的内容

git add 文件  #添加到暂存区
git commit -m '版本提示符' #添加到仓库
```

​		![1561021014575](assets/1561021014575.png)

​		![1561021047704](assets/1561021047704.png)





