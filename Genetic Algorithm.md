## 遗传算法 GA

#### 1 GA概述

​		遗传算法是把**问题参数**编码为染色体，再利用迭代的方式进行**选择、交叉以及变异**等运算来交换种群中染色体的信息，最终生成符合优化目标的染色体。

​		**染色体**——**数据或数组**，通常是由一维的串结构数据来表示，串上各位置对应基因的取值。基因组成的串就是染色体，或者称为基因型个体。

​		**群体规模**——一定数量的个体组成了群体。群体中个体的数目称为群体大小。

​		**适应度**——各个个体对环境的适应程度。

##### 1.1 基本步骤

​		1）编码

​		GA在进行搜索之前将解空间的解数据表示成遗传空间的基因型串结构数据，这些串结构数据的不同组合便构成了不同的点。

​		2）初始化群体的生成

​		随机产生N个初始串结构数据，每个串结构数据称为一个个体，N个个体构成了一个群体，GA以这N个串结构数据作为初始点开始进化。

​		3）适应度评估

​		适应度表明个体或解的优劣性。不同的问题，适应性函数的定义方式不同。

​		4）选择

​		选择的原则是适应性强的个体为下一代贡献一个或多个后代的概率大。

​		5）交叉

​		通过交叉操作可以得到新一代个体，新个体组合了其父辈个体的特性，体现了信息交换的思想。

​		6）变异

​		变异首先在群体中随机选择一个个体，对于选中的个体以一定的概率随机地改变串结构数据中某个串地值。通常取值很小。



##### 1.2 解题思路及步骤

​		将自变量在给定范围内进行编码，得到种群编码，按照所选择的适应度函数并通过遗传算法中的选择、变异和交叉对个体进行筛选和优化，使适应度值大的个体被保留。新的群体继承了上一代的信息，又优于上一代，这样反复循环，直至满足条件，最后留下来的个体集中分布在最优解的周围，筛选出其中最优个体作为问题的解。

​		

##### 1.3 MATLAB实现

​		工具箱结构：

| 函数分类     | 函数     | 功能                         |
| ------------ | -------- | ---------------------------- |
| 创建种群     | crtbase  | 创建基向量                   |
|              | crtbp    | 创建任意离散随机种群         |
|              | crtrp    | 创建实值初始化种群           |
| 适应度计算   | ranking  | 基于排序的适应度分配         |
|              | scaling  | 比率适应度计算               |
| 选择函数     | reins    | 一致随机和基于适应度的重插入 |
|              | rws      | 轮盘选择                     |
|              | select   | 高级选择例程                 |
|              | sus      | 随机遍历采样                 |
| 交叉算子     | recdis   | 离散重组                     |
|              | recint   | 中间重组                     |
|              | recline  | 线性重组                     |
|              | recmut   | 具有变异特征的线性重组       |
|              | recombin | 高级重组算子                 |
|              | xovdp    | 两点交叉算子                 |
|              | xovdprs  | 减少代理的两点交叉           |
|              | xovmp    | 通常多点交叉                 |
|              | xovsh    | 洗牌交叉                     |
|              | xovshrs  | 减少代理的洗牌交叉           |
|              | xovsp    | 单点交叉                     |
|              | xovsprs  | 减少代理的单点交叉           |
| 变异算子     | mut      | 离散变异                     |
|              | mutate   | 高级变异函数                 |
|              | mutbga   | 实值变异                     |
| 子种群的支持 | migrate  | 在子种群间交换个体           |
| 实用函数     | bs2rv    | 二进制串到实值的转换         |
|              | rep      | 矩阵的复制                   |



##### 1.4 选择策略

​		**轮盘赌选择法**

​		依据个体的适应度值计算每个个体在子代中出现的概率，并按照此概率随机选择个体构成子代种群。

​		轮盘赌选择策略的出发点是适应度值越好的个体被选择的概率越大。因此，在求解最大化问题的适合，我们可以直接采用适应度值来进行选择。但是在求解最小化问题的时候，我们必须首先将问题的适应度函数进行转换，以将问题转化为最大化问题。下面给出最大化问题求解中遗传算法轮盘赌选择策略的一般步骤：

		1. 将种群中个体的适应度值叠加，得到总适应度值=1；
  		2. 每个个体的适应度值除以总适应度值得到个体被选择的概率；
    		3. 计算个体的累积概率以构造一个轮盘；
      		4. 轮盘选择：产生一个[0,1]区间内的随机数，若该随机数小于或等于个体的累积概率且大于个体1的累积概率，选择个体进入子代种群。
        		5. 重复步骤4，得到的个体构成新一代种群。		

​	

​		**随机遍历抽样法**

​		像轮盘赌一样计算选择概率，只是在随机遍历选择法中等距离的选择个体，设npoint为需要选择的个体数目，等距离的选择个体，选择指针的距离是1/npoint，第一个指针的位置由[0,1/point]的均匀随机数决定。

​	

​		



----

#### 2 基于遗传算法的TSP算法

​		TSP，旅行商问题，是典型的NP完全问题，即其最坏情况下的时间复杂度随着问题规模的增大按指数方式增大，到目前为止还未找到一个多项式时间的有效算法。

​		【装配线上的螺母问题】

​		【产品的生产安排问题】

​		































