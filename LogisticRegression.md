### 逻辑回归算法

1、逻辑回归与线性回归的联系与区别

2、逻辑回归的原理

3、逻辑回归损失函数推到及优化

4、正则化与模型评估指标

5、逻辑回归的优缺点

6、样本不均衡问题解决办法

7、sklearn参数





----

#### 1 逻辑回归

​		**解决分类问题**。

​		回归问题怎么解决分类问题？

​		将样本的特征和样本发生的概率联系起来，概率是一个数。

​		![1566727526856](LogisticRegression.assets/1566727526856.png)

​		如果对于计算出的概率进行判断，那么就可以进行分类操作，即分类算法。



----

#### 2 逻辑回归的损失函数

​		![1566727888926](LogisticRegression.assets/1566727888926.png)

​		![1566727905725](LogisticRegression.assets/1566727905725.png)

​		![1566731596811](LogisticRegression.assets/1566731596811.png)

​		![1566731651026](LogisticRegression.assets/1566731651026.png)

​		![1566732483573](LogisticRegression.assets/1566732483573.png)

​		![1566732501385](LogisticRegression.assets/1566732501385.png)

​		![1566732657701](LogisticRegression.assets/1566732657701.png)

​		![1566732770117](LogisticRegression.assets/1566732770117.png)

​		只能使用**梯度下降法**求解。

​		

---

#### 3 决策边界

​		![1566736702870](LogisticRegression.assets/1566736702870.png)

​		![1566736925024](LogisticRegression.assets/1566736925024.png)

​		![1566737078896](LogisticRegression.assets/1566737078896.png)

​		



​		![1566737335730](LogisticRegression.assets/1566737335730.png)

---

#### 4 在逻辑回归中使用多项式特征

​		





​		![1566739273855](LogisticRegression.assets/1566739273855.png)



​		![1566740070817](LogisticRegression.assets/1566740070817.png)

​		![1566740214455](LogisticRegression.assets/1566740214455.png)

​		![1566740324089](LogisticRegression.assets/1566740324089.png)







----

#### 5 分类准确度的评价

​		![1566742037545](LogisticRegression.assets/1566742037545.png)

​		

​		**精准率和召回率**：

​		![1566742240288](LogisticRegression.assets/1566742240288.png)

​		![1566742353540](LogisticRegression.assets/1566742353540.png)

​		![1566742519753](LogisticRegression.assets/1566742519753.png)



​		**权衡精准率和召回率**：

​		![1566785873859](LogisticRegression.assets/1566785873859.png)

​		![1566785914158](LogisticRegression.assets/1566785914158.png)

​		

​		![1566821363029](LogisticRegression.assets/1566821363029.png)

​		![1566824105670](LogisticRegression.assets/1566824105670.png)

​		



​		![1566824175923](LogisticRegression.assets/1566824175923.png)

​		![1566824210823](LogisticRegression.assets/1566824210823.png)

​		![1566824239398](LogisticRegression.assets/1566824239398.png)

​		![1566826100590](LogisticRegression.assets/1566826100590.png)

