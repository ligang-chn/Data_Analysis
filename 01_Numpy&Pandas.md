###   NumPy和Pandas的区别

---

> NumPy是Python的数值计算扩展，专门用来处理矩阵，运算效率比列表高；
>
> Scipy是基于NumPy的科学计算包，包括统计、线性代数等工具；
>
> Pandas是基于NumPy的数据分析工具，能更更方便的操作大型数据集。



#### 1 NumPy

​		numpy的数据结构——**ndarray**（n维数组对象）；

##### 1.1 创建数组

​		创建数组使用numpy的`array`函数，

		>通过numpy，系统自带的列表list可以转换为numpy中的数组；
		>
		>嵌套列表会被转换为一个多维数组（即，矩阵）。

​		==**注意**==：array数组内部的元素必须为==相同类型==，比如数值或字符串。（可以使用dtype查询其类型）

​		**numpy的属性**：

  - `ndim`：维度

  - `shape`：行数和列数

  - `size`：元素个数

    

    **关键字**：	

- `array`：创建数组

- `dtype`：指定数据类型 

- `zeros`：创建数据全为0

- `ones`：创建数据全为1

- `empty`：创建数据接近0

- `arrange`：按指定范围创建数据（是python内置函数range的数组版）

- `linspace`：创建线段

  **numpy的一些数组创建函数**：

  <center>
  <img style="border-radius: 0.3125em;
  box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
  src="assets/1561962173144.png">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9;
  display: inline-block;
  color: #999;
  padding: 2px;">数组创建函数</div>
  </center>

  示例：

  ```python
  a=np.array([2,3,4],dtype=np.float)
  
  #用arange创建连续数据
  a = np.arange(10,20,2) # 10-19 的数据，2步长
  """
  array([10, 12, 14, 16, 18])
  """
  
  #使用reshape改变数据形状
  a = np.arange(12).reshape((3,4))    # 3行4列，0到11
  """
  array([[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]])
  """
  
  #用linspace创建线段型数据
  a = np.linspace(1,10,20)    # 开始端1，结束端10，且分割成20个数据，生成线段
  """
  array([  1.        ,   1.47368421,   1.94736842,   2.42105263,
           2.89473684,   3.36842105,   3.84210526,   4.31578947,
           4.78947368,   5.26315789,   5.73684211,   6.21052632,
           6.68421053,   7.15789474,   7.63157895,   8.10526316,
           8.57894737,   9.05263158,   9.52631579,  10.        ])
  """
  
  ```

  ---

  

##### 1.2 数据类型

​		dtype（数据类型）

​		通常只需要知道所处理的数据大致是浮点数、复数、整数、布尔值、字符串，还是普通的Python对象。

​		`astype`：将一个数组从一个dtype转换成另一个dtype；

----

##### 1.3 数组运算

		>数组的计算非常方便，不用大量的循环即可批量运算。
		>
		>大小相等的数组之间的任何算术运算都会将运算应用到元素级。
		>
		>数组与标量的算术运算会将标量值传播到各个元素。
		>
		>大小相同的数组之间的比较会生成布尔值数组。
		>
		>不同大小的数组之间的运算叫做**广播**。



```ptyhon
c=a-b
c=a+b
c=a*b
c=b**2
c=10*np.sin(a)  #sin函数
print(b<3)  #在脚本中对print函数进行修改可以进行逻辑判断，返回一个bool类型的矩阵
```

​		对于多维数组的运算：

​		Numpy中的矩阵乘法分为两种：

  - 对应元素相乘；

  - 标准的矩阵乘法运算，即对应行乘对应列得到相应元素；

    ```
    #第一种表示方法
    c_dot=np.dot(a,b)
    
    #第二种表示方法
    c_dot=a.dot(b)
    ```

    - `sum()`
    - `min()`
    - `max()`
    - `argmin()`：求最小元素的索引；
    - `argmax()`：求最大元素的索引；
    - `mean()`：均值=`average`
    - `median()`：中位数；
    - `cumsum()`：累加函数，生成的每一项矩阵元素均是从原矩阵首项累加到对应项的元素之后；
    - `diff()`：累差运算，每一行中后一项与前一项之差；
    - `nonzero()`：将所有非零元素的行与列坐标分隔开，重构成两个分别关于行和列的矩阵；
    - `sort()`：排序
    - `transppose()`：转置，即**np.transpose(A)=A.T**
    - `clip(Array,Array_min,Array_max)`：后面的最大最小值让函数判断矩阵中元素是否有比最小值小的或者比最大值大的元素，并将这些指定的元素转换为最小值或者最大值。

    

    如果需要对行或列进行查找运算，就需要在上述代码中为`axis`进行赋值，
    
    - `axis=0`：以列作为查找单元；
    - `axis=1`：以行作为查找单元；

---

##### 1.4 索引

​		**一维索引**

​		与元素列表或者数组中的表示方法类似；

​		==**注意**==：

		- array[5:8]在这里不包括8这个位置，即不包括结尾。
		- 跟列表最重要的区别在于，数组切片是原始数组的视图，数据不会被复制，视图上的任何修改都会直接反映到源数组上。【但是，某些情况下，你需要得到的是ndarray切片的一份副本而非视图，就需要明确地进行复制操作，例如`arr[5:8].copy()`。】

​		

​		**二维索引**

​		可以利用`:`对一定范围内的元素进行切片操作；

​		

​		**切片索引**

​		切片是沿着一个轴向选取元素的；



​		**关于迭代输出的问题**：

```python
import numpy as np
A = np.arange(3,15).reshape((3,4))
         
print(A.flatten())   
# array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

for item in A.flat:
    print(item)
    
# 3
# 4
……
# 14
```

​		这一脚本中的`flatten`是一个展开性质的函数，将多维的矩阵进行展开成1行的数列。而`flat`是一个迭代器，本身是一个object属性。

---

##### 1.5 array合并

​		对一个`array`的合并，可以按行、列等多种方式进行合并。






