## python_pandas
This repository is used for learning pandas

### Install
```python
pip install pandas
```

### Pandas 数据结构

### Series
Series 是一种一维数组，和 NumPy 里的数组很相似。事实上，Series 基本上就是基于 NumPy 的数组对象来的。和 NumPy 的数组不同，Series 能为数据自定义标签，也就是索引（index），然后通过索引来访问数组中的数据。创建一个 Series 的基本语法如下：
```python
my_series = pd.Series(data, index)
```
data参数可以是任意的数据对象，比如dict、list、numpy array.
index参数是对data的索引值，类似字典的key。
例：
```python
index = ['USA', 'Nigeria', 'France', 'Ghana']
data = [100, 200, 300, 400]
my_series = pd.Series(data,index)

#out:
USA        100
Nigeria    200
France     300
Ghana      400
dtype: int64
```
index参数可以省略，默认索引[0,1,... len(data)-1]
```python
#从Numpy数组创建Series
data = [100, 200, 300, 400]
np_arr = np.array(data)
my_series = pd.Series(data)

#out:
0    100
1    200
2    300
3    400
dtype: int64

#从dict创建Series
my_dict = {'a': 100, 'b': 200, 'c':300, 'd':400}
my_series = pd.Series(my_dict)

#out:
a    100
b    200
c    300
d    400
dtype: int64

#访问Series的类似dict
a = my_series[1]
usa_data = my_series['USA']

#out
100
```
#### 对 Series 进行算术运算操作
对Series的算术运算都是基于index进行的。加减乘除等运算对两个Series，pandas都是根据对应index，对数据进行计算，结果会以浮点数保存，以避免精度丢失。
```python
series1 = pd.Series([1,2,3,4], ['London','USA', 'France', 'HK'])
series2 = pd.Series([1,2,3,4], ['London','USA', 'JP', 'HK'])
res1 = series1 - series2

#out
France    NaN
HK        0.0
JP        NaN
London    0.0
USA       0.0
dtype: float64

res2 = series1 + series2

#out
France    NaN
HK        8.0
JP        NaN
London    2.0
USA       4.0
dtype: float64

res3 = series1 * series2

#out
France     NaN
HK        16.0
JP         NaN
London     1.0
USA        4.0
dtype: float64

res4 = series1/series2

#out
France    NaN
HK        1.0
JP        NaN
London    1.0
USA       1.0
dtype: float64
```
如果 Pandas 在两个 Series 里找不到相同的 index，对应的位置就返回一个空值 NaN。

### DataFrames
Pandas的DataFrame是一种二维数据结果，数据以表格的形式存储，分成若干行和列。通过 DataFrame，你能很方便地处理数据。常见的操作比如选取、替换行或列的数据，还能重组数据表、修改索引、多重筛选等。

#### Create DataFrames
##### 1.通过二维数组创建数据框
```python
df1 = pd.DataFrame(np.arange(12).reshape((3, 4)))
#创建3行四列的数据，默认index是[0,1,2],默认columns [0,1,2,3]

dates = pd.date_range('20190425', periods=6)
#out:
DatetimeIndex(['2019-04-25', '2019-04-26', '2019-04-27', '2019-04-28',
               '2019-04-29', '2019-04-30'],
              dtype='datetime64[ns]', freq='D')
 
df = pd.DataFrame(np.random.randn(6,4),index=dates, columns=['a', 'b', 'c','d'])

#out
                   a         b         c         d
2019-04-25 -1.166267  0.435387  1.410483  0.052759
2019-04-26  0.937992 -0.480694  0.318787 -0.449206
2019-04-27 -0.733083 -0.982927 -0.966134  0.422478
2019-04-28  0.480559  0.466605  1.753447  0.324091
2019-04-29  1.024889 -0.962393  0.908883  0.676290
2019-04-30  0.211393  0.897806  0.129817 -2.332275
```

##### 2.通过字典构造数据
```python
df2 = pd.DataFrame({'A' : 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series( 1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3]*4, dtype='int32'),
                    'E': pd.Categorical(['TEST', 'TRAIN', 'test', 'train']),
                    'F': 'foo'})
#out
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   TEST  foo
1  1.0 2013-01-02  1.0  3  TRAIN  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo
```
##### 3.通过字典和Series构造数据
```python
df = {'Name': pd.Series(['Jon', 'Aaron', 'Todd'], index=['a', 'b', 'c']),
      'Age': pd.Series(['39', '34', '32', '33'], index=['a', 'b', 'c','d']),
      'Nationality': pd.Series(['US', 'USA', 'China'], ['a', 'b', 'c'])}
df1 = pd.DataFrame(df)

#out
	Name	Age	Nationality
a	Jon	39	US
b	Aaron	34	USA
c	Todd	32	China
d	NaN	33	NaN
```
上面表中的每一列基本上就是一个 Series ，它们都用了同一个 index。因此，我们基本上可以把 DataFrame 理解成一组采用同样索引的 Series 的集合。

### 查看数据
以一下DataFrames为例
```python
dates = pd.date_range('20190425', periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates, columns=['a', 'b', 'c','d'])
#out
                   a         b         c         d
2019-04-25 -1.166267  0.435387  1.410483  0.052759
2019-04-26  0.937992 -0.480694  0.318787 -0.449206
2019-04-27 -0.733083 -0.982927 -0.966134  0.422478
2019-04-28  0.480559  0.466605  1.753447  0.324091
2019-04-29  1.024889 -0.962393  0.908883  0.676290
2019-04-30  0.211393  0.897806  0.129817 -2.332275

```python
data = {'Name': pd.Series(['Jon', 'Aaron', 'Todd'], index=['a', 'b', 'c']),
      'Age': pd.Series(['39', '34', '32', '33'], index=['a', 'b', 'c','d']),
      'Nationality': pd.Series(['US', 'USA', 'China'], ['a', 'b', 'c'])}
df3 = pd.DataFrame(data)

#out
	Name	Age	Nationality
a	Jon	39	US
b	Aaron	34	USA
c	Todd	32	China
d	NaN	33	NaN
```
##### index
pd.index

```python
df2.index

#out
DatetimeIndex(['2019-04-25', '2019-04-26', '2019-04-27', '2019-04-28',
               '2019-04-29', '2019-04-30'],
              dtype='datetime64[ns]', freq='D')
```

##### Columns
df2.columns
```python
df2.columns

#out
Index(['a', 'b', 'c', 'd'], dtype='object')
```
##### values
df2.values
```python
df2.values

#out
array([[-0.27965023, -1.23736234,  1.01619884, -0.0467824 ],
       [ 1.36451631, -1.70382268,  1.75504341,  1.19869696],
       [-0.24890785,  0.56749002, -2.37075569, -0.45060636],
       [ 0.534585  , -0.84531519,  0.49276816, -0.44888891],
       [ 1.15828998, -1.45520638, -2.24446852, -1.17805232],
       [-1.74294812, -1.70252682,  0.23831575,  0.59816592]])
```

##### describe
df.describe()
df.count() #非空元素计算
df.min() #最小值
df.max() #最大值
df.idxmin() #最小值的位置，类似于R中的which.min函数
df.idxmax() #最大值的位置，类似于R中的which.max函数
df.quantile(0.1) #10%分位数
df.sum() #求和
df.mean() #均值
df.median() #中位数
df.mode() #众数
df.var() #方差
df.std() #标准差
df.mad() #平均绝对偏差
df.skew() #偏度
df.kurt() #峰度
df.describe() #一次性输出多个描述性统计指标
```python
df.describe()

#out
	a	b	c	d
count	6.000000	6.000000	6.000000	6.000000
mean	0.130981	-1.062791	-0.185483	-0.054578
std	1.145811	0.861366	1.724190	0.845145
min	-1.742948	-1.703823	-2.370756	-1.178052
25%	-0.271965	-1.640697	-1.623772	-0.450177
50%	0.142839	-1.346284	0.365542	-0.247836
75%	1.002364	-0.943327	0.885341	0.436929
max	1.364516	0.567490	1.755043	1.198697
```
##### T
df.T table行装列
```python
df2.columns

#out
	2019-04-25 00:00:00	2019-04-26 00:00:00	2019-04-27 00:00:00	2019-04-28 00:00:00	2019-04-29 00:00:00	2019-04-30 00:00:00
a	-0.279650	1.364516	-0.248908	0.534585	1.158290	-1.742948
b	-1.237362	-1.703823	0.567490	-0.845315	-1.455206	-1.702527
c	1.016199	1.755043	-2.370756	0.492768	-2.244469	0.238316
d	-0.046782	1.198697	-0.450606	-0.448889	-1.178052	0.598166
```
##### sort_index
sort_index，对索引进行排序，默认是行索引排序axis=0.如果要对列索引排序axis=1
```python
s = pd.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, 4])
df = pd.DataFrame(s)

#out
	0
3	a
2	b
1	c
4	d

df.sort_index()

#out

0
1	c
2	b
3	a
4	d

df1 = pd.DataFrame(np.random.randn(2,3), columns=['b', 'a','c'])

#out
	b	a	c
0	-1.826025	-1.351159	2.421328
1	0.145775	0.607800	0.447299

df1.sourt_index(axis=1)

#out
	a	b	c
0	-1.351159	-1.826025	2.421328
1	0.607800	0.145775	0.447299
```

##### sort_values
默认对行上数据排序axis=0,要对列上数据排序需要axis=1
```python
df = pd.DataFrame(np.random.randn(2,3), columns=['b', 'a','c'])

#out
b	a	c
0	-1.826025	-1.351159	2.421328
1	0.145775	0.607800	0.447299

df.sort_values(by='b')

#out
b	a	c
0	-1.826025	-1.351159	2.421328
1	0.145775	0.607800	0.447299

df.sort_values(by=1, axis=1)

#out
b	a	c
0	-1.826025	-1.351159	2.421328
1	0.145775	0.607800	0.447299
```

##### 查看最上面几行
df.head()
```python
df = pd.DataFrame(np.random.randn(2,3), columns=['b', 'a','c'])
#out
b	a	c
0	-1.826025	-1.351159	2.421328
1	0.145775	0.607800	0.447299

df.head(1)
#out
b	a	c
0	-1.826025	-1.351159	2.421328
```
##### 查看最底下几行
df.tail()
```python
df = pd.DataFrame(np.random.randn(2,3), columns=['b', 'a','c'])
#out
b	a	c
0	-1.826025	-1.351159	2.421328
1	0.145775	0.607800	0.447299

df.tail(1)
#out
b	a	c
1	0.145775	0.607800	0.447299
```
##### 获取列内容
df.<Column_name> 或者 df.[<Column_name>], 获取单独的一列其实是获取一个Seires.
获取多列数据时，df[[<Column_name>,<Column_name>]]
```python
df = pd.DataFrame(np.random.randn(2,3), columns=['b', 'a','c'])
#out
b	a	c
0	-1.826025	-1.351159	2.421328
1	0.145775	0.607800	0.447299

df.a or df['a']
#out
0   -1.351159
1    0.607800
Name: a, dtype: float64

df[['a', 'b']]

#out
a	b
0	0.705811	-0.662005
1	-0.439263	-0.007571
2	-1.557856	0.653185
```

##### 获取行内容
通过[]获取行数据.输入内容为行index的切片，不可以是单独的行index。
```python
df = pd.DataFrame(np.random.randn(3,3), columns=['b', 'a','c'])
#out

b	a	c
0	-0.662005	0.705811	1.550029
1	-0.007571	-0.439263	0.910821
2	0.653185	-1.557856	2.209643

df[:1]

#out
b	a	c
0	-0.662005	0.705811	1.550029

df[0:3]

#out
b	a	c
0	-0.662005	0.705811	1.550029
1	-0.007571	-0.439263	0.910821
2	0.653185	-1.557856	2.209643

df[0]

#out
KeyError: 0
```
