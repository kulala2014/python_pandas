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
##### 获取一行或多行数据
要获取某一行，你需要用 .loc[] 来按索引（标签名）引用这一行，或者用 .iloc[]，按这行在表中的位置（行数）来引用。
也可以用 .loc[] 来指定具体的行列范围，并生成一个子数据表，就像在 NumPy里做的一样。
也可以制定多行和/或多列。
at类似于loc，更快的访问速度，但是一次只能访问一个元素。
iat类似于iloc, 更快的访问速度，但是一次只能访问一个元素。
```python
df = pd.DataFrame(np.random.randn(3,3), columns=['b', 'a','c'], index=['row1', 'row2', 'row3'])
#out

	b	a	c
row1	0.680541	-0.485404	0.022410
row2	0.521382	1.008719	-0.473681
row3	-0.873195	0.771257	-0.973217

df.loc['row1']

#out
b   -0.662005
a    0.705811
c    1.550029
Name: 0, dtype: float64

df.loc['row1','a'] #获取第0行的a列

#out
0.7058105108405714

df.loc[['row1', 'row2'], ['a', 'b']]  or df.loc['row1':'row2', ['a','b']]# 获取第0行和第1行的a列和b列

#out
a	b
row1	-0.485404	0.680541
row2	1.008719	0.521382

df.loc[:,['a','b']] #获取所有的行的a列和b列

#out
a	b
row1	-0.485404	0.680541
row2	1.008719	0.521382
row3	0.771257	-0.873195


#----------------iloc get data with position-----------------------------------------------------
df.iloc[1] #获取第2行， 1是index

#out
b    0.521382
a    1.008719
c   -0.473681
Name: row2, dtype: float64

df.iloc[1:2,0:2] # 获取第2行的 0列和第一列

#out
b	a
row2	0.521382	1.008719

df.iloc[[1,2],[0,2]] #获取第二行和第三行的第一列和第三列

#out
b	c
row2	0.521382	-0.473681
row3	-0.873195	-0.973217

df.iloc[0:2,:] # 获取第1行和第二行的所有列

#out
b	a	c
row1	0.680541	-0.485404	0.022410
row2	0.521382	1.008719	-0.473681


df.iloc[1,1] # 获取第2行第二列的值

#out
1.0087189837346555

#-------------------------at-------------------------------------
df.at['row1', 'a'] # 获取row1行的a列的值
-0.4854038624550438

df.at['row1']

#out
TypeError: _get_value() missing 1 required positional argument: 'col'

#-------------------------iat-------------------------------------
df.iat[1,1] #获取第2列和第二行的值

#out
1.0087189837346555
```

##### 使用大于、等于、小于等布尔运算选择数据

```python
df = pd.DataFrame(np.random.randn(3,3), columns=['b', 'a','c'], index=['row1', 'row2', 'row3'])
#out

	b	a	c
row1	0.680541	-0.485404	0.022410
row2	0.521382	1.008719	-0.473681
row3	-0.873195	0.771257	-0.973217
```
###### 通过某一列的值选择数据
```python
df[df['a']>0] #获取所有a列数据大于0的行

#out
	b	a	c
row2	0.521382	1.008719	-0.473681
row3	-0.873195	0.771257	-0.973217
```
###### 从满足布尔条件的DataFrame中选择数据，不满足的地方用np.NaN替代
```python
df[df>0] #获取所有a列数据大于0的行

#out
b	a	c
row1	0.680541	NaN	0.02241
row2	0.521382	1.008719	NaN
row3	NaN	0.771257	NaN
```

###### 使用 isin() 方法过滤数据

```python
df2 = df.copy()
df2['d'] = ['one','two','three']
df2[df2['E'].isin(['two','four'])]

#out
b	a	c	d
row1	0.680541	-0.485404	0.022410	one
row2	0.521382	1.008719	-0.473681	two
row3	-0.873195	0.771257	-0.973217	three

df2[df2['d'].isin(['two','one'])] # 获取df2 中d列数据中包含 'two' 和'one'的行

#out
b	a	c	d
row1	0.680541	-0.485404	0.022410	one
row2	0.521382	1.008719	-0.473681	two
```

### 设置数据
```python
df = pd.DataFrame(np.random.randn(3,3), columns=['b', 'a','c'], index=['row1', 'row2', 'row3'])
#out

	b	a	c
row1	0.680541	-0.485404	0.022410
row2	0.521382	1.008719	-0.473681
row3	-0.873195	0.771257	-0.973217
```
###### 设置新列会自动根据索引对齐数据
```python
s1 = pd.Series([1,2,3], index=['row1','row2','row3'])
df['e'] = s1
df
#out

b	a	c	e
row1	0.680541	-0.485404	0.022410	1
row2	0.521382	1.008719	-0.473681	2
row3	-0.873195	0.771257	-0.973217	3
```
###### 使用at或loc根据行列名设置值
```python
df.at['row1','e'] = 4 or df.loc['row1','e'] = 4
df
#out

b	a	c	e
row1	0.680541	-0.485404	0.022410	4
row2	0.521382	1.008719	-0.473681	2
row3	-0.873195	0.771257	-0.973217	3
```
###### 使用iat或iloc根据行列名设置值
```python
df.iat[0,3] = 5 or df.iloc[0, 3] = 5
df
#out

b	a	c	e
row1	0.680541	-0.485404	0.022410	5
row2	0.521382	1.008719	-0.473681	2
row3	-0.873195	0.771257	-0.973217	3
```
###### 使用loc根据设置多个值
```python
df.loc[:,'c'] = np.array([5] * len(df)) # 设置c列的所以值新Series
df
#out

b	a	c	e
row1	0.680541	-0.485404	5	5
row2	0.521382	1.008719	5	2
row3	-0.873195	0.771257	5	3
```

###### 使用where 设置值
```python
df2 = df.copy()
df2[df2>1] = 1 # 设置c列的所以值新Series
df
#out

b	a	c	e
row1	0.781701	0.277362	1.000000	1.0
row2	0.940658	0.444926	0.305101	1.0
row3	0.482586	1.000000	0.288509	1.0
```
###### 使用drop删除列或者行
默认情况下，drop删除的是行，axis=0;需要删除列，axis=1
```python
df2 = df.copy()
df3.drop('row1') # 删除row1行
df

#out
b	a	c	e
row2	0.940658	0.444926	0.305101	2
row3	0.482586	-1.909210	0.288509	3

df3.drop('e', axis=1) # 删除e列

#out
b	a	c
row2	0.940658	0.444926	0.305101
row3	0.482586	-1.909210	0.288509
```

### Missing Data
pandas主要使用值np.nan来表示缺失的数据。 默认情况下，它不包含在计算中。
Reindexing允许您更改/添加/删除指定轴上的索引。 这将返回数据的副本。

```python
df1 = df.reindex(index=['row1','row2','row3','row4'], columns=list(df.columns) + ['E'])
df1

#out
b	a	c	e	E
row1	0.781701	-0.277362	1.511954	1.0	NaN
row2	0.940658	0.444926	0.305101	2.0	NaN
row3	0.482586	-1.909210	0.288509	3.0	NaN
row4	NaN	NaN	NaN	NaN	NaN
```
#### 删除任何含NaN的行
dropna 默认删除行,axis=0, how={any, all}; 要想删除列需要设置axis=1
any:任何含有NaN行或列;all: 所有行或者列都是NaN
```python
df2 = df1.copy()
df2.dropna(how='any') #删除任何含NaN的行
#out
b	a	c	e	E
row1	0.781701	-0.277362	1.511954	1.0	1.0
row2	0.940658	0.444926	0.305101	2.0	1.0

df2.dropna(how='all') #删除所有值都是NaN的行
#out
b	a	c	e	E
row1	0.781701	-0.277362	1.511954	1.0	1.0
row2	0.940658	0.444926	0.305101	2.0	1.0
row3	0.482586	-1.909210	0.288509	3.0	NaN

df2.dropna(axis=1, how='all') #删除所有的值都是NaN的列

#out
b	a	c	e
row1	0.781701	-0.277362	1.511954	NaN
row2	0.940658	0.444926	0.305101	NaN
row3	0.482586	-1.909210	0.288509	3.0
row4	NaN	NaN	NaN	NaN

df2.dropna(axis=1, how='any') #删除任何含有NaN的列

#out
b	a	c
row1	0.781701	-0.277362	1.511954
row2	0.940658	0.444926	0.305101
row3	0.482586	-1.909210	0.288509
row4	1.000000	1.000000	1.000000

```

#### 填充NaN
```python
df1 = df.reindex(index=['row1','row2','row3','row4'], columns=list(df.columns) + ['E'])
df1

#out
b	a	c	e	E
row1	0.781701	-0.277362	1.511954	1.0	NaN
row2	0.940658	0.444926	0.305101	2.0	NaN
row3	0.482586	-1.909210	0.288509	3.0	NaN
row4	NaN	NaN	NaN	NaN	NaN

df2.fillna(value=5) #replace NaN with 5

#out
	b	a	c	e	E
row1	0.781701	-0.277362	1.511954	5.0	5.0
row2	0.940658	0.444926	0.305101	5.0	5.0
row3	0.482586	-1.909210	0.288509	3.0	5.0
row4	1.000000	1.000000	1.000000	5.0	5.0
```

#### where语句NaN
获取NaN的值，如果为NaN, True; else: False
```python
df1 = df.reindex(index=['row1','row2','row3','row4'], columns=list(df.columns) + ['E'])
df1

#out
b	a	c	e	E
row1	0.781701	-0.277362	1.511954	1.0	NaN
row2	0.940658	0.444926	0.305101	2.0	NaN
row3	0.482586	-1.909210	0.288509	3.0	NaN
row4	NaN	NaN	NaN	NaN	NaN

pd.isna(df2) # NaN->True;Else->False

#out
	b	a	c	e	E
row1	False	False	False	True	True
row2	False	False	False	True	True
row3	False	False	False	False	True
row4	False	False	False	True	True
```

## DataFrame Operations
操作通常排除丢失的数据。
```python
df = pd.DataFrame(np.random.randn(3,3), columns=['b', 'a','c'], index=['row1', 'row2', 'row3'])
#out

	b	a	c
row1	0.680541	-0.485404	0.022410
row2	0.521382	1.008719	-0.473681
row3	-0.873195	0.771257	-0.973217
```
### 执行pd.describe 函数
#### 1.df.mean()
求df的列上的平均值，默认axis=0;如果要求行上的平均值，axis=1
```python
df.mean() #求列的平均值

#out
b    0.734981
a   -0.580549
c    0.701855
e    2.000000
dtype: float64

df.mean(1) #求行的平均值

#out
row1    0.754073
row2    0.922671
row3    0.465471
dtype: float64
```
#### 2.df.sub()
对应index相减。
shift默认向下移动value,用nan补充新的位置。
```python
s = pd.Series([1,2,3], index=['row1','row2','row3'])

#out
row1    1
row2    2
row3    3
dtype: int64

s = s.shift(2) #s的value向下移动

#out
row1    NaN
row2    NaN
row3    1.0
dtype: float64

df.sub(s, axis='index')

#out
b	a	c	e
row1	NaN	NaN	NaN	NaN
row2	NaN	NaN	NaN	NaN
row3	-0.517414	-2.90921	-0.711491	2.0
```
#### Apply
Apply functions to data:

```python
b	a	c	e	E
row1	0.781701	-0.277362	1.511954	1.0	1.0
row2	0.940658	0.444926	0.305101	2.0	1.0
row3	0.482586	-1.909210	0.288509	3.0	0.0
row4	0.000000	0.000000	0.000000	0.0	0.0


df.apply(np.cumsum) #np.cumsum 累加

#out
b	a	c	e	E
row1	0.781701	-0.277362	1.511954	1.0	1.0
row2	1.722359	0.167563	1.817055	3.0	2.0
row3	2.204944	-1.741647	2.105564	6.0	2.0
row4	2.204944	-1.741647	2.105564	6.0	2.0

df1.apply(lambda x: x.max() - x.min())

#out
b    0.940658
a    2.354136
c    1.511954
e    3.000000
E    1.000000
dtype: float64
```

### Histogramming statistics
直方图数据统计
```python
s = pd.Series(np.random.randint(0, 7, size=10))

#out
0    4
1    2
2    1
3    2
4    6
5    4
6    4
7    6
8    4
9    4

s.value_counts() #统计各个value出现的次数
#out
4    5
6    2
2    2
1    1
dtype: int64
```

### 字符串函数
dataFrame的值转为str后可以使用几乎所有的python字符串函数。
```python
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])

s.str.lower()

#out
0       a
1       b
2       c
3    aaba
4    baca
5     NaN
6    caba
7     dog
8     cat
dtype: object
```


## DataFrame Merge

### Concat
cancat 默认合并行数据，队列名合并，index保持不变。若要合并列，需要指定axis=1。若要重新生成index，ignore_indx=True
```python
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])

#out
a	b	c	d
0	0.0	0.0	0.0	0.0
1	0.0	0.0	0.0	0.0
2	0.0	0.0	0.0	0.0

df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])

#out
a	b	c	d
0	1.0	1.0	1.0	1.0
1	1.0	1.0	1.0	1.0
2	1.0	1.0	1.0	1.0

df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])

#out
a	b	c	d
0	2.0	2.0	2.0	2.0
1	2.0	2.0	2.0	2.0
2	2.0	2.0	2.0	2.0

res = pd.concat([df1, df2, df3]) # merge df1,df2,df3

#out
a	b	c	d
0	0.0	0.0	0.0	0.0
1	0.0	0.0	0.0	0.0
2	0.0	0.0	0.0	0.0
0	1.0	1.0	1.0	1.0
1	1.0	1.0	1.0	1.0
2	1.0	1.0	1.0	1.0
0	2.0	2.0	2.0	2.0
1	2.0	2.0	2.0	2.0
2	2.0	2.0	2.0	2.0

res2 = pd.concat([df1, df2, df3], ignore_index=True) #重新生成index

#out
a	b	c	d
0	0.0	0.0	0.0	0.0
1	0.0	0.0	0.0	0.0
2	0.0	0.0	0.0	0.0
3	1.0	1.0	1.0	1.0
4	1.0	1.0	1.0	1.0
5	1.0	1.0	1.0	1.0
6	2.0	2.0	2.0	2.0
7	2.0	2.0	2.0	2.0
8	2.0	2.0	2.0	2.0

res3 = pd.concat([df1, df2, df3], axis=1, ignore_index=True) #列合并，重新生成index

#out
	0	1	2	3	4	5	6	7	8	9	10	11
0	0.0	0.0	0.0	0.0	1.0	1.0	1.0	1.0	2.0	2.0	2.0	2.0
1	0.0	0.0	0.0	0.0	1.0	1.0	1.0	1.0	2.0	2.0	2.0	2.0
2	0.0	0.0	0.0	0.0	1.0	1.0	1.0	1.0	2.0	2.0	2.0	2.0

res3 = pd.concat([df1, df2, df3], axis=1) #列合并

#out
a	b	c	d	a	b	c	d	a	b	c	d
0	0.0	0.0	0.0	0.0	1.0	1.0	1.0	1.0	2.0	2.0	2.0	2.0
1	0.0	0.0	0.0	0.0	1.0	1.0	1.0	1.0	2.0	2.0	2.0	2.0
2	0.0	0.0	0.0	0.0	1.0	1.0	1.0	1.0	2.0	2.0	2.0	2.0
```

### Concat join
join包含： inner, outer。 默认情况下的join为outer。outer得意思就是把所有的列或行合起来。
inner只把index label或列名相同的数据合并。
```pyhton
Signature:
pd.concat(
    ['objs', 'axis=0', "join='outer'", 'join_axes=None', 'ignore_index=False', 'keys=None', 'levels=None', 'names=None', 'verify_integrity=False', 'sort=None', 'copy=True'],
```

```pyhton
df4 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])

#out
a	b	c	d
1	0.0	0.0	0.0	0.0
2	0.0	0.0	0.0	0.0
3	0.0	0.0	0.0	0.0

df5 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[1,2,3])

#out
b	c	d	e
1	1.0	1.0	1.0	1.0
2	1.0	1.0	1.0	1.0
3	1.0	1.0	1.0	1.0

res = pd.concat([df4, df5]) # default: outer join

#out
a	b	c	d	e
1	0.0	0.0	0.0	0.0	NaN
2	0.0	0.0	0.0	0.0	NaN
3	0.0	0.0	0.0	0.0	NaN
1	NaN	1.0	1.0	1.0	1.0
2	NaN	1.0	1.0	1.0	1.0
3	NaN	1.0	1.0	1.0	1.0

res = pd.concat([df4, df5], join='inner', ignore_index=True) #df4,df5 inner join的列名为bcd，所以只合并bcd

#out
b	c	d
0	0.0	0.0	0.0
1	0.0	0.0	0.0
2	0.0	0.0	0.0
3	1.0	1.0	1.0
4	1.0	1.0	1.0
5	1.0	1.0	1.0
```

join_axes 指定合并时使用那个df的index
```pyhton
df6 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])

#out
a	b	c	d
1	0.0	0.0	0.0	0.0
2	0.0	0.0	0.0	0.0
3	0.0	0.0	0.0	0.0

df7 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])

#out
b	c	d	e
2	1.0	1.0	1.0	1.0
3	1.0	1.0	1.0	1.0
4	1.0	1.0	1.0	1.0

res1 = pd.concat([df6, df7], axis=1, join_axes=[df7.index]) #合并列，行index以df7的为准

#out
a	b	c	d	b	c	d	e
2	0.0	0.0	0.0	0.0	1.0	1.0	1.0	1.0
3	0.0	0.0	0.0	0.0	1.0	1.0	1.0	1.0
4	NaN	NaN	NaN	NaN	1.0	1.0	1.0	1.0
```

### Merge
SQL样式合并。 

#### on=[column_name]指定按照合并的列名
```pyhton
#default值设置如下：
pd.merge(
    ['left', 'right', "how='inner'", 'on=None', 'left_on=None', 'right_on=None', 'left_index=False', 'right_index=False', 'sort=False', "suffixes=('_x', '_y')", 'copy=True', 'indicator=Fals
```
```pyhton
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})

right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

res =  pd.merge(left, right, on='key') #按照指定合并的列名

#out
 key  lval  rval
0  foo     1     4
1  foo     1     5
2  foo     2     4
3  foo     2     5

left1 = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
#out
   key  lval
0  foo     1
1  bar     2

right2 = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
#out
   key  rval
0  foo     4
1  bar     5

res = pd.merge(left1, right1, on='key')

#out
	key	lval	rval
0	foo	1	4
1	bar	2	5

```

### how指定merge的方式
合并方式how，可选值 how=['left','right','inner','outer']。默认选中inner
```pyhton
left1 = pd.DataFrame({'key1': ['K0', 'K0', 'k1', 'K2'],
                      'key2': ['K0', 'K1', 'k0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
#out
	key1	key2	A	B
0	K0	K0	A0	B0
1	K0	K1	A1	B1
2	k1	k0	A2	B2
3	K2	K1	A3	B3

right1 = pd.DataFrame({'key1': ['K0', 'K1', 'k1', 'K2'],
                      'key2': ['K0', 'K0', 'k0', 'K0'],
                     'C': ['C0', 'C1', 'C2', 'C3'],
                     'D': ['D0', 'D1', 'D2', 'D3']})
		     
#out
key1	key2	C	D
0	K0	K0	C0	D0
1	K1	K0	C1	D1
2	k1	k0	C2	D2
3	K2	K0	C3	D3

res1 = pd.merge(left1, right1, on=['key1', 'key2'], how='right') #以右边的df为基中，以key1,key2列为比较列

#out
key1	key2	A	B	C	D
0	K0	K0	A0	B0	C0	D0
1	k1	k0	A2	B2	C2	D2
2	K1	K0	NaN	NaN	C1	D1
3	K2	K0	NaN	NaN	C3	D3


df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
#out
	col1	col_left
0	0	a
1	1	b


df2 = pd.DataFrame({'col1':[1,2,3], 'col_right':[2,2,2]})
#out
col1	col_right
0	1	2
1	2	2
2	3	2

res = pd.merge(df1,df2, on='col1', how='left', indicator='indicator_col') # 以左边的df为基中，比较col1列，indicator显示合并的方式

#out
col1	col_left	col_right	indicator_col
0	0	a	NaN	left_only
1	1	b	2.0	both



left1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])
#out
	A	B
K0	A0	B0
K1	A1	B1
K2	A2	B2

right1 = pd.DataFrame({'C': ['C0', 'C2', 'C2'],
                     'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])
#out
C	D
K0	C0	D0
K2	C2	D2
K3	C2	D3

res = pd.merge(left1, right1, left_index=True, right_index=True, how='outer') #sql left join,比较左边，右边的index，相同的合并，没有的补充no.nan

#out
A	B	C	D
K0	A0	B0	C0	D0
K1	A1	B1	NaN	NaN
K2	A2	B2	C2	D2
K3	NaN	NaN	C2	D3
```
##### 使用 suffixes给相同列起别名
```pyhton
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'],
                     'age': [1, 2, 3]})
#out
	k	age
0	K0	1
1	K1	2
2	K2	3

girls = pd.DataFrame({'k': ['K0', 'K2', 'K3'],
                     'age': [4, 5, 6]})
#out
	k	age
0	K0	4
1	K2	5
2	K3	6

#按照k列比较，但是保持各自的age列，使用inner join方式
res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
#out
k	age_boy	age_girl
0	K0	1	4
1	K2	3	5
```

### Append
使用Append合并。

#### 合并Seirs到DataFrame
```pyhton
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
#out
a	b	c	d
0	0.0	0.0	0.0	0.0
1	0.0	0.0	0.0	0.0
2	0.0	0.0	0.0	0.0

s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
#out
a    1
b    2
c    3
d    4
dtype: int64

res = df1.append(s1, ignore_index=True)
#out
a	b	c	d
0	0.0	0.0	0.0	0.0
1	0.0	0.0	0.0	0.0
2	0.0	0.0	0.0	0.0
3	1.0	2.0	3.0	4.0
```

### 合并DataFrame
```pyhton
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df4 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['b','c','d','e'],index=[2,3,4])

res1 = df1.append([df2, df4], ignore_index=True) # df2,df4合并到df1中
#out
a	b	c	d
0	0.0	0.0	0.0	0.0
1	0.0	0.0	0.0	0.0
2	0.0	0.0	0.0	0.0
3	1.0	1.0	1.0	1.0
4	1.0	1.0	1.0	1.0
5	1.0	1.0	1.0	1.0
6	1.0	1.0	1.0	1.0
7	1.0	1.0	1.0	1.0
8	1.0	1.0	1.0	1.0

```

## 读取/存储数据
pandas支持各种各样的数据格式的输入和存储, read data to DataFrame。请参考官方：
```pyhton
https://pandas.pydata.org/pandas-docs/stable/reference/io.html#

df.to_csv('foo.csv')
df.to_hdf('foo.h5','df')
df.to_excel('foo.xlsx', sheet_name='Sheet1')
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
```

## Plotting 画图
可以调用matplotlib.pyplot画各种各样的图。 详细请参考matplotlib 官网。[https://matplotlib.org/index.html]
```pyhton
# plot data

#Series
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
data = data.cumsum()
data.plot()

#out
<matplotlib.axes._subplots.AxesSubplot at 0x283abf4c0f0>

#  dataFrame
data = pd.DataFrame(np.random.randn(1000,4),
                   index=np.arange(1000),
                    columns=list('ABCD'))
data = data.cumsum()
data.plot()
plt.show()	    
```
plt.methods:
bar, hist, box, kde, area, scatter, hexbin, pie

```pyhton
# plot data

#Series
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
data = data.cumsum()
data.plot()

#out
<matplotlib.axes._subplots.AxesSubplot at 0x283abf4c0f0>

#  dataFrame
data = pd.DataFrame(np.random.randn(1000,4),
                   index=np.arange(1000),
                    columns=list('ABCD'))
data = data.cumsum()

ax=data.plot.scatter(x='A',y='B', color='DarkBlue', label='Class1')
data.plot.scatter(x='A', y='C', color='DarkGreen', label='Class2', ax=ax)
plt.show()	    
```
## Time Series
pandas具有简单，强大且高效的功能，用于在频率转换期间执行重采样操作（例如，将第二数据转换为5分钟数据）。 这在财务应用程序中非常常见，但不仅限于此。

```pyhton
dates = pd.date_range('20190425', periods=6)
#out
DatetimeIndex(['2019-04-25', '2019-04-26', '2019-04-27', '2019-04-28',
               '2019-04-29', '2019-04-30'],
              dtype='datetime64[ns]', freq='D')

df['e'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20190425', periods=6))
#out
2019-04-25    1
2019-04-26    2
2019-04-27    3
2019-04-28    4
2019-04-29    5
2019-04-30    6
Freq: D, Name: e, dtype: int64

dates = pd.date_range('20190425', periods=6)

df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['a', 'b', 'c','d'])
#out
a	b	c	d
2019-04-25	0	1	2	3
2019-04-26	4	5	6	7
2019-04-27	8	9	10	11
2019-04-28	12	13	14	15
2019-04-29	16	17	18	19
2019-04-30	20	21	22	23
```

## Pivot Tables

