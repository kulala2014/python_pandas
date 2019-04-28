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

####Create DataFrames
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


