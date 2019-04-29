#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

s = pd.Series([1, 3, 5, np.nan, 44,1])
print(s)

dates = pd.date_range('20190425', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4),index=dates, columns=['a', 'b', 'c','d'])
print(df)

df1 = pd.DataFrame(np.arange(12).reshape((3, 4)))
print(df1)

df2 = pd.DataFrame({'A' : 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series( 1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3]*4, dtype='int32'),
                    'E': pd.Categorical(['TEST', 'TRAIN', 'test', 'train']),
                    'F': 'foo'})
print(df2)


# In[2]:





# In[45]:


df


# In[46]:


df.index


# In[48]:


df.columns


# In[49]:


df.values


# In[50]:


df.describe()


# In[51]:


df.T


# In[54]:


df


# In[58]:


df.sort_index(ascending=True)


# In[9]:


df2.sort_index(axis=0, ascending=False)


# In[11]:


df2.sort_values(by='E')


# In[12]:


df2


# In[13]:


df2.sort_values(by='E')


# In[15]:


df2.sort_values('E', axis=0)


# In[16]:


df2.A


# In[17]:


df2.B


# In[18]:


df2.bool


# In[20]:


df2.A.abs()


# In[21]:


df2.A.add(2)


# In[23]:


df2


# In[69]:


dates = pd.date_range('20190425', periods=6)
print(dates)

df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['a', 'b', 'c','d'])


# In[70]:


df


# In[33]:


df['a']


# In[35]:


df.a


# In[36]:


df[0:3]


# In[37]:


df['2019-04-25': '2019-04-27']


# In[39]:


# select by label: loc
df.loc['2019-04-25']


# In[40]:


df.loc[:,['a','b']]


# In[41]:


df.loc['2019-04-25',['a','b']]


# In[49]:


df


# In[48]:


# select by position: iloc
df.iloc[3,1]


# In[50]:


df.iloc[3:5, 1:3]


# In[51]:


df.iloc[[1,3,5], 1:3]


# In[52]:


# mixed selection: ix deprecated
df.ix[:3, ['a','c']]


# In[53]:


df


# In[55]:


df[df.a > 8]


# In[57]:


df[df.a<5]


# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[58]:


df


# In[59]:


df.iloc[2,2]


# In[66]:


df.iloc[2,2] = 111


# In[67]:


df


# In[62]:


df.loc['2019-04-25', 'b']


# In[68]:


df.loc['2019-04-25', 'b'] = 222


# In[77]:


df


# In[76]:


df.a[df.a > 4] = 0


# In[78]:


df


# In[79]:


df['f'] = np.nan


# In[80]:


df


# In[67]:


df['e'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20190425', periods=6))


# In[68]:


df['e']


# In[85]:


# process missing data
df.drop('f', axis=1)


# In[90]:





# In[87]:


df


# In[91]:


df


# In[93]:


df=df.drop('f', axis=1)


# In[94]:


df


# In[95]:


df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan


# In[96]:


df


# In[99]:


df = df.dropna(axis=0, how='all') # how={any,all}


# In[98]:


df


# In[100]:


df.dropna()


# In[101]:


df.dropna(axis=1)


# In[107]:


df=df.fillna(value=0)


# In[108]:


df.isnull()


# In[109]:


np.any(df.isnull())


# In[111]:


df.to_csv('test.csv')


# In[114]:


data = pd.read_csv('test.csv')


# In[115]:


data


# In[117]:


data.drop('Unnamed: 0',axis=1)


# In[119]:


data.to_pickle('data.pickle')


# In[120]:


data1 = pd.read_pickle('data.pickle')


# In[121]:


data1


# In[122]:


data.to_json('test.json')


# In[3]:


# contatenating

df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])


# In[4]:


df1


# In[128]:


df2


# In[129]:


df3


# In[5]:


res = pd.concat([df1, df2, df3])


# In[6]:


res


# In[15]:


res2 = pd.concat([df1, df2, df3], axis=1)


# In[16]:


res2


# In[136]:


# join, ['inner', 'outer']


# In[20]:


df4 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df5 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[1,2,3])


# In[22]:


df5


# In[23]:


res = pd.concat([df4, df5])


# In[149]:


res


# In[152]:


res = pd.concat([df4, df5], join='inner', ignore_index=True)
res


# In[24]:


df6 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df7 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])


# In[25]:


df6


# In[60]:


# append
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df4 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['b','c','d','e'],index=[2,3,4])


# In[61]:


df1


# In[27]:


res1 = pd.concat([df6, df7], axis=1, join_axes=[df7.index])


# In[28]:


res1


# In[63]:


s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])


# In[64]:


s1


# In[179]:


res = df1.append(s1, ignore_index=True)
res


# In[65]:


res1 = df1.append([df2, df4], ignore_index=True)
res1


# In[184]:


left = pd.DataFrame({'key': ['K0', 'K1', 'k2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})


# In[185]:


right = pd.DataFrame({'key': ['K0', 'K1', 'k2', 'K3'],
                     'C': ['C0', 'C1', 'C2', 'C3'],
                     'D': ['D0', 'D1', 'D2', 'D3']})


# In[186]:


left


# In[187]:


right


# In[188]:


res = pd.merge(left, right, on='key')


# In[189]:


res


# In[48]:


left1 = pd.DataFrame({'key1': ['K0', 'K0', 'k1', 'K2'],
                      'key2': ['K0', 'K1', 'k0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right1 = pd.DataFrame({'key1': ['K0', 'K1', 'k1', 'K2'],
                      'key2': ['K0', 'K0', 'k0', 'K0'],
                     'C': ['C0', 'C1', 'C2', 'C3'],
                     'D': ['D0', 'D1', 'D2', 'D3']})


# In[49]:


left1


# In[51]:


right1


# In[203]:


# how=['left','right','inner','outer']
res1 = pd.merge(left1, right1, on=['key1', 'key2'], how='right')


# In[204]:


res1


# In[205]:


df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,3], 'col_right':[2,2,2]})


# In[206]:


df1


# In[207]:


df2


# In[220]:


res = pd.merge(df1,df2, on='col1', how='left', indicator='indicator_col')


# In[221]:


res


# In[54]:


left1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])

right1 = pd.DataFrame({'C': ['C0', 'C2', 'C2'],
                     'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])


# In[55]:


left1


# In[56]:


right1


# In[57]:


res = pd.merge(left1, right1, left_index=True, right_index=True, how='outer')


# In[58]:


res


# In[233]:


boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'],
                     'age': [1, 2, 3]})

girls = pd.DataFrame({'k': ['K0', 'K2', 'K3'],
                     'age': [4, 5, 6]})


# In[234]:


boys


# In[235]:


girls


# In[236]:


res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')


# In[237]:


res


# In[248]:


import matplotlib.pyplot as plt


# In[249]:


# plot data

#Series
data = pd.Series(np.random.randn(1000),index=np.arange(1000))


# In[241]:


pd.read


# In[250]:


data = data.cumsum()


# In[251]:


data


# In[244]:


data.plot()


# In[254]:


plt.plot(data)


# In[256]:


#  dataFrame
data = pd.DataFrame(np.random.randn(1000,4),
                   index=np.arange(1000),
                    columns=list('ABCD'))
data


# In[257]:


data = data.cumsum()


# In[258]:


data


# In[263]:


data.plot()
plt.show()


# In[264]:


# plt.methods:
#  bar, hist, box, kde, area, scatter, hexbin, pie


# In[269]:


ax=data.plot.scatter(x='A',y='B', color='DarkBlue', label='Class1')
data.plot.scatter(x='A', y='C', color='DarkGreen', label='Class2', ax=ax)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


import pandas as pd


# In[4]:


s = pd.Series([1,2,3,4,5])


# In[5]:


s


# In[6]:


s[1]


# In[12]:


import numpy as np
data = [100, 200, 300, 400]
np_arr = np.array(data)
my_series = pd.Series(data)


# In[13]:


my_series


# In[14]:


my_dict = {'a': 100, 'b': 200, 'c':300, 'd':400}
my_series = pd.Series(my_dict)


# In[15]:


my_series


# In[16]:


index = ['USA', 'Nigeria', 'France', 'Ghana']
data = [100, 200, 300, 400]
my_series = pd.Series(data,index)


# In[17]:


my_series['USA']


# In[18]:


series1 = pd.Series([1,2,3,4], ['London','USA', 'France', 'HK'])
series2 = pd.Series([1,2,3,4], ['London','USA', 'JP', 'HK'])
res = series1 -series2


# In[19]:


res


# In[20]:


res1 = series1 + series2


# In[21]:


res1


# In[22]:


res3 = series1 * series2
res3


# In[23]:


res4 = series1 / series2


# In[24]:


res4


# In[66]:


dates = pd.date_range('20190425', periods=6)
print(dates)


# In[30]:


df = {'Name': pd.Series(['Jon', 'Aaron', 'Todd'], index=['a', 'b', 'c']),
      'Age': pd.Series(['39', '34', '32', '33'], index=['a', 'b', 'c','d']),
      'Nationality': pd.Series(['US', 'USA', 'China'], ['a', 'b', 'c'])}
df1 = pd.DataFrame(df)


# In[31]:


df1


# In[32]:


df2 = pd.DataFrame({'A' : 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series( 1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3]*4, dtype='int32'),
                    'E': pd.Categorical(['TEST', 'TRAIN', 'test', 'train']),
                    'F': 'foo'})


# In[36]:


df2


# In[38]:


df2.index


# In[40]:


df2[[0]]
# df['e']
# s4[[1,3,5]]
# s4[['a','b','d','f']]
# s4[:4]
# s4['c':]
# s4['b':'e']


# In[42]:


df2[1,]


# In[59]:


s = pd.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, 4])
df = pd.DataFrame(s)


# In[101]:


df = pd.DataFrame(np.random.randn(3,3), columns=['b', 'a','c'], index=['row1', 'row2', 'row3'])
df


# In[64]:


df.sort_index(axis=1)


# In[65]:


df.sort_values(by='b')


# In[70]:


df.sort_values(by=0,axis=1)


# In[73]:


a = df.bool


# In[74]:


print(a)


# In[76]:


df.head(1)


# In[78]:


df.tail()


# In[79]:


df.


# In[89]:


df[['a','b']]


# In[82]:


df.head()


# In[97]:


df.loc[0,'a']


# In[98]:


df.loc[[0, 1], ['a', 'b']]


# In[99]:


df.loc[:,['a','b']]


# In[100]:


df.reindex('row1', 'row2', 'row3')


# In[5]:


df = pd.DataFrame(np.random.randn(3,3), columns=['b', 'a','c'], index=['row1', 'row2', 'row3'])
df


# In[103]:


df.loc['row1':'row2', ['a', 'b']]


# In[107]:


df.loc['row1']


# In[108]:


df.loc[['row1', 'row2'], ['a', 'b']]


# In[110]:


df.loc['row1':'row2', ['a','b']]


# In[111]:


df.loc[:,['a','b']]


# In[113]:


df.iloc[1]


# In[114]:


df.iloc[1:2,0:2]


# In[116]:


df.loc['row1':'row2']


# In[117]:


df.iloc[[1,2],[0,2]]


# In[118]:


df.iloc[0:2,:]


# In[119]:


df.iloc[1,1]


# In[121]:


df.at['row1','a']


# In[122]:


df.at['row1']


# In[123]:


df.iat[1,1]


# In[124]:


df


# In[127]:


df[df>0]


# In[129]:


df2 = df.copy()
df2['d'] = ['one','two','three']


# In[130]:


df2


# In[131]:


df2[df2['d'].isin(['two','one'])]


# In[6]:


s1 = pd.Series([1,2,3], index=['row1','row2','row3'])


# In[7]:


df['e'] = s1


# In[8]:


df


# In[143]:


df.ioc['row1','e'] = 4


# In[142]:


df


# In[145]:


df.loc['row1','e'] = 4


# In[146]:


df


# In[147]:


df.iat[0,3] = 5


# In[148]:


df


# In[149]:


df.loc[:,'c'] = np.array([5] * len(df))


# In[150]:


df


# In[9]:


df2 = df.copy()


# In[10]:


df2


# In[11]:


df2[df2 >0] = -df2


# In[12]:


df2


# In[13]:


df2


# In[14]:


df2 = -df2


# In[20]:


df2


# In[17]:


df2[df2>1]


# In[21]:


df2 = -df2


# In[22]:


df2


# In[23]:


df2[df2>1] = 1


# In[24]:


df2


# In[25]:


df3 = df.copy()


# In[28]:


df3.drop('row1',inplace=True)


# In[30]:


df3.drop('e', axis=1)


# In[33]:


df1 = df.reindex(index=['row1','row2','row3','row4'], columns=list(df.columns) + ['E'])


# In[34]:


df1


# In[35]:


df1.loc['row1':'row2','E'] = 1


# In[36]:


df1


# In[48]:


df2 = df1.copy()


# In[49]:


df2


# In[39]:





# In[42]:


df2


# In[43]:


df2.dropna(axis=0, how='all') # how={any,all}


# In[45]:


df2


# In[46]:


df2.dropna(axis=1)


# In[53]:


df2.loc['row1':'row2','E'] = np.nan


# In[54]:


df2


# In[56]:


df2.dropna(axis=1, how='all')


# In[57]:


df2.at['row4',['a','b', 'c']]=1


# In[58]:


df2


# In[60]:


df2.dropna(axis=1, how='any')


# In[63]:


df2


# In[64]:


pd.isna(df2)


# In[65]:


df


# In[66]:


df.mean()


# In[67]:


df.mean(1)


# In[78]:


s = pd.Series([1,2,3], index=['row1','row2','row3']).shift(2)


# In[82]:


s = pd.Series([1,2,3], index=['row1','row2','row3'])


# In[83]:


s


# In[70]:


s.shift(2)


# In[77]:


s


# In[80]:


df.sub(s, axis='index')


# In[76]:


df


# In[84]:


np.cumsum


# In[85]:


df.apply(np.cumsum)


# In[86]:


df1


# In[89]:


df1.fillna(value=0, inplace=True)


# In[90]:


df1


# In[91]:


df1.apply(np.cumsum)


# In[92]:


df1.apply(lambda x: x.max() - x.min())


# In[93]:


s = pd.Series(np.random.randint(0, 7, size=10))


# In[94]:


s


# In[95]:


s.value_counts()


# In[43]:


left1 = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right1 = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})


# In[44]:


left1


# In[45]:


right1


# In[46]:


res = pd.merge(left1, right1, on='key')


# In[47]:


res


# In[71]:


df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                   'B' : ['A', 'B', 'C'] * 4,
                    'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D' : np.random.randn(12),
                    'E' : np.random.randn(12)})


# In[72]:


df


# In[73]:


pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])


# In[74]:


df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})


# In[75]:


df


# In[76]:


df["grade"] = df["raw_grade"].astype("category")


# In[77]:


df


# In[78]:


df["grade"]


# In[79]:


df["grade"].cat.categories = ["very good", "good", "very bad"]


# In[81]:


df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])


# In[82]:


df


# In[83]:


df.sort_values(by="grade")


# In[84]:


df.groupby("grade").size()


# In[ ]:




