import numpy as np
import pandas as pd
pd.__version__

pd.<TAB>
pd.read<TAB>

# Pandas Series is a one-dimensional array of indexed data

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
data.values
data.index
data[1]
data[1:3]

# can explicitly define index for Series object
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data
data['b']

#Series directly from dictionary

population_dict = {'California': 38332521,
'Texas': 26448193,
'New York': 19651127,
'Florida': 19552860,
'Illinois': 12882135}
population = pd.Series(population_dict)
population
population['California']
population['California':'Illinois']

#can behave how you want through key argument
pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])


#DataFrame
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area
states = pd.DataFrame({'population': population,
'area': area})
states
states.index
states.columns

states['area']

pd.DataFrame(population, columns=['population'])

data = [{'a': i, 'b': 2 * i} for i in range(3)]
pd.DataFrame(data)

pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])

#dictionary of Series objects
pd.DataFrame({'population': population, 'area': area})

#from a 2-d NumPy array
pd.DataFrame(np.random.rand(3, 2),
columns=['foo', 'bar'],
index=['a', 'b', 'c'])

#Index as immutable array or ordered set
ind = pd.Index([2, 3, 5, 7, 11])
ind
print(ind.size, ind.shape, ind.ndim, ind.dtype)
ind[1] = 0 #immutable

indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB #intersection
indA | indB #union
indA ^ indB #symmetric difference


#data selection of Series
data = pd.Series([0.25, 0.5, 0.75, 1.0],
index=['a', 'b', 'c', 'd'])
data
data['b']
'a' in data
data.keys()
list(data.items())

data['e'] = 1.25
data
data['a':'c']
data[0:2]
data[(data > 0.3) & (data < 0.8)]
data[['a', 'e']]

data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data
data[1]
data[1:3] #notice when slicing, behavior is typical (not based on keys)

#reference the explicit index
data.loc[1]
data.loc[1:3]

#reference the implicit index
data.iloc[1]
data.iloc[1:3]


#data selection of DataFrame
area = pd.Series({'California': 423967, 'Texas': 695662,
'New York': 141297, 'Florida': 170312,
'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
'New York': 19651127, 'Florida': 19552860,
'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data

data['area']
data.area

data.area is data['area']
data.pop is data['pop'] #data.pop is a bound method

data['density'] = data['pop'] / data['area']
data

data.values

data.T
data.values[0]
#vs
data['area']

data.iloc[:3, :2]
data.loc[:'Illinois', :'pop']
#hybrid (deprecated)
data.ix[:3, :'pop']

data.loc[data.density > 100, ['pop', 'density']]

#any indexing convention can be used to set values
data.iloc[0, 2] = 90
data

#indexing refers to columns, slicing refers to rows
data['Florida':'Illinois']

#direct masking operations are also interpreted row-wise
data[data.density > 100]


#Ufuncs index preservation
df = pd.DataFrame(np.random.randint(0, 10, (3, 4)),
columns=['A', 'B', 'C', 'D'])
df

#indices preserved
np.exp(df)
np.sin(df * np.pi / 4)

#index alignment
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
'New York': 19651127}, name='population')
population / area

area.index | population.index

A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B

A.add(B)
A.add(B, fill_value = 0)

A = pd.DataFrame(np.random.randint(0, 20, (2, 2)),
columns=list('AB'))
A
B = pd.DataFrame(np.random.randint(0, 10, (3, 3)),
columns=list('BAC'))
B
A + B

fill = A.stack().mean()
A.add(B, fill_value=fill)

#operations
A = np.random.randint(10, size=(3, 4))
A
A - A[0]

#operations rowwise by default
df = pd.DataFrame(A, columns=list('QRST'))
df - df.iloc[0]
#columnwise
df.subtract(df['R'], axis=0)

halfrow = df.iloc[0, ::2]
halfrow

df - halfrow

vals1 = np.array([1, None, 3, 4])
vals1

for dtype in ['object', 'int']:
    print("dtype =", dtype)
    %timeit np.arange(1E6, dtype=dtype).sum()
    print()
    
vals1.sum() #unsupported

vals2 = np.array([1, np.nan, 3, 4])
vals2.dtype

1 + np.nan
0 * np.nan

vals2.sum(), vals2.min(), vals2.max()

np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)

#upcasting
pd.Series([1, np.nan, 2, None])

x = pd.Series(range(2), dtype=int)
x

x[0] = None
x

data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
#masking
data[data.notnull()]

data.dropna()
df = pd.DataFrame([[1, np.nan, 2],
[2, 3, 5],
[np.nan, 4, 6]])
df

df.dropna()
df.dropna(axis='columns')
df[3] = np.nan
df
df.dropna(axis='columns', how='all')
df.dropna(axis='rows', thresh=3) #needs to be at least 3 non NA
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
data.fillna(0)

#forward-fill to propogate previous values forward
data.fillna(method='ffill')

#back-fill
data.fillna(method='bfill')

#along rows
df.fillna(method='ffill', axis=1)
df.fillna(method='bfill', axis=1)


#hierarchical indexing
index = [('California', 2000), ('California', 2010),
('New York', 2000), ('New York', 2010),
('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
18976457, 19378102,
20851820, 25145561]
pop = pd.Series(populations, index=index)
pop

pop[('California', 2010):('Texas', 2000)]

pop[[i for i in pop.index if i[1] == 2010]] #not as efficient as slicing

#Pandas MultiIndex
index = pd.MultiIndex.from_tuples(index)
index

pop = pop.reindex(index)
pop

pop[:, 2010] #much more efficient

pop_df = pop.unstack()
pop_df

pop_df.stack()

pop_df = pd.DataFrame({'total': pop,
'under18': [9267089, 9284094,
4687374, 4318033,
5906301, 6879014]})
pop_df

f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()

#ways to create hierarchical indexing
df = pd.DataFrame(np.random.rand(4, 2),
index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
columns=['data1', 'data2'])
df

data = {('California', 2000): 33871648,
('California', 2010): 37253956,
('Texas', 2000): 20851820,
('Texas', 2010): 25145561,
('New York', 2000): 18976457,
('New York', 2010): 19378102}
pd.Series(data)

#from simple list of arrays
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
#from tuples
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
#from Cartesian product
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
#from levels and labels
pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
labels=[[0, 0, 1, 1], [0, 1, 0, 1]])

#MultiIndex level names
pop.index.names = ['state', 'year']
pop

# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
names=['subject', 'type'])
# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data

health_data['Guido']
health_data.loc[2013]

#indexing and slicing MultiIndex
#Series
pop
pop['California', 2000]
pop['California']
pop.loc['California':'New York']
#lower levels
pop[:, 2000]
#boolean masks
pop[pop > 22000000]
#fancy indexing
pop[['California', 'Texas']]

#DataFrames
health_data
health_data['Guido', 'HR']
health_data.iloc[:2, :2]
#tuple of multiple indices
health_data.loc[:, ('Bob', 'HR')]
#IndexSlice object
idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, 'HR']]

#Rearranging MultiIndices
#slicing will fail if index is not sorted
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data
data['a':'b']

data = data.sort_index()
data
data['a':'b']

pop.unstack(level=0)
pop.unstack(level=1)
pop.unstack().stack()

#turn index into columns
pop_flat = pop.reset_index(name='population')
pop_flat
#turn columns into index
pop_flat.set_index(['state', 'year'])

#data aggregations
health_data
#average out measurements in two visits each year
data_mean = health_data.mean(level='year')
data_mean
#mean along levels of the columns
data_mean.mean(axis=1, level='type')

#combining datasets
#function for this example
def make_df(cols, ind):
    """Quickly make a DataFrame"""
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)

# example DataFrame
make_df('ABC', range(3))

x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
np.concatenate([x, y, z])

x = [[1, 2],
[3, 4]]
np.concatenate([x, x], axis=1)

#dataframes
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2])

df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print(df1); print(df2); print(pd.concat([df1, df2])) #row-wise by default

df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
print(df3); print(df4); print(pd.concat([df3, df4], axis=1))

#pd.concat preserves indices
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index # make duplicate indices!
print(x); print(y); print(pd.concat([x, y]))
#if undesirable, use verify_integrity
pd.concat([x, y], verify_integrity=True)
#ignore index
print(x); print(y); print(pd.concat([x, y], ignore_index=True))
#adding multiindex keys
print(x); print(y); print(pd.concat([x, y], keys=['x', 'y']))
#concatenation with joins
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
print(df5); print(df6); print(pd.concat([df5, df6]))
#inner join
print(df5); print(df6);
print(pd.concat([df5, df6], join='inner'))
#returned columns same as first input
print(df5); print(df6);
print(pd.concat([df5, df6], join_axes=[df5.columns]))


#merge and join
#one-to-one joins
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
'hire_date': [2004, 2008, 2012, 2014]})
print(df1); print(df2)
df3 = pd.merge(df1, df2)
df3
#many-to-one, preserve duplicates as appropriate
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
'supervisor': ['Carly', 'Guido', 'Steve']})
print(df3); print(df4); print(pd.merge(df3, df4))
#many-to-many
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
'Engineering', 'Engineering', 'HR', 'HR'], 'skills': ['math', 'spreadsheets', 'coding', 'linux',
'spreadsheets', 'organization']})
print(df1); print(df5); print(pd.merge(df1, df5))
#using the on keyword
print(df1); print(df2); print(pd.merge(df1, df2, on='employee'))
#left_on and right_on
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
'salary': [70000, 80000, 120000, 90000]})
print(df1); print(df3);
print(pd.merge(df1, df3, left_on="employee", right_on="name"))
#drop the column to avoid duplicate columns
pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)

#left_index and right_index
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(df1a); print(df2a)
print(pd.merge(df1a, df2a, left_index=True, right_index=True))
#join method merges joining on indices by default
print(df1a); print(df2a); print(df1a.join(df2a))
#combine methods
print(df1a); print(df3);
print(pd.merge(df1a, df3, left_index=True, right_on='name'))
#inner join by default
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
'food': ['fish', 'beans', 'bread']},
columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
'drink': ['wine', 'beer']},
columns=['name', 'drink'])
print(df6); print(df7); print(pd.merge(df6, df7))
#specify outer
print(df6); print(df7); print(pd.merge(df6, df7, how='outer'))
#left or right join
print(df6); print(df7); print(pd.merge(df6, df7, how='left'))
#overlapping column names
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'], 'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
'rank': [3, 1, 4, 2]})
print(df8); print(df9); print(pd.merge(df8, df9, on="name"))
#specify a suffix
print(df8); print(df9);
print(pd.merge(df8, df9, on="name", suffixes=["_L", "_R"]))

#example - US states
path = '~/KocPython2020/in-classMaterial/day8/'
pop = pd.read_csv(path + 'state-population.csv')
areas = pd.read_csv(path + 'state-areas.csv')
abbrevs = pd.read_csv(path + 'state-abbrevs.csv')
merged = pd.merge(pop, abbrevs, how='outer',
left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', 1) # drop duplicate info
merged.head()
#check for missingness
merged.isnull().any()
merged[merged['population'].isnull()].head()
merged.loc[merged['state'].isnull(), 'state/region'].unique()
#fix the issue
merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()
final = pd.merge(merged, areas, on='state', how='left')
final.head()
final.isnull().any()
final['state'][final['area (sq. mi)'].isnull()].unique()
#drop missing
final.dropna(inplace=True)
final.head()
#filter to 2010 and ages to total
data2010 = final.query("year == 2010 & ages == 'total'")
data2010.head()
#compute the density
data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']
#sort it
density.sort_values(ascending=False, inplace=True)
density.head()
density.tail()

#aggregation and grouping - exoplanets example
import seaborn as sns
planets = sns.load_dataset('planets')
planets.shape
planets.head()

rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
ser
ser.sum()
ser.mean()
df = pd.DataFrame({'A': rng.rand(5),
'B': rng.rand(5)})
df
df.mean() #by column
df.mean(axis='columns') #by row

planets.dropna().describe()

#split, apply, combine
# The split step involves breaking up and grouping a DataFrame depending on the
#value of the specified key.
# The apply step involves computing some function, usually an aggregate, transfor‐
#mation, or filtering, within the individual groups.
# The combine step merges the results of these operations into an output array.

df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
'data': range(6)}, columns=['key', 'data'])
df
df.groupby('key')
df.groupby('key').sum()

planets.groupby('method')
planets.groupby('method')['orbital_period']
planets.groupby('method')['orbital_period'].median()
planets.groupby('method')['year'].describe()

#aggregate, filter, transform, apply
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
'data1': range(6),
'data2': rng.randint(0, 10, 6)},
columns = ['key', 'data1', 'data2'])
df
#aggreagate can take string, function, list
df.groupby('key').aggregate(['min', np.median, max])
df.groupby('key').aggregate({'data1': 'min',
'data2': 'max'})
#filtering with boolean function
def filter_func(x):
    return x['data2'].std() > 4

print(df); print(df.groupby('key').std());
print(df.groupby('key').filter(filter_func)) #A does not have std greater than 4, so dropped
#transformation
#center the data by key
df.groupby('key').transform(lambda x: x - x.mean())
#apply - takes a df and returns a Pandas object or scalar
def norm_by_data2(x):
    # x is a DataFrame of group values
    x['data1'] /= x['data2'].sum()
    return x

print(df); print(df.groupby('key').apply(norm_by_data2))

#specifying the split key
L = [0, 1, 0, 1, 2, 0]
print(df); print(df.groupby(L).sum())
print(df); print(df.groupby(df['key']).sum())
#by dictionary
df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
print(df2); print(df2.groupby(mapping).sum())
#any Python function
print(df2); print(df2.groupby(str.lower).mean())
#a list of valid keys
df2.groupby([str.lower, mapping]).mean()
#grouping example
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)

#pivot tables - GroupBy but split and combine across 2-D grid
titanic = sns.load_dataset('titanic')
titanic.head()
titanic.groupby('sex')[['survived']].mean()
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
#instead
titanic.pivot_table('survived', index='sex', columns='class')

age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', ['sex', age], 'class') #notice the NaNs

titanic.pivot_table(index='sex', columns='class',
aggfunc={'survived':sum, 'fare':'mean'})
#compute totals along each grouping
titanic.pivot_table('survived', index='sex', columns='class', margins=True)

#time series data
from datetime import datetime
datetime(year=2015, month=7, day=4)

from dateutil import parser
date = parser.parse("4th of July, 2015")
date
date.strftime('%A')

date = np.array('2015-07-04', dtype=np.datetime64)
date
date + np.arange(12)
np.datetime64('2015-07-04')
np.datetime64('2015-07-04 12:00')
np.datetime64('2015-07-04 12:59:59.50', 'ns')
np.datetime64('now')
date = pd.to_datetime("4th of July, 2015")
date
date.strftime('%A')
date + pd.to_timedelta(np.arange(12), 'D')
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
'2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)
data
data['2014-07-04':'2015-07-04']
data['2015']
dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
'2015-Jul-6', '07-07-2015', '20150708'])
dates
dates.to_period('D')
dates - dates[0]
pd.date_range('2015-07-03', '2015-07-10')
pd.date_range('2015-07-03', periods=8)
pd.date_range('2015-07-03', periods=8, freq='H')
pd.period_range('2015-07', periods=8, freq='M')
pd.timedelta_range(0, periods=10, freq='H')
pd.timedelta_range(0, periods=9, freq="2H30T")
from pandas.tseries.offsets import BDay
pd.date_range('2015-07-01', periods=5, freq=BDay())

#financial data
from pandas_datareader import data
goog = data.DataReader('GOOG', start='2004', end='2016',
data_source='yahoo')
goog.head()
goog = goog['Close']

%matplotlib qt
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

goog.plot()

goog.plot(alpha=0.5, style='-')
goog.resample('BA').mean().plot(style=':')
goog.asfreq('BA').plot(style='--');
plt.legend(['input', 'resample', 'asfreq'],
loc='upper left');
#resample reports average of previous year, asfreq reports value at end of year

#upsample
fig, ax = plt.subplots(2, sharex=True)
data = goog.iloc[:10]
data.asfreq('D').plot(ax=ax[0], marker='o')
data.asfreq('D', method='bfill').plot(ax=ax[1], style='-o')
data.asfreq('D', method='ffill').plot(ax=ax[1], style='--o')
ax[1].legend(["back-fill", "forward-fill"]);


#time shifts
fig, ax = plt.subplots(3, sharey=True)
# apply a frequency to the data
goog = goog.asfreq('D', method='pad')
goog.plot(ax=ax[0])
goog.shift(900).plot(ax=ax[1])
goog.tshift(900).plot(ax=ax[2])
# legends and annotations
local_max = pd.to_datetime('2007-11-05')
offset = pd.Timedelta(900, 'D')
ax[0].legend(['input'], loc=2)
ax[0].get_xticklabels()[4].set(weight='heavy', color='red')
ax[0].axvline(local_max, alpha=0.3, color='red')
ax[1].legend(['shift(900)'], loc=2)
ax[1].get_xticklabels()[4].set(weight='heavy', color='red')
ax[1].axvline(local_max + offset, alpha=0.3, color='red')
ax[2].legend(['tshift(900)'], loc=2)
ax[2].get_xticklabels()[1].set(weight='heavy', color='red')
ax[2].axvline(local_max + offset, alpha=0.3, color='red');
#shift() shifts the data, tshift() shifts the index values

#computing differences over time
ROI = 100 * (goog.tshift(-365) / goog - 1)
ROI.plot()
plt.ylabel('% Return on Investment');

#rolling windows
rolling = goog.rolling(365, center=True)
data = pd.DataFrame({'input': goog,
'one-year rolling_mean': rolling.mean(),
'one-year rolling_std': rolling.std()})
ax = data.plot(style=['-', '--', ':'])
ax.lines[0].set_alpha(0.3)


#query and evals - can save memory and time for large arrays / data sets
import numexpr
nrows, ncols = 100000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols)) for i in range(4))
%timeit df1 + df2 + df3 + df4
%timeit pd.eval('df1 + df2 + df3 + df4')
df1, df2, df3, df4, df5 = (pd.DataFrame(rng.randint(0, 1000, (100, 3))) for i in range(5))
result1 = -df1 * df2 / (df3 + df4) - df5
result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')
np.allclose(result1, result2)

result1 = (df1 < df2) & (df2 <= df3) & (df3 != df4)
result2 = pd.eval('df1 < df2 <= df3 != df4')
np.allclose(result1, result2)

result1 = (df1 < 0.5) & (df2 < 0.5) | (df3 < df4)
result2 = pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')
np.allclose(result1, result2)
result3 = pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')
np.allclose(result1, result3)

result1 = df2.T[0] + df3.iloc[1]
result2 = pd.eval('df2.T[0] + df3.iloc[1]')
np.allclose(result1, result2)

df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
df.head()

result1 = (df['A'] + df['B']) / (df['C'] - 1)
result2 = pd.eval("(df.A + df.B) / (df.C - 1)")
np.allclose(result1, result2)
result3 = df.eval('(A + B) / (C - 1)')
np.allclose(result1, result3)

#can create new column
df.head()
df.eval('D = (A + B) / C', inplace=True)
df.head()
#can overwrite column
df.eval('D = (A - B) / C', inplace=True)
df.head()

#can access local variables
column_mean = df.mean(1)
result1 = df['A'] + column_mean
result2 = df.eval('A + @column_mean')
np.allclose(result1, result2)

#querying
result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
np.allclose(result1, result2)
result2 = df.query('A < 0.5 and B < 0.5')
np.allclose(result1, result2)

Cmean = df['C'].mean()
result1 = df[(df.A < Cmean) & (df.B < Cmean)]
result2 = df.query('A < @Cmean and B < @Cmean')
np.allclose(result1, result2)




