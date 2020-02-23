#ipython --pylab

import numpy as np
np.__version__

np?
#http://www.numpy.org

#unlike Python lists, NumPy is constrained to arrays that all contain the same type
np.array([3.14, 4, 2, 3])
np.array([1, 2, 3, 4], dtype='float32')

#can be multidimensional - inner lists are treated as rows
np.array([range(i, i + 3) for i in [2, 4, 6]])

# Create a length-10 integer array filled with zeros
np.zeros(10, dtype=int)

# Create a 3x5 floating-point array filled with 1s
np.ones((3, 5), dtype=float)

# Create a 3x5 array filled with 3.14
np.full((3, 5), 3.14)

# Create an array filled with a linear sequence
# Starting at 0, ending at 20, stepping by 2
# (this is similar to the built-in range() function)
np.arange(0, 20, 2)

# Create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)

# Create a 3x3 array of uniformly distributed
# random values between 0 and 1
np.random.random((3, 3))

# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))

# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))

# Create a 3x3 identity matrix
np.eye(3)

# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that
# memory location
np.empty(3)

#data types

#bool_ Boolean (True or False) stored as a byte
#int_ Default integer type (same as C long ; normally either int64 or int32 )
#intc Identical to C int (normally int32 or int64 )
#intp Integer used for indexing (same as C ssize_t ; normally either int32 or int64 )
#int8 Byte (–128 to 127)
#int16 Integer (–32768 to 32767)
#int32 Integer (–2147483648 to 2147483647)
#int64 Integer (–9223372036854775808 to 9223372036854775807)
#uint8 Unsigned integer (0 to 255)
#uint16 Unsigned integer (0 to 65535)
#uint32 Unsigned integer (0 to 4294967295)
#uint64 Unsigned integer (0 to 18446744073709551615)
#float_ Shorthand for float64
#float16 Half-precision float: sign bit, 5 bits exponent, 10 bits mantissa
#float32 Single-precision float: sign bit, 8 bits exponent, 23 bits mantissa
#float64 Double-precision float: sign bit, 11 bits exponent, 52 bits mantissa
#complex_ Shorthand for complex128
#complex64 Complex number, represented by two 32-bit floats
#complex128 Complex number, represented by two 64-bit floats

np.random.seed(0) # seed for reproducibility (can be any number, just used to rerun)
x1 = np.random.randint(10, size=6) # One-dimensional array
x2 = np.random.randint(10, size=(3, 4)) # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5)) # Three-dimensional array

print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)

x2
x2[0,0]
x2[2,0]
x2[2,-1]

x2[0, 0] = 12
x2

x1[0] = 3.14159
x1

#slicing similar to Python standard
x = np.arange(10)
x
x[:5]
x[5:]
x[4:7]
x[::2]
x[1::2]
x[::-1]
x[5::-2]

#multidimensional slicing
x2
x2[:2, :3]
x2[:3, ::2]
x2[::-1, ::-1]

print(x2[:, 0]) # first column of x2
print(x2[0, :]) # first row of x2
print(x2[0]) # equivalent to x2[0, :]

#slices are not copies! they are views
print(x2)
x2_sub = x2[:2, :2]
print(x2_sub)
x2_sub[0, 0] = 99
print(x2_sub)
print(x2)

#to create copies, use .copy()
x2_sub_copy = x2[:2, :2].copy()
x2_sub_copy[0, 0] = 42
print(x2_sub_copy)
print(x2)

#reshaping
grid = np.arange(1, 10).reshape((3, 3))
print(grid)

x = np.array([1, 2, 3])
# row vector via reshape
x.reshape((1, 3))

# row vector via newaxis
x[np.newaxis, :]

# column vector via reshape
x.reshape((3, 1))

# column vector via newaxis
x[:, np.newaxis]

#concatenation

x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])
z = [99, 99, 99]
print(np.concatenate([x, y, z]))

grid = np.array([[1, 2, 3], [4, 5, 6]])
# concatenate along the first axis
np.concatenate([grid, grid])
# concatenate along the second axis (zero-indexed)
np.concatenate([grid, grid], axis=1)

x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
[6, 5, 4]])
# vertically stack the arrays
np.vstack([x, grid])
# horizontally stack the arrays
y = np.array([[99],
[99]])
np.hstack([grid, y])
#third axis
np.dstack([x3, x3])


#splitting
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)

grid = np.arange(16).reshape((4, 4))
grid
upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)
left, right = np.hsplit(grid, [2])
print(left)
print(right)
x3 = np.random.randint(10, size=(3, 4, 5))
np.dsplit(x3, [2])


#vectorize!
def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)

big_array = np.random.randint(1, 100, size=1000000)
%timeit compute_reciprocals(big_array)

print(compute_reciprocals(values))
print(1.0 / values)

%timeit (1.0 / big_array)

x = np.arange(9).reshape((3, 3))
2 ** x

np.arange(9) + np.arange(1,10)

x = np.array([-2, -1, 0, 1, 2])
abs(x)

x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
abs(x)

#trigonometric functions

theta = np.linspace(0, np.pi, 3) #array of angles
theta
np.sin(theta)
np.cos(theta)
np.tan(theta)
x = [-1, 0, 1]
np.arcsin(x)
np.arccos(x)
np.arctan(x)

#exponentials

x = [1, 2, 3]
print("x=", x)
print("e^x=", np.exp(x))
print("2^x=", np.exp2(x))
print("3^x=", np.power(3, x))

#logarithms

x = [1, 2, 4, 10]
np.log(x)
np.log2(x)
np.log10(x)

x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))

from scipy import special
# Gamma functions (generalized factorials) and related functions
x = [1, 5, 10]
print("gamma(x) =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2) =", special.beta(x, 2))

# Error function (integral of Gaussian)
# its complement, and its inverse
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x) =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))


#out argument can save time by allocation of memory
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)

y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)

#outer product
x = np.arange(1, 6)
np.multiply.outer(x, x)

#sum
big_array = np.random.rand(1000000)
%timeit sum(big_array)
%timeit np.sum(big_array)

#min and max
np.min(big_array), np.max(big_array)
%timeit min(big_array)
%timeit np.min(big_array)

print(big_array.min(), big_array.max(), big_array.sum())

M = np.random.random((3, 4))
print(M)

M.sum()
M.min(axis=0)
M.max(axis=1)
M.max()

#np.sum np.nansum Compute sum of elements
#np.prod np.nanprod Compute product of elements
#np.mean np.nanmean Compute median of elements
#np.std np.nanstd Compute standard deviation
#np.var np.nanvar Compute variance
#np.min np.nanmin Find minimum value
#np.max np.nanmax Find maximum value
#np.argmin np.nanargmin Find index of minimum value
#np.argmax np.nanargmax Find index of maximum value
#np.median np.nanmedian Compute median of elements
#np.percentile np.nanpercentile Compute rank-based statistics of elements
#np.any N/A Evaluate whether any elements are true
#np.all N/A Evaluate whether all elements are true

#presidential height
import os
os.chdir('KocPython2020/in-classMaterial/day7')
!head -4 president_heights.csv

import pandas as pd
data = pd.read_csv('president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)

print("Mean height:", heights.mean())
print("Standard deviation:", heights.std())
print("Minimum height:", heights.min())
print("Maximum height:", heights.max())

print("25th percentile:", np.percentile(heights, 25))
print("Median:", np.median(heights))
print("75th percentile:", np.percentile(heights, 75))


%matplotlib osx
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # set plot style
plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number');

#broadcasting
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b
a + 5
M = np.ones((3, 3))
M
M + a
a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a)
print(b)
a + b

#Rule 1: If the two arrays differ in their number of dimensions, the shape of the
#one with fewer dimensions is padded with ones on its leading (left) side.
a.shape
M.shape
a+M
#Rule 2: If the shape of the two arrays does not match in any dimension, the array
#with shape equal to 1 in that dimension is stretched to match the other shape.
#Rule 3: If in any dimension the sizes disagree and neither is equal to 1, an error is
#raised.

M = np.ones((2, 3))
a = np.arange(3)
M
a
M + a

a = np.arange(3).reshape((3, 1))
b = np.arange(3)
a + b

M = np.ones((3, 2))
a = np.arange(3)
M + a
M + a[:, np.newaxis] #right padding

#TODO: scale the mean of the following array by column (center and standardize)
X = np.random.random((10, 3))

# x and y have 50 steps from 0 to 5
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
plt.colorbar();


# use Pandas to extract rainfall inches as a NumPy array
rainfall = pd.read_csv('Seattle2014.csv')['PRCP'].values
inches = rainfall / 254 # 1/10mm -> inches
inches.shape
plt.hist(inches, 40);

# booleans
x = np.array([1, 2, 3, 4, 5])
x < 3 # less than
x > 3 # greater than
x <= 3
x == 3
x != 3
x >= 3
(2 * x) == (x ** 2)

rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))
x
x < 6
np.count_nonzero(x < 6)
np.sum(x < 6)
# how many values less than 6 in each row?
np.sum(x < 6, axis=1)
# are there any values greater than 8?
np.any(x > 8)
# are there any values less than zero?
np.any(x < 0)
# are all values less than 10?
np.all(x < 10)
# are all values equal to 6?
np.all(x == 6)
# are all values in each row less than 8?
np.all(x < 8, axis=1)

np.sum((inches > 0.5) & (inches < 1))
np.sum(~( (inches <= 0.5) | (inches >= 1) ))

print("Number days without rain:", np.sum(inches == 0))
print("Number days with rain:", np.sum(inches != 0))
print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
print("Rainy days with < 0.2 inches :", np.sum((inches > 0) & (inches < 0.2)))


x[x < 5]
# construct a mask of all rainy days
rainy = (inches > 0)
# construct a mask of all summer days (June 21st is the 172nd day)
summer = (np.arange(365) - 172 < 90) & (np.arange(365) - 172 > 0)
print("Median precip on rainy days in 2014 (inches):", np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches): ", np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches): ", np.max(inches[summer]))
print("Median precip on non-summer rainy days (inches):", np.median(inches[rainy & ~summer]))

rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
print(x)
[x[3], x[7], x[2]]
ind = [3, 7, 2]
x[ind]
ind = np.array([[3, 7], [4, 5]])
x[ind]

X = np.arange(12).reshape((3, 4))
X
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]
#broadcasting indices
X[row[:, np.newaxis], col]

X[2, [2, 0, 1]]
X[1:, [2, 0, 1]]
mask = np.array([1, 0, 1, 0], dtype=bool)
X[row[:, np.newaxis], mask]

mean = [0, 0]
cov = [[1, 2],
[2, 5]]
X = random.multivariate_normal(mean, cov, 100)
X.shape

plt.scatter(X[:, 0], X[:, 1]);

indices = np.random.choice(X.shape[0], 20, replace=False)
indices
selection = X[indices]
selection.shape
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1]);



x = np.zeros(10)
x[[0, 0]] = [4, 6]
print(x)

i = [2, 3, 3, 4, 4, 4]
x[i] += 1
x

x = np.zeros(10)
np.add.at(x, i, 1)
print(x)

x = np.array([2, 1, 4, 3, 5])
np.sort(x)

x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print(i)
x[i]

rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(X)

# sort each column of X
np.sort(X, axis=0)

# sort each row of X
np.sort(X, axis=1)


x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3)

np.partition(X, 2, axis=1)

#nearest neighbor

X = random.rand(10, 2)
plt.scatter(X[:, 0], X[:, 1], s=100);
dist_sq = np.sum((X[:,np.newaxis,:] - X[np.newaxis,:,:]) ** 2, axis=-1)
dist_sq
nearest = np.argsort(dist_sq, axis=1)
print(nearest)
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)

plt.scatter(X[:, 0], X[:, 1], s=100)
# draw lines from each point to its two nearest neighbors
K = 2
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]:
        # plot a line from X[i] to X[j]
        # use some zip magic to make it happen:
        plt.plot(*zip(X[j], X[i]), color='black')


name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
# Use a compound data type for structured arrays
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'), 'formats':('U10', 'i4', 'f8')}) #unicode, int, float
print(data.dtype)
data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)
# Get all names
data['name']
# Get first row of data
data[0]
# Get the name from the last row
data[-1]['name']
# Get names where age is under 30
data[data['age'] < 30]['name']

# transpose

a = np.array([1,2,3,4])
a.T
a.reshape(1,4).T

M = np.array(np.arange(16)).reshape(4,4)
M
M.T

# matrix multiplication

a @ M
M @ a
M @ a.reshape(4,1)
a @ M

# inverse

np.linalg.inv(M)

np.linalg.inv(M.T @ M) #singular

X = np.random.random((15, 3))
X.T @ X
np.linalg.inv(X.T @ X)

# linear regression

y = np.random.random((15,1))
b = np.linalg.inv(X.T @ X) @ X.T @ y



# empty filled with NaN

p = np.empty((4,4))
p
p.fill(np.nan)
p


#TODO: Answer the following questions (solutions: https://www.machinelearningplus.com/python/101-numpy-exercises-python/ continue on the site if you finish)

#Create a 3×3 numpy array of all True’s


#Extract all odd numbers from arr
#Input:

#arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#Desired output:

##> array([1, 3, 5, 7, 9])


#Replace all odd numbers in arr with -1


#Replace all odd numbers in arr with -1 without changing arr


#Convert a 1D array to a 2D array with 2 rows


#Stack arrays a and b vertically
#Input

#a = np.arange(10).reshape(2,-1)
#b = np.repeat(1, 10).reshape(2,-1)

#Desired Output:

#> array([[0, 1, 2, 3, 4],
#>        [5, 6, 7, 8, 9],
#>        [1, 1, 1, 1, 1],
#>        [1, 1, 1, 1, 1]])


#Stack the arrays a and b horizontally.


#Create the following pattern without hardcoding. Use only numpy functions and the below input array a.
#Input:

#a = np.array([1,2,3])`

#Desired Output:

#> array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])


#Get the common items between a and b
#Input:

#a = np.array([1,2,3,2,3,4,3,4,5,6])
#b = np.array([7,2,10,2,7,4,9,4,9,8])

#Desired Output:

#array([2, 4])


#From array a remove all items present in array b
#Input:

#a = np.array([1,2,3,4,5])
#b = np.array([5,6,7,8,9])

#Desired Output:

#array([1,2,3,4])


#Get the positions where elements of a and b match
#Input:

#a = np.array([1,2,3,2,3,4,3,4,5,6])
#b = np.array([7,2,10,2,7,4,9,4,9,8])

#Desired Output:

#> (array([1, 3, 5, 7]),)


#Get all items between 5 and 10 from a.
#Input:

#a = np.array([2, 6, 1, 9, 10, 3, 27])

#Desired Output:

#(array([6, 9, 10]),)




