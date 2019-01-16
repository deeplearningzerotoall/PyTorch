"""
Corresponds to Lab 8 of 모두를 위한 딥러닝 강좌 시즌 1 for TensorFlow.
"""
import numpy as np
import torch


print('-----------------------')
print('NumPy Review - 1D Array')
print('-----------------------')
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
print(t.ndim)  # rank
print(t.shape) # shape
print(t[0], t[1], t[-1]) # Element
print(t[2:5], t[4:-1])   # Slicing
print(t[:2], t[3:])      # Slicing

print()
print('-----------------------')
print('NumPy Review - 2D Array')
print('-----------------------')
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
print(t.ndim)  # rank
print(t.shape) # shape

print()
print('-------------------------------')
print('PyTorch is just like NumPy - 1D')
print('-------------------------------')
t = np.array([0., 1., 2., 3., 4., 5., 6.])
ft = torch.FloatTensor(t)
print(ft)
print(ft.dim())  # rank
print(ft.shape)  # shape
print(ft.size()) # shape
print(ft[0], ft[1], ft[-1]) # Element
print(ft[2:5], ft[4:-1])    # Slicing
print(ft[:2], ft[3:])       # Slicing

print()
print('-------------------------------')
print('PyTorch is just like NumPy - 2D')
print('-------------------------------')
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
ft = torch.FloatTensor(t)
print(ft)
print(ft.dim())  # rank
print(ft.size()) # shape

print()
print('-----------------')
print('Shape, Rank, Axis')
print('-----------------')
t = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
               [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
ft = torch.FloatTensor(t)
print(ft.dim())  # rank  = 4
print(ft.size()) # shape = (1, 2, 3, 4)

print()
print('-------------')
print('Mul vs Matmul')
print('-------------')
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2))

print()
print('----------------------')
print('Broadcasting [WARNING]')
print('----------------------')
# Same shape
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # 3 -> [[3, 3]]
print(m1 + m2)

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

print()
print('---------')
print('Mean - 1D')
print('---------')
t = torch.FloatTensor([1, 2])
print(t.mean())

# Can't use mean() on integers
t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)

print()
print('---------')
print('Mean - 2D')
print('---------')
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))

print()
print('--------')
print('Sum - 2D')
print('--------')
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))

print()
print('--------------')
print('Max and Argmax')
print('--------------')
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.max()) # Returns one value: max
print(t.max(dim=0)) # Returns two values: max and argmax
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])

print(t.max(dim=1))
print(t.max(dim=-1))

print()
print('------')
print('View**')
print('------')
t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)

print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)

print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)

print()
print('---------------------')
print('Squeeze and Unsqueeze')
print('---------------------')
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)

ft = torch.Tensor([0, 1, 2])
print(ft.shape)

print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)

print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)

print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)

print()
print('----------------------------')
print('Scatter for One-hot encoding')
print('----------------------------')
lt = torch.LongTensor([[0], [1], [2], [0]])
print(lt)

one_hot = torch.zeros(4, 3) # batch_size = 4, classes = 3
one_hot.scatter_(1, lt, 1)
print(one_hot)

print()
print('-------')
print('Casting')
print('-------')
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())

bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())

print()
print('--------')
print('Stacking')
print('--------')
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim=1))

print()
print('-------------------')
print('Ones and Zeros like')
print('-------------------')

x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
print(torch.ones_like(x))
print(torch.zeros_like(x))

print()
print('---')
print('Zip')
print('---')

for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)

for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
