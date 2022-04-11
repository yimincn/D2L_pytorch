import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x * y, x / y, x ** y
(tensor([5.]), tensor([6.]), tensor([1.5000]), tensor([9.]))
[2]
x = torch.arange(4)
x
tensor([0, 1, 2, 3])
[3]
x[3]
tensor(3)
[4]
len(x)
4
[5]
x.shape
torch.Size([4])
[6]
A = torch.arange(20).reshape(5, 4)
A
tensor([[0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])
[7]
A.T
tensor([[0, 4, 8, 12, 16],
        [1, 5, 9, 13, 17],
        [2, 6, 10, 14, 18],
        [3, 7, 11, 15, 19]])
[8]
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
tensor([[1, 2, 3],
        [2, 0, 4],
        [3, 4, 5]])
[9]
B == B.T
tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
[10]
X = torch.arange(24).reshape(2, 3, 4)
X
tensor([[[0, 1, 2, 3],
         [4, 5, 6, 7],
         [8, 9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
[11]
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
A, A + B
(tensor([[0., 1., 2., 3.],
         [4., 5., 6., 7.],
         [8., 9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]),
 tensor([[0., 2., 4., 6.],
         [8., 10., 12., 14.],
         [16., 18., 20., 22.],
         [24., 26., 28., 30.],
         [32., 34., 36., 38.]]))
[13]
id(A), id(B)
(2058959021936, 2058959023456)
[14]
A * B
tensor([[0., 1., 4., 9.],
        [16., 25., 36., 49.],
        [64., 81., 100., 121.],
        [144., 169., 196., 225.],
        [256., 289., 324., 361.]])
[15]
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
(tensor([[[2, 3, 4, 5],
          [6, 7, 8, 9],
          [10, 11, 12, 13]],

         [[14, 15, 16, 17],
          [18, 19, 20, 21],
          [22, 23, 24, 25]]]),
 torch.Size([2, 3, 4]))
[16]
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
(tensor([0., 1., 2., 3.]), tensor(6.))
[17]
A.shape, A.sum()
(torch.Size([5, 4]), tensor(190.))
[23]
A_sum_axis0 = A.sum(axis=0)
A, A_sum_axis0, A_sum_axis0.shape
(tensor([[0., 1., 2., 3.],
         [4., 5., 6., 7.],
         [8., 9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]),
 tensor([40., 45., 50., 55.]),
 torch.Size([4]))
[21]
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
(tensor([6., 22., 38., 54., 70.]), torch.Size([5]))
[22]
A.sum(axis=[0, 1])
tensor(190.)
[24]
A.mean(), A.sum() / A.numel()
(tensor(9.5000), tensor(9.5000))
[25]
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
(tensor([8., 9., 10., 11.]), tensor([8., 9., 10., 11.]))
[31]
# sum_A = A.sum(axis=1, keepdims=True)
sum_A = A.sum(axis=1)
sum_A, sum_A.shape
(tensor([6., 22., 38., 54., 70.]), torch.Size([5]))
[32]
A / sum_A
---------------------------------------------------------------------------
RuntimeError
Traceback(most
recent
call
last)
Input
In[32], in < cell
line: 1 > () - ---> 1
A / sum_A
RuntimeError: The
size
of
tensor
a(4)
must
match
the
size
of
tensor
b(5)
at
non - singleton
dimension
1
[34]
A, A.cumsum(axis=0)
(tensor([[0., 1., 2., 3.],
         [4., 5., 6., 7.],
         [8., 9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]),
 tensor([[0., 1., 2., 3.],
         [4., 6., 8., 10.],
         [12., 15., 18., 21.],
         [24., 28., 32., 36.],
         [40., 45., 50., 55.]]))
[35]
y = torch.ones(4, dtype=torch.float32)
x, y, torch.dot(x, y)
(tensor([0., 1., 2., 3.]), tensor([1., 1., 1., 1.]), tensor(6.))
[36]
torch.sum(x * y)
tensor(6.)
[38]
A, A.shape, x.shape, torch.mv(A, x)
(tensor([[0., 1., 2., 3.],
         [4., 5., 6., 7.],
         [8., 9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]),
 torch.Size([5, 4]),
 torch.Size([4]),
 tensor([14., 38., 62., 86., 110.]))
[39]
B = torch.ones(4, 3)
torch.mm(A, B)
tensor([[6., 6., 6.],
        [22., 22., 22.],
        [38., 38., 38.],
        [54., 54., 54.],
        [70., 70., 70.]])
[40]
u = torch.tensor([3.0, -4.0])
torch.norm(u)
tensor(5.)
[41]
torch.abs(u).sum()
tensor(7.)
[42]
torch.norm(torch.ones((4, 9)))
tensor(6.)