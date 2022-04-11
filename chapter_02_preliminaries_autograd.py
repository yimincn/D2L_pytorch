import torch

x = torch.arange(4.0)
x
tensor([0., 1., 2., 3.])
[18]
x.requires_grad_(True)
x.grad
[19]
y = 2 * torch.dot(x, x)
y
tensor(28., grad_fn=<MulBackward0>)
[20]
y.backward()
[21]
x.grad
tensor([ 0.,  4.,  8., 12.])
[22]
x.grad == 4 * x
tensor([True, True, True, True])
[23]
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
tensor([1., 1., 1., 1.])
[24]
x.grad.zero_()
y = x * x
y.sum().backward()
x.grad
tensor([0., 2., 4., 6.])
[26]
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u, x.grad, u
(tensor([True, True, True, True]),
 tensor([0., 1., 4., 9.]),
 tensor([0., 1., 4., 9.]))
[33]
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
tensor([True, True, True, True])
[40]
def f(a):
    b = a * 2
    while b.norm() < 1000:
        print("b.norm:",b.norm())
        b = b * 2
    if b.sum() > 0:
        print("b.sum:",b.sum())
        c = b
        print(c)
    else:
        c = 100 * b
        print(c)
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

a, d, a.grad == d / a , a.grad
b.norm: tensor(3.0625, grad_fn=<CopyBackwards>)
b.norm: tensor(6.1251, grad_fn=<CopyBackwards>)
b.norm: tensor(12.2501, grad_fn=<CopyBackwards>)
b.norm: tensor(24.5003, grad_fn=<CopyBackwards>)
b.norm: tensor(49.0005, grad_fn=<CopyBackwards>)
b.norm: tensor(98.0010, grad_fn=<CopyBackwards>)
b.norm: tensor(196.0020, grad_fn=<CopyBackwards>)
b.norm: tensor(392.0040, grad_fn=<CopyBackwards>)
b.norm: tensor(784.0081, grad_fn=<CopyBackwards>)
tensor(-156801.6094, grad_fn=<MulBackward0>)

(tensor(-1.5313, requires_grad=True),
 tensor(-156801.6094, grad_fn=<MulBackward0>),
 tensor(True),
 tensor(102400.))
