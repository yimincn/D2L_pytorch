# %matplotlib inline
import torch
from IPython import display
from d2l import torch as d2l

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot a function in a normal range
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big ** x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
d2l.plt.show()

# Plot a the same function in a tiny range
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med ** x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
d2l.plt.show()

# Plot a the same function in a tiny range
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small ** x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')


# Define our function
def L(x):
    return x ** 2 + 1701 * (x - 4) ** 3


# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4 + epsilon) - L(4)) / epsilon:.5f}')

# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
d2l.plt.show()

# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)) - (xs - x0) ** 2 *
                 torch.sin(torch.tensor(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])

# Compute the exponential function
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs ** 2 / 2
P5 = 1 + xs + xs ** 2 / 2 + xs ** 3 / 6 + xs ** 4 / 24 + xs ** 5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=["Exponential", "Degree 1 Taylor Series",
                                                    "Degree 2 Taylor Series", "Degree 5 Taylor Series"])
d2l.plt.show()
