# %matplotlib inline
import torch
from IPython import display
from d2l import torch as d2l

# # torch.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64), eigenvectors=True)
# print(torch.linalg.eig(torch.tensor([[2.0, 1], [2, 3]])))
#
# A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
#                   [0.1, 3.0, 0.2, 0.3],
#                   [0.1, 0.2, 5.0, 0.5],
#                   [0.1, 0.3, 0.5, 9.0]])
# # v, _ = torch.eig(A)
# v, _ = torch.linalg.eig(A)
# print(v)
#
torch.manual_seed(42)
k = 5
A = torch.randn(k, k, dtype=torch.float64)
print(A)
# # Calculate the sequence of norms after repeatedly applying `A`
# v_in = torch.randn(k, 1, dtype=torch.float64)
#
# norm_list = [torch.norm(v_in).item()]
# for i in range(1, 100):
#     v_in = A @ v_in
#     # v_in = torch.mv(A, v_in)
#     norm_list.append(torch.norm(v_in).item())
#
# d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
# d2l.plt.show()
#
# # Compute the scaling factor of the norms
# norm_ratio_list = []
# for i in range(1, 100):
#     norm_ratio_list.append(norm_list[i] / norm_list[i - 1])
#
# d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
# d2l.plt.show()


# Compute the eigenvalues
M = torch.eig(A)
print(type(M), M.__sizeof__())
# eigs = torch.eig(A)[0][:, 0].tolist()
eigs = torch.eig(A)[0][:, 0].tolist()
print(eigs)
print()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')

# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
d2l.plt.show()

# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i] / norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
d2l.plt.show()
