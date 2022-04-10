# v = [1, 7, 0, 1]
# print(v)

# %matplotlib inline
import torch
import torchvision
from IPython import display
from torchvision import transforms
from d2l import torch as d2l

# def angle(v, w):
#     return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))
#
#
# anglevw = angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
# print(anglevw)


# # Load in the dataset
# trans = []
# trans.append(transforms.ToTensor())
# trans = transforms.Compose(trans)
# train = torchvision.datasets.FashionMNIST(root="./data", transform=trans, train=True, download=True)
# test = torchvision.datasets.FashionMNIST(root="./data", transform=trans, train=False, download=True)
#
# # x0 = train[0]
# # x1 = train[1]
# # x2 = train[2]
# # x3 = train[3]
# # x4 = train[4]
# # # print(x0, '\n', type(x0), '\n', x0.__sizeof__())
# # # print(x1, '\n', type(x1), '\n', x1.__sizeof__())
# # # print(x2, '\n', type(x2), '\n', x2.__sizeof__())
# # # print(x3, '\n', type(x3), '\n', x3.__sizeof__())
# # print(x4[1], '\n', type(x4[0]), '\n', x4[0].size())
# #
# # from matplotlib import pyplot as plt
# #
# # # Plot average t-shirt
# # d2l.set_figsize()
# # plt.imshow(256 * x4[0].reshape(28, 28), cmap='Greys')
# # # d2l.plt.imshow(x4.reshape(28, 28).tolist(), cmap='Greys')
# # plt.show()
#
# X_train_0 = torch.stack([x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
# X_train_1 = torch.stack([x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
# X_test = torch.stack([x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
# y_test = torch.stack([torch.tensor(x[1]) for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
#
# # Compute averages
# ave_0 = torch.mean(X_train_0, axis=0)
# ave_1 = torch.mean(X_train_1, axis=0)
# # print(ave_0)
# # print(ave_1)
#
# # Plot average t-shirt
# d2l.set_figsize()
# d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
# d2l.plt.show()
#
# # Plot average trousers
# d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
# d2l.plt.show()
#
# # Print test set accuracy with eyeballed threshold
# w = (ave_1 - ave_0).T
# # '@' is Matrix Multiplication operator in pytorch.
# predictions = X_test.reshape(2000, -1) @ (w.flatten()) > -1500000
#
# # Accuracy
# accuracy = torch.mean((predictions.type(y_test.dtype) == y_test).float(), dtype=torch.float64)
# print(accuracy)

M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
temp = M_inv @ M
print(temp)

ans = torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
print(ans)

# Define tensors
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Print out the shapes
print(A.shape, B.shape, v.shape)

# Reimplement matrix multiplication
ans = torch.einsum("ij, j -> i", A, v)
print(ans)
print(A@v)

torch.einsum("ijk, il, j -> kl", B, A, v)
