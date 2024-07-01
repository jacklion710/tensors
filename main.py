import torch

points = torch.tensor([[4.0, 1.0], [5., 3.], [2., 1.]]) # Create an explicit tensor
# Querying tensor elements
print(points)
print(points[0, 1])
print(points[0])

# Slicing tensors
print(points[1:])
print(points[1:, :])
print(points[1:, 0])
print(points[None])

img_t = torch.randn(3, 5, 5) # shape [channels, rows, columns]
weights = torch.tensor([0.2126, 0.7152, 0.0722])
batch_t = torch.randn(2, 3, 5, 5) # shape [batch, channels, rows, columns]

# Convert image to grayscale
img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)
print(img_gray_naive.shape, batch_gray_naive.shape)

unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)
img_weights = img_t * unsqueezed_weights
batch_weights = batch_t * unsqueezed_weights
img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)
print(batch_weights.shape, batch_t.shape, unsqueezed_weights.shape)

# Named tensors

weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
print(weights_named)

img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')
print("img named:", img_named.shape, img_named.names)
print("batch named:", batch_named.shape, batch_named.names)

weights_aligned = weights_named.align_as(img_named)
print(weights_aligned.shape, weights_aligned.names)

gray_named = (img_named * weights_aligned).sum('channels')
print(gray_named.shape, gray_named.names)

gray_plain = gray_named.rename(None) # Remove names
print(gray_plain.shape, gray_plain.names)

# Tensor data types

double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)

print(short_points.dtype)

double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()

double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)

points_64 = torch.rand(5, dtype=torch.double)
points_short = points_64.to(torch.short)
print(points_64 * points_short)

# Tensor API

a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)

print(a.shape, a_t.shape)

a_t = a.transpose(0, 1)

print(a.shape, a_t.shape)

# Indexing into storage

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points.storage())

points_storage = points.storage()
print(points_storage[0])

print(points.storage()[1])

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storage = points.storage()
points_storage[0] = 2.0
print(points)

# Modifying stored values: In-place operations

a = torch.ones(3, 2)

print(a.zero_())

# Tensor metadata: Size, offset, and stride

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
print(second_point.storage_offset())

print(second_point.size())

print(second_point.shape)

print(points.stride())

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_points = points[1]
second_point[0] = 10.0
print(points)

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1].clone()
second_point[0] = 10.0
print(points)

# Transposing without copying

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points)

points_t = points.t()
print(points_t)

print(id(points.storage()) == id(points_t.storage()))

print(points.stride())

print(points_t.stride())

# Transposing in higher dimensions

some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
print(some_t.shape)

print(transpose_t.shape)

print(some_t.stride)

print(transpose_t.stride())

# Contiguous tensors

print(points.is_contiguous())

print(points_t.is_contiguous())

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_t = points.t()
print(points_t)

print(points_t.storage())

print(points_t.stride())

points_t_cont = points_t.contiguous()
print(points_t_cont)

print(points_t_cont.stride())

print(points_t_cont.storage())

# Managing a tensors device attribute

points_gpu = torch.tensor([[4.0, 3.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
