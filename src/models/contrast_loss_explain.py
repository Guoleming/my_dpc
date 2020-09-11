import torch

import math

# pred: [B, pred_step, D, last_size, last_size]
# GT: [B, N, D, last_size, last_size]

B = 5
pred_step = 3
last_size = int(math.ceil(64 / 32))
feature_size = 512
N = pred_step


pred = torch.randn(B, pred_step, feature_size, last_size, last_size)
feature_inf = torch.randn(B, pred_step, feature_size, last_size, last_size)
pred = pred.permute(0,1,3,4,2).contiguous().view(B*pred_step*last_size**2, feature_size)
feature_inf = feature_inf.permute(0,1,3,4,2).contiguous().view(B*N*last_size**2, feature_size).transpose(0,1)

score = torch.matmul(pred, feature_inf).view(B, pred_step, last_size**2, B, N, last_size**2)
print(score.shape)
# exit()


# the first three dimension is for predict. the last three dimension is for ground truth
mask = torch.zeros((B, pred_step, last_size ** 2, B, N, last_size ** 2), dtype=torch.int8, requires_grad=False).detach()

# easy neg: 0
# in the same batch, but different sample/video

# spatial negative: -3
# in the same sample/video, different temporal and spatial.
mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3  # spatial negative

# temporal neg: -1
# in the same sample/video, same spatial, different temporal
for k in range(B):
    mask[k, :, torch.arange(last_size ** 2), k, :, torch.arange(last_size ** 2)] = -1

# mask.permute: [B, last_size**2, pred_step, B, las_size ** 2, N]
tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B * last_size ** 2, pred_step, B * last_size ** 2, N)

# positive: 1
# positive: same sample, same tempoal and spatial
for j in range(B * last_size ** 2):
    tmp[j, torch.arange(pred_step), j, torch.arange(N - pred_step, N)] = 1

mask = tmp.view(B, last_size ** 2, pred_step, B, last_size ** 2, N).permute(0, 2, 1, 3, 5, 4)

# let's print the first sample, the same temporal:
# the result should be: the diagonal is positive sample: 1, the others is spatial negative sample: -3
print(mask[0, 0, :, 0, 0, :])
# the shape is [2**2, N, 2**2]

# let's print the first sample, the different temporal
# the result should be: the diagonal is temporal negative sample :-1, the others is spatio negative sample: -3
print(mask[0, 0, :, 0, 1, :])

# let's print the different sample
# all of them is zero, easy negative, sample from different sample/video
print(mask[0, 0, :, 1, 0, :])

(B, NP, SQ, B2, NS, _) = mask.size()

target = mask == 1
target.requires_grad = False

# print(target)

# loss
import torch.nn as nn


criterion = nn.CrossEntropyLoss()

# the shape of probability is [B * NP * SQ,  B2 * NS * SQ]
# This  cna seen [B * NP * SQ] instance, and class number of each instance is [B2 * NS * SQ]
score_flattened = score.view(B * NP * SQ, B2 * NS * SQ)
target_flattened = target.view(B * NP * SQ, B2 * NS * SQ)
target_flattened = target_flattened.long().argmax(dim=1)

loss = criterion(score_flattened, target_flattened)

