import numpy as np


vlen = 300
num_seq = 4
seq_len = 4
downsample = 1

n = 1
start_idx = np.random.choice(range(vlen - num_seq * seq_len * downsample), n)

seq_idx = np.expand_dims(np.arange(num_seq), -1) * downsample * seq_len + start_idx
seq_idx_block = seq_idx + np.expand_dims(np.arange(seq_len), 0) * downsample


print(seq_idx_block)