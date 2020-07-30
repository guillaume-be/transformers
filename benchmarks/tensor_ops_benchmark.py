import math
import time

import torch

if __name__ == '__main__':

    input = torch.rand((32, 128, 512)).cuda()
    weights = torch.rand((512, 512)).cuda()
    _ = input.matmul(weights)
    n_iter = 15000
    matmul_times = []
    outputs = []
    t1 = time.time()
    for idx in range(n_iter):
        with torch.no_grad():
            input.matmul(weights)
    t2 = time.time()

    mean_matmul = (t2-t1)/n_iter
    print(f'Matmul: {mean_matmul * 1e6}s')
