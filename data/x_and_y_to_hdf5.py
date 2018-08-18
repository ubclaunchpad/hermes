# -*- coding: utf-8 -*-
"""
    Sample script for constructing a sequence dataset with DeepDish
"""

import numpy as np
import random
import deepdish as dd
import string
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

N = 5
X = []
for i in range(0,N):
  rand = random.randint(0, 10)
  np_a = np.array([[rand for x in range(rand)] for y in range(rand)])
  X.append(np_a)

Y = [''.join(random.choices(string.ascii_uppercase + string.digits, k=N*5)) for i in range(0, N)]
dd.io.save('test.h5', (X, Y))

print((X,Y))

recreation = dd.io.load('test.h5', '/')
recreation

len(recreation['data'][1])

dataset = XY('test.h5')

len(dataset), dataset[3]
