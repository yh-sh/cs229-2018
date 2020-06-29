from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
K = 16
A = imread(os.path.join('..', 'data', 'peppers-large.tiff'))

B = imread(os.path.join('..', 'data', 'peppers-small.tiff'))
x = B.reshape((-1, 3)).copy()
m, n = x.shape
mu = x[np.random.choice(np.arange(m), size=K)].copy().astype(np.float)
c = np.zeros(m)
conver = 10
eps = 1e-3
max_it = 30
it = 0
while (it < max_it) and (conver > 1e-3):
    dist = np.linalg.norm(x[:, None] - mu[None], axis=2)
    c = np.argmin(dist, axis=1)
    for i in range(K):
        center = x[c==i]
        mu[i] = center.mean(axis=0)
    dist_new = np.linalg.norm(x[:, None] - mu[None], axis=2)
    conver =np.linalg.norm(dist - dist_new, axis=1).sum()
    it+=1
    print("it:{}, conver:{}".format(it, conver))

y = A.reshape((-1, 3)).copy()
dist = np.linalg.norm(y[:, None] - mu[None], axis=2)
c = np.argmin(dist, axis=1)
for i in range(K):
    y[c==i] = mu[i].astype(np.uint8)
print(mu)
C = y.reshape(A.shape)
plt.imshow(C)
plt.savefig('output/km.png')