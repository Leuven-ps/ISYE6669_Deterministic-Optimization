import os

import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "tech_tower.txt")

X = np.loadtxt(file_path)

U, S, V = np.linalg.svd(X)

X_5 = U[:, :5] @ np.diag(S[:5]) @ V[:5, :]
X_15 = U[:, :15] @ np.diag(S[:15]) @ V[:15, :]
X_25 = U[:, :25] @ np.diag(S[:25]) @ V[:25, :]

plt.gray()
plt.imshow(X_5)
plt.savefig(os.path.join(script_dir, "k5.png"))
plt.close()

plt.gray()
plt.imshow(X_15)
plt.savefig(os.path.join(script_dir, "k15.png"))
plt.close()

plt.gray()
plt.imshow(X_25)
plt.savefig(os.path.join(script_dir, "k25.png"))
plt.close()
