import numpy as np
import matplotlib.pyplot as plt

losses = np.load("logs/2021_10_05.npy")
sample_num = 30
epochs = [i for i in range(len(losses))][::sample_num]
losses = np.log(np.log(losses))[::sample_num]

plt.plot(epochs, losses)
plt.ylabel("log(log(loss))")
plt.xlabel("epoch")
plt.show()
