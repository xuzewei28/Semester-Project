import numpy as np
import matplotlib.pyplot as plt
import os


def str_to_dict(s):
    import re
    x = re.split(",", s[1:-1])
    res = dict()
    for i in x:
        key, value = i.strip().split(":")
        res[key[1:-1]] = float(value)
    return res


def plot_ris(dir, axs, name, f, size=2000):
    a = open(dir, 'r')
    a = a.read().split('\n')[:-1]
    a = [str_to_dict(b) for b in a]
    train = [s['train_loss'] for s in a]
    test = [s['test_loss'] for s in a]

    axs[0].plot(np.arange(size), train[:size], label=f)
    axs[1].plot(np.arange(size), test[:size], label=f)
    axs[0].set_title(name + " train result")
    axs[0].legend()
    axs[1].set_title(name + " test result")
    axs[1].legend()
    return max(test), test.index(max(test)), test[-1]


dir = 'Prova/AutoEncoder/'
size = 50
files = os.listdir(dir)
fig, axs = plt.subplots(2, figsize=(20, 20))
a = []
names = os.listdir(dir)
for name in names:
    v, i, last = plot_ris(dir + name, axs, '', name, size=size)
    a.append((dir + name, v, i, last))

a.sort(key=lambda x: x[1])
for i in a:
    print(i)
plt.savefig("ResNet.png")
plt.ylim([0,1e-3])
plt.show()
