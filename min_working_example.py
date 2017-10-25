import ordinal_TSA
import numpy as np


TS = np.random.rand(50000)

wpe = np.asarray(ordinal_TSA.permutation_entropy(TS.reshape(len(TS),1),dim=3,step=1,w=1))
print(wpe)
TS = np.ones(50000)
wpe = np.asarray(ordinal_TSA.permutation_entropy(TS.reshape(len(TS),1),dim=3,step=1,w=1))

print(wpe)