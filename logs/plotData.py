import os
import numpy as np
from posixpath import split
import sys

from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No file specified")
        exit()

    log = False
    if len(sys.argv) == 3:
        if sys.argv[2] == "log":
            log = True

    filename = sys.argv[1]


    if not os.path.isfile(filename):
        print("File", filename, "does not exist")
        exit()

    accuracy = []
    gamma = []
    C_val = []
    with(open(filename, 'r')) as f:
        lines = f.readlines()

        for l in lines:
            markers = l.split(" ")

            g, c, acc = (x.split(":")[1] for x in markers[:3])
            accuracy.append(float(acc))
            gamma.append(float(g))
            C_val.append(float(c))


        x = list(set(gamma))
        y = list(set(C_val))

        x.sort()
        y.sort()

        accuracy = np.array(accuracy).reshape((len(x), len(y)))
        accuracy = np.rot90(accuracy, 1)
        #print(accuracy)

        plt.imshow(accuracy, cmap='hot', interpolation='nearest', 
                    extent=[min(x), max(x), min(y), max(y)], aspect='auto',
                    norm=MidpointNormalize(vmin=0.986, midpoint=0.99))

        plt.xlabel("Gamma")
        plt.ylabel("C")

        plt.locator_params(axis='x', nbins=len(x))
        plt.locator_params(axis='y', nbins=len(y))

        if log:
            plt.xticks(np.arange(1, 11, step=10/len(x)), x)
            plt.yticks(np.arange(1, 11, step=10/len(y)), y)

        #plt.xlim(min(x), max(x))
        #plt.xscale('log')
        #plt.yscale('log')
        # plt.xticks(x)

        #plt.yticks(y)
        #plt.ylim(min(y), max(y))

        plt.colorbar()
        plt.show()