#######################################################################
# Copyright (C)                                                       #
# 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# for figure 8.7, run a simulation of 2 * @b steps
def b_steps(b):
    # set the value of the next b states
    # it is not clear how to set this
    # 通过本函数可以返回一个或一组服从标准正态分布的随机样本值。
    distribution = np.random.randn(b)

    # true value of the current state
    # 真实的值
    true_v = np.mean(distribution)

    samples = []
    errors = []

    # sample 2b steps
    # 就是采样2*b步，就是根据分支的数目对其进行采样
    for t in range(2 * b):
        v = np.random.choice(distribution)
        samples.append(v)
        errors.append(np.abs(np.mean(samples) - true_v))

    return errors

def figure_8_7():
    runs = 100
    branch = [2, 10, 100, 1000]
    for b in branch:
        # 因为是采样2*b步，所以这里面是2*b
        errors = np.zeros((runs, 2 * b))
        for r in tqdm(np.arange(runs)):
            errors[r] = b_steps(b)
        errors = errors.mean(axis=0)
        x_axis = (np.arange(len(errors)) + 1) / float(b)
        plt.plot(x_axis, errors, label='b = %d' % (b))

    plt.xlabel('number of computations')
    plt.xticks([0, 1.0, 2.0], ['0', 'b', '2b'])
    plt.ylabel('RMS error')
    plt.legend()

    plt.savefig('../images/figure_8_7.png')
    plt.close()

if __name__ == '__main__':
    figure_8_7()
