#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
# 这个就是例题5.5
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 回报值，到达目的地就是1
ACTION_BACK = 0
ACTION_END = 1

# behavior policy 要么返回0，要么返回1
def behavior_policy():
    return np.random.binomial(1, 0.5)

# target policy
def target_policy():
    return ACTION_BACK

# one turn
def play():
    # track the action for importance ratio
    trajectory = []
    # 返回目的地的时候就会停止，并且返回
    while True:
        # 随机产生一个动作
        action = behavior_policy()
        trajectory.append(action)
        # 这个是有概率进行动作的，当选择ACTION_END，达到目标点，就直接返回回报0
        if action == ACTION_END:
            return 0, trajectory
        # 有0.9的概率回到s，有0.1的概率回到目的地，==0就是回到目的地
        if np.random.binomial(1, 0.9) == 0:
            return 1, trajectory

def figure_5_4():
    runs = 1
    episodes = 100000
    for run in range(runs):
        rewards = []
        for episode in range(0, episodes):
            reward, trajectory = play()
            # 只要是往右的策略 都是0，就是不要这个
            if trajectory[-1] == ACTION_END:
                rho = 0
            else:
                rho = 1.0 / pow(0.5, len(trajectory))
            rewards.append(rho * reward)
        # 每一个位置的元素和前面的所有元素加起来求和
        rewards = np.add.accumulate(rewards)
        estimations = np.asarray(rewards) / np.arange(1, episodes + 1)
        plt.plot(estimations)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Ordinary Importance Sampling')
    plt.xscale('log')

    plt.savefig('../images/figure_5_4.png')
    plt.close()

if __name__ == '__main__':
    figure_5_4()
