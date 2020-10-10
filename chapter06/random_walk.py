#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# 0 is the left terminal state
# 6 is the right terminal state
# 1 ... 5 represents A ... E
# 0就是左边的终点，6就是右边的终点
VALUES = np.zeros(7)
# 所有的价值函数都是按照0.5来说的
#  [start_index，stop_index]是包括start_index，不包括stop_index，意思就是不包括6
VALUES[1:6] = 0.5
# For convenience, we assume all rewards are 0
# and the left terminal state has value 0, the right terminal state has value 1
# This trick has been used in Gambler's Problem
# 0 0.5 0.5 0.5 0.5 0.5 1
VALUES[6] = 1

# set up true state values
# 这个就是真实的价值函数A-E 0 1/6 -----5/6 1
TRUE_VALUE = np.zeros(7)
# 这个没有6
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUE[6] = 1

# 通过下面的二项分布进行选择动作
ACTION_LEFT = 0
ACTION_RIGHT = 1

# @values: current states value, will be updated if @batch is False
# @alpha: step size
# @batch: whether to update @values
# 请注意，函数内部的values就是输入参数所对应的值，所以里面处理的就是地址所对应的值
# 返回的是从策略和策略所对应的回报
def temporal_difference(values, alpha=0.1, batch=False):
    # 这个3就是C，就是从C开始
    state = 3
    trajectory = [state]
    rewards = [0]
    while True:
        # 这个和MC的区别就是，这个没进行一步，就会更新这步所对应的价值函数
        old_state = state
        # 这一段就是伪代码中的执行策略，并且观察执行策略后的状态
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        # Assume all rewards are 0
        reward = 0
        trajectory.append(state)
        # TD update 如果batch为假，就执行TD更新
        if not batch:
            values[old_state] += alpha * (reward + values[state] - values[old_state])
        # 这里就直接退出了，所以也就不会进行终止状态的价值函数的评估
        if state == 6 or state == 0:
            break
        rewards.append(reward)
    return trajectory, rewards

# @values: current states value, will be updated if @batch is False
# @alpha: step size
# @batch: whether to update @values
# 返回策略，和策略所对应的价值
def monte_carlo(values, alpha=0.1, batch=False):
    state = 3
    trajectory = [3]

    # if end up with left terminal state, all returns are 0
    # if end up with right terminal state, all returns are 1
    while True:
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        # 下面的判断就是检查这一幕是否结束
        if state == 6:
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break
    # 表示一幕已经结束，这个就是MC的缺点，必须结束才可以进行判断
    if not batch:
        for state_ in trajectory[:-1]:
            # MC update 为什么这个returns就是一个值呢，对于MC来说这个就是G，G的含义就是从这个时刻到终止时刻的回报的累加，
            # 所以说这个值不是1就是0
            values[state_] += alpha * (returns - values[state_])
    # 用数字x乘以一个列表生成一个新列表，即原来的列表被重复x次（字符串也这样
    return trajectory, [returns] * (len(trajectory) - 1)

# Example 6.2 left
# 这个就是计算左端，但是temporal_difference这个遇到右端也会停止，所以这个怎么说
def compute_state_value():
    episodes = [0, 1, 10, 100]
    # VALUES = np.zeros(7)，表示左端点+A-E+右端点
    current_values = np.copy(VALUES)
    plt.figure(1)
    for i in range(episodes[-1] + 1):
        # 这个就是按照episodes中的数字进行每次显示一下current_values的值
        if i in episodes:
            plt.plot(current_values, label=str(i) + ' episodes')
        # 带入函数，返回的价值函数还是存放在current_values中
        temporal_difference(current_values)
    plt.plot(TRUE_VALUE, label='true values')
    plt.xlabel('state')
    plt.ylabel('estimated value')
    plt.legend()

# Example 6.2 right rms就是均方根
def rms_error():
    # Same alpha value can appear in both arrays
    # 这个alpha就是常量步长参数
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，
    for i, alpha in enumerate(td_alphas + mc_alphas):
        # 这个就是每一幕的误差
        total_errors = np.zeros(episodes)
        if i < len(td_alphas):
            method = 'TD'
            linestyle = 'solid'
        else:
            method = 'MC'
            linestyle = 'dashdot'
        for r in tqdm(range(runs)):
            errors = []
            current_values = np.copy(VALUES)
            # 这个errors就是每一幕的误差都被放进去
            for i in range(0, episodes):
                errors.append(np.sqrt(np.sum(np.power(TRUE_VALUE - current_values, 2)) / 5.0))
                if method == 'TD':
                    temporal_difference(current_values, alpha=alpha)
                else:
                    monte_carlo(current_values, alpha=alpha)
            total_errors += np.asarray(errors)
        # total_errors存储着每一幕的误差
        total_errors /= runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', alpha = %.02f' % (alpha))
    plt.xlabel('episodes')
    plt.ylabel('RMS')
    plt.legend()

# Figure 6.2
# @method: 'TD' or 'MC'   这里面有批量更新数据
def batch_updating(method, episodes, alpha=0.001):
    # perform 100 independent runs
    runs = 100
    total_errors = np.zeros(episodes)
    for r in tqdm(range(0, runs)):
        current_values = np.copy(VALUES)
        errors = []
        # track shown trajectories and reward/return sequences
        trajectories = []
        rewards = []
        # 进行一幕一幕的遍历
        for ep in range(episodes):
            if method == 'TD':
                # 意思这里面就没有进行TD预测
                trajectory_, rewards_ = temporal_difference(current_values, batch=True)
            else:
                trajectory_, rewards_ = monte_carlo(current_values, batch=True)
            # 这个就是批量更新中的每经历，之前所有幕的数据就被视为一个批次
            trajectories.append(trajectory_)
            rewards.append(rewards_)
            # 这个就是批量更新过程，循环，知道价值不变
            while True:
                # keep feeding our algorithm with trajectories seen so far until state value function converges
                # 继续用迄今为止看到的轨迹填充我们的算法，直到状态值函数收敛
                updates = np.zeros(7)
                # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
                '''
                原来 批量TP 与 批量MC 的差别只在于两点：
                - 产生序列的方式：
                - - 在这个例子中，在 TD 看来，每步的收益与本身的动作有关，即前面动作收益皆为 0 ，与最后一次触发终止的动作无关 0 或 1
                - - 在 MC 看来，（因为没有折扣），每步的收益与最后一次触发终止的动作有关 0 或 1
                - 更新公式，如下
                '''
                for trajectory_, rewards_ in zip(trajectories, rewards):
                    for i in range(0, len(trajectory_) - 1):
                        if method == 'TD':
                            # trajectory_[i]对应的状态，updates[trajectory_[i]]意思就是状态所对应的价值函数
                            # 这个就是TD里面的，所有幕中，该状态所有增量和
                            updates[trajectory_[i]] += rewards_[i] + current_values[trajectory_[i + 1]] - current_values[trajectory_[i]]
                        else:
                            updates[trajectory_[i]] += rewards_[i] - current_values[trajectory_[i]]
                updates *= alpha
                if np.sum(np.abs(updates)) < 1e-3:
                    break
                # perform batch updating
                current_values += updates
            # calculate rms error
            errors.append(np.sqrt(np.sum(np.power(current_values - TRUE_VALUE, 2)) / 5.0))
        total_errors += np.asarray(errors)
    # 这里存储这每一幕的误差
    total_errors /= runs
    return total_errors

def example_6_2():
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    compute_state_value()

    plt.subplot(2, 1, 2)
    rms_error()
    #tight_layout会自动调整子图参数，使之填充整个图像区域它
    plt.tight_layout()

    plt.savefig('../images/example_6_2.png')
    plt.close()

def figure_6_2():
    episodes = 100 + 1
    td_erros = batch_updating('TD', episodes)
    mc_erros = batch_updating('MC', episodes)

    plt.plot(td_erros, label='TD')
    plt.plot(mc_erros, label='MC')
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.legend()

    plt.savefig('../images/figure_6_2.png')
    plt.close()

if __name__ == '__main__':
    example_6_2()
    figure_6_2()
