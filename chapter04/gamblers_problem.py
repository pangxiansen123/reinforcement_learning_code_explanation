#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
#### 赌徒问题
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# goal
GOAL = 100

# all states, including state 0 and state 100
# 生成一个列表[0 1 2 ......100]
STATES = np.arange(GOAL + 1)

# probability of head
HEAD_PROB = 0.4


def figure_4_3():
    # state value
    # state_value = [0 0 0 0]101个
    state_value = np.zeros(GOAL + 1)
    # 收益其他的状态为0,最终态为1
    state_value[GOAL] = 1.0

    sweeps_history = []

    # value iteration
    # 这个是值函数，就是从起点到终点的各个值（V）
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)
        # 遍历所有的状态，就是每一个状态都找一个最优的动作
        for state in STATES[1:GOAL]:
            # get possilbe actions for current state
            # 动作，就是下的赌资[0 1 2 3 。。。。min{state,100-state}]
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            #这个就是对所有的动作进行遍历 找到最优的动作
            for action in actions:
                # 这个就是p*[r+v]=(0.4)*(成功后的状态)+(0.6)*(失败后的状态)
                action_returns.append(
                    HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])
            new_value = np.max(action_returns)
            state_value[state] = new_value
        # 这个矩阵中的最大值
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break

    # compute the optimal policy
    # 这个就是计算策略的，和上面的一样，只是这里面重新弄了一下，只是为了显示数据
    # 这个下面的程序其实就是为了得出上面的状态所对应的函数
    policy = np.zeros(GOAL + 1)
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(
                HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])

        # round to resemble the figure in the book, see
        # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
        # 从小数点第五位开始四舍五入
        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('../images/figure_4_3.png')
    plt.close()


if __name__ == '__main__':
    figure_4_3()
