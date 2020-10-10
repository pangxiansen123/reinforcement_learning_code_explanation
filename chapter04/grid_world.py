#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')
# 世界是4*4的网格
WORLD_SIZE = 4
# left, up, right, down
ACTIONS = [np.array([0, -1]), # 往左移动
           np.array([-1, 0]), # 往上移动
           np.array([0, 1]), # 往右移动
           np.array([1, 0])] # 往下移动
# 各个动作的概率
ACTION_PROB = 0.25


def is_terminal(state):
    x, y = state
    # 左上角和右下角就是终点
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)

# 就是根据动作和吃时刻的值，来计算下一时刻的状态和即时回报值
def step(state, action):
    #  检查是否到达目的地 价值是0
    if is_terminal(state):
        return state, 0

    next_state = (np.array(state) + action).tolist() # 就是将数组转化为列表

    # 把列表的值复制给x、y
    x, y = next_state
    # 如果碰到了边界，那么就回到原处
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state
    # 所有的汇报都是-1
    reward = -1
    return next_state, reward

# 就是画图
def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

        # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)

# 计算状态价值函数 in_place是什么意思？我觉得应该就是用这个变量进行持续下去的计算
def compute_state_value(in_place=True, discount=1.0):
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    # 动作价值评估，进行迭代，直到到达稳定值
    while True:
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
                # 计算每一个位置执行所有的动作时候的动作价值函数
                new_state_values[i, j] = value

        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break

        iteration += 1

    return new_state_values, iteration


def figure_4_1():
    # While the author suggests using in-place iterative policy evaluation,
    # Figure 4.1 actually uses out-of-place version.
    _, asycn_iteration = compute_state_value(in_place=True)
    values, sync_iteration = compute_state_value(in_place=False)
    draw_image(np.round(_, decimals=2))

    # 通过比较 使用同一个位置的时候迭代次数为113 而使用新的位置的时候迭代次数为172
    # 为什么会这样呢？state_values = new_state_values他们使用的是同一块内存，这个意思就是说本次更新可以用本次更新的数据
    # 这样就可以加快迭代次数了
    print('In-place: {} iterations'.format(asycn_iteration))
    print('Synchronous: {} iterations'.format(sync_iteration))

    plt.savefig('../images/figure_4_1.png')
    plt.close()


if __name__ == '__main__':
    figure_4_1()
