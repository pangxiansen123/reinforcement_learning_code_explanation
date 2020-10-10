#######################################################################
# Copyright (C)                                                       #
# 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top
# 轨迹采样                                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# 2 actions
ACTIONS = [0, 1]

# each transition has a probability to terminate with 0
# 所有状态中，执行动作跳到终止状态结束整幕交互的概率为0.1
# 意思就是每一次执行一个动作都有0.1的概率跳转到终止状态
TERMINATION_PROB = 0.1

# maximum expected updates
MAX_STEPS = 2000

# epsilon greedy for behavior policy
# epsilon贪心策略
EPSILON = 0.1

# break tie randomly
# 选择一个最大值
def argmax(value):
    max_q = np.max(value)
    return np.random.choice([a for a, q in enumerate(value) if q == max_q])
#  有多少个状态，每个状态有多少个分支
class Task():
    # @n_states: number of non-terminal states 不是终止状态的数量，所以这里n_states就是终止状态，因为是从0开始的
    # @b: branch 每个状态下面有多少个分支
    # Each episode starts with state 0, and state n_states is a terminal state
    def __init__(self, n_states, b):
        self.n_states = n_states
        self.b = b

        # transition matrix, each state-action pair leads to b possible states
        # 2 actions
        # 返回从0到n_states，输出随机数的尺寸n_states*len(ACTIONS)*b
        # 每个动作都导致b种后寄状态。
        # 是一个矩阵
        self.transition = np.random.randint(n_states, size=(n_states, len(ACTIONS), b))

        # it is not clear how to set the reward, I use a unit normal distribution here
        # reward is determined by (s, a, s')
        # 通过本函数可以返回一个或一组服从标准正态分布的随机样本值。
        # 正态分布的数组，维度为n_states*len(ACTIONS)*b
        # 这个矩阵的意思就是某个状态执行某个动作，达到某个状态的即时回报
        self.reward = np.random.randn(n_states, len(ACTIONS), b)
    # 返回的就是下个状态所对应的下标（是一个数），也就是0-n_states中间的值。和到达这个状态所对应的即时回报值
    def step(self, state, action):
        if np.random.rand() < TERMINATION_PROB:
            return self.n_states, 0
        next = np.random.randint(self.b)
        # 返回的就是下个状态所对应的下边，也就是0-n_states中间的值
        return self.transition[state, action, next], self.reward[state, action, next]

# Evaluate the value of the start state for the greedy policy
# derived from @q under the MDP @task
# q[state, action]的值就是一个1*1的
# 这个就是根据迭代好的q值进行一次评估，看看回报值是多少
def evaluate_pi(q, task):
    # use Monte Carlo method to estimate the state value
    runs = 10
    returns = []
    for r in range(runs):
        rewards = 0
        state = 0
        while state < task.n_states:
            action = argmax(q[state])
            state, r = task.step(state, action)
            rewards += r
        returns.append(rewards)
    return np.mean(returns)

# perform expected update from a uniform state-action distribution of the MDP @task
# evaluate the learned q value every @eval_interval steps
# 不知道这个到底有什么含义，这个就是图中说的随机均匀分配算力
def uniform(task, eval_interval):
    performance = []
    q = np.zeros((task.n_states, 2))
    for step in tqdm(range(MAX_STEPS)):
        # 向下取整，在去余数
        state = step // len(ACTIONS) % task.n_states
        # 去余数，要么是0要么是1
        action = step % len(ACTIONS)

        next_states = task.transition[state, action]
        q[state, action] = (1 - TERMINATION_PROB) * np.mean(
            task.reward[state, action] + np.max(q[next_states, :], axis=1))

        if step % eval_interval == 0:
            v_pi = evaluate_pi(q, task)
            performance.append([step, v_pi])

    return zip(*performance)

# perform expected update from an on-policy distribution of the MDP @task
# evaluate the learned q value every @eval_interval steps
# 有所疑问，这个不是异步策略吗？
def on_policy(task, eval_interval):
    performance = []
    # Q(S,A),其中a有两个动作，所以这里面的矩阵为n_states*2
    q = np.zeros((task.n_states, 2))
    # 初始的下标是0
    state = 0
    for step in tqdm(range(MAX_STEPS)):
        # 先根据epsilon产生一个动作
        if np.random.rand() < EPSILON:
            action = np.random.choice(ACTIONS)
        else:
            action = argmax(q[state])
        # 下一个动作
        next_state, _ = task.step(state, action)
        # 执行这个动作可以导致b个后继状态，所以next_states的维度是b*1
        next_states = task.transition[state, action]
        # task.reward维度是n_states*len(ACTIONS)*b 那么task.reward[state, action]这个就是b*1
        # q的维度是n_states*2 q[next_states, :]就是b*1，因为next_states就是b个
        # 所以q[state, action]的值就是一个1*1的
        q[state, action] = (1 - TERMINATION_PROB) * np.mean(
            task.reward[state, action] + np.max(q[next_states, :], axis=1))
        # aaaa = task.reward[state, action]
        # baaa = np.max(q[next_states, :], axis=1)
        # cccc = np.mean(task.reward[state, action] + np.max(q[next_states, :], axis=1))
        # 就是走到的终止状态，那么就回到原处
        if next_state == task.n_states:
            next_state = 0
        state = next_state

        if step % eval_interval == 0:
            v_pi = evaluate_pi(q, task)
            performance.append([step, v_pi])
    # 打包成一个一个元组，然后弄成一个列表
    return zip(*performance)

def figure_8_8():
    # 状态的数量
    num_states = [100, 1000]
    branch = [1, 3, 10]
    methods = [on_policy, uniform]

    # average accross 30 tasks
    n_tasks = 10

    # number of evaluation points
    x_ticks = 200

    #plt.figure(figsize=(10, 20))
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，
    for i, n in enumerate(num_states):
        plt.subplot(2, 1, i+1)
        for b in branch:
            # 30个任务，就有30个Task(n, b)
            tasks = [Task(n, b) for _ in range(n_tasks)]
            for method in methods:
                value = []
                for task in tasks:
                    # eval_interval就是MAX_STEPS / x_ticks=20000/100，也就是200次计算一下最优的G
                    # 这个steps就是x_ticks*1 同样v也是
                    steps, v = method(task, MAX_STEPS / x_ticks)
                    value.append(v)
                value = np.mean(np.asarray(value), axis=0)
                plt.plot(steps, value, label='b = %d, %s' % (b, method.__name__))
        plt.title('%d states' % (n))

        plt.ylabel('value of start state')
        plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('computation time, in expected updates')

    plt.savefig('../images/figure_8_8.png')
    plt.close()

if __name__ == '__main__':
    figure_8_8()
