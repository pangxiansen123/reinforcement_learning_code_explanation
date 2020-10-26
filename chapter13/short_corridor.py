#######################################################################
# Copyright (C)                                                       #
# 2018 Sergii Bondariev (sergeybondarev@gmail.com)                    #
# 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# 这个函数是算真值的，但是不知道怎么算的
def true_value(p):
    """ True value of the first state
    Args:
        p (float): probability of the action 'right'.
    Returns:
        True value of the first state.
        The expression is obtained by manually solving the easy linear system
        of Bellman equations using known dynamics.
        该表达式是通过手动求解简单的线性系统的Bellman方程使用已知的动力学。
    """
    return (2 * p - 4) / (p * (1 - p))

# 小型走廊环境，根据状态和动作得到回报和是否到达目标点
class ShortCorridor:
    """
    Short corridor environment, see Example 13.1
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 0
    # go_right是bool值
    def step(self, go_right):
        """
        Args:
            go_right (bool): chosen action
        Returns:
            tuple of (reward, episode terminated?)
        """
        if self.state == 0 or self.state == 2:
            if go_right:
                self.state += 1
            else:
                self.state = max(0, self.state - 1)
        else:
            # 就是到达第1个状态，运动方向是反的
            if go_right:
                self.state -= 1
            else:
                self.state += 1
        # 判断下一个状态在哪里
        if self.state == 3:
            # terminal state
            return 0, True
        else:
            return -1, False

def softmax(x):
    t = np.exp(x - np.max(x))
    return t / np.sum(t)

class ReinforceAgent:
    """
    ReinforceAgent that follows algorithm
    'REINFORNCE Monte-Carlo Policy-Gradient Control (episodic)'
    """
    def __init__(self, alpha, gamma):
        # set values such that initial conditions correspond to left-epsilon greedy
        # theta是一个矩阵
        self.theta = np.array([-1.47, 1.47])
        self.alpha = alpha
        self.gamma = gamma
        # first column - left, second - right
        self.x = np.array([[0, 1],
                           [1, 0]])
        self.rewards = []
        self.actions = []

    # 这个函数的作用就是得到策略pi
    def get_pi(self):
        # 这个就是P318中的求h的点积
        h = np.dot(self.theta, self.x)
        # 这里h后面为什么会减去一个这个，不太懂
        t = np.exp(h - np.max(h))
        pmf = t / np.sum(t)
        # never become deterministic,不会变成确定性，
        # guarantees episode finish
        imin = np.argmin(pmf)
        epsilon = 0.05

        if pmf[imin] < epsilon:
            pmf[:] = 1 - epsilon
            pmf[imin] = epsilon

        return pmf

    def get_p_right(self):
        return self.get_pi()[1]
    # 这个是选择一个动作
    def choose_action(self, reward):
        if reward is not None:
            self.rewards.append(reward)

        pmf = self.get_pi()
        # 根据随机概率产生一个动作
        go_right = np.random.uniform() <= pmf[1]
        # 加入的是bool值
        self.actions.append(go_right)

        return go_right

    # 如果这一幕已经结束了，那么我们要做的就是对这一幕的策略进行学习得到最新的策略theta
    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        # 从倒数一个个元素开始赋值
        G[-1] = self.rewards[-1]


        # 计算每个时刻的G值
        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        # 开始对策略进行学习
        for i in range(len(G)):
            # 根据动作，给j赋值
            j = 1 if self.actions[i] else 0
            # 这个函数的作用就是得到策略pi
            pmf = self.get_pi()
            # 这个就是P324的练习，线性动作偏好的偏导数
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
            # 这个i是特定时刻的
            update = self.alpha * gamma_pow * G[i] * grad_ln_pi

            self.theta += update
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []

class ReinforceBaselineAgent(ReinforceAgent):
    def __init__(self, alpha, gamma, alpha_w):
        super(ReinforceBaselineAgent, self).__init__(alpha, gamma)
        self.alpha_w = alpha_w
        self.w = 0

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        for i in range(len(G)):
            self.w += self.alpha_w * gamma_pow * (G[i] - self.w)

            j = 1 if self.actions[i] else 0
            pmf = self.get_pi()
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * gamma_pow * (G[i] - self.w) * grad_ln_pi

            self.theta += update
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []

# 运行num_episodes幕，然后返回每幕的回报（其是一个矩阵num_episodes*1，回报是进行学习后的回报）
def trial(num_episodes, agent_generator):
    # 创建一个小型走廊环境
    env = ShortCorridor()
    # 创建一个策略学习
    agent = agent_generator()
    # 每一幕的回报，以为是MC方法，所以必须考虑到终止状态
    rewards = np.zeros(num_episodes)
    for episode_idx in range(num_episodes):
        rewards_sum = 0
        reward = None
        # 设置初始化的状态
        env.reset()
        # 运行一幕，并且得到这一幕的G值
        while True:
            # 这个就是上一个时刻的回报，所以这里应该注意，到达最后一个状态的时候是不会把最后一个回报放进去的
            # 因为他不会执行这个函数了，所以会进行agent.episode_end(reward)这个方法
            go_right = agent.choose_action(reward)
            # 小型走廊环境，根据状态和动作得到回报和是否到达目标点
            reward, episode_end = env.step(go_right)
            rewards_sum += reward
            # 如果到达终点，要做的就是计算G
            if episode_end:
                # 如果这一幕已经结束了，那么我们要做的就是对这一幕的策略进行学习得到最新的策略theta
                agent.episode_end(reward)
                break
        # 注意这里的rewards与agent中rewards是不一样的，这里的记录的是这一幕的回报，而agent中rewards记录的是每一个状态的回报
        rewards[episode_idx] = rewards_sum

    return rewards

def example_13_1():
    # epsilon贪心策略的值为0.05
    epsilon = 0.05
    fig, ax = plt.subplots(1, 1)

    # Plot a graph
    p = np.linspace(0.01, 0.99, 100)
    # 计算各个概率下的真值
    y = true_value(p)
    ax.plot(p, y, color='red')

    # Find a maximum point, can also be done analytically by taking a derivative
    imax = np.argmax(y)
    pmax = p[imax]
    ymax = y[imax]
    ax.plot(pmax, ymax, color='green', marker="*", label="optimal point: f({0:.2f}) = {1:.2f}".format(pmax, ymax))

    # Plot points of two epsilon-greedy policies
    # magenta紫红色
    ax.plot(epsilon, true_value(epsilon), color='magenta', marker="o", label="epsilon-greedy left")
    ax.plot(1 - epsilon, true_value(1 - epsilon), color='blue', marker="o", label="epsilon-greedy right")

    ax.set_ylabel("Value of the first state")
    ax.set_xlabel("Probability of the action 'right'")
    ax.set_title("Short corridor with switched actions")
    ax.set_ylim(ymin=-105.0, ymax=5)
    ax.legend()

    plt.savefig('../images/example_13_1.png')
    plt.close()

def figure_13_1():
    num_trials = 100
    # 每次是1000幕
    num_episodes = 1000
    gamma = 1
    # python 使用 lambda 来创建匿名函数。
    agent_generators = [lambda : ReinforceAgent(alpha=2e-4, gamma=gamma),
                        lambda : ReinforceAgent(alpha=2e-5, gamma=gamma),
                        lambda : ReinforceAgent(alpha=2e-3, gamma=gamma)]
    labels = ['alpha = 2e-4',
              'alpha = 2e-5',
              'alpha = 2e-3']

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            # 运行num_episodes幕，然后返回每幕的回报（其是一个矩阵num_episodes*1）
            reward = trial(num_episodes, agent_generator)
            # 一下子放num_episodes幕数据
            rewards[agent_index, i, :] = reward
    # 这条线是理论线
    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='red', label='-11.6')
    for i, label in enumerate(labels):
        # 对num_trials进行求平均
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('../images/figure_13_1.png')

    plt.close()

def figure_13_2():
    num_trials = 100
    num_episodes = 1000
    alpha = 2e-4
    gamma = 1
    agent_generators = [lambda : ReinforceAgent(alpha=alpha, gamma=gamma),
                        lambda : ReinforceBaselineAgent(alpha=alpha*10, gamma=gamma, alpha_w=alpha*100)]
    labels = ['Reinforce without baseline',
              'Reinforce with baseline']

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward

    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='red', label='-11.6')
    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('../images/figure_13_2.png')
    plt.close()

if __name__ == '__main__':
    example_13_1()
    figure_13_1()
    # figure_13_2()
