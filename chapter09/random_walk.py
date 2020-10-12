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

# # of states except for terminal states
N_STATES = 1000

# all states
STATES = np.arange(1, N_STATES + 1)

# start from a central state
START_STATE = 500

# terminal states
END_STATES = [0, N_STATES + 1]

# possible actions
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

# maximum stride for an action
STEP_RANGE = 100

# 这个函数的作用就是就算真正的价值函数 返回true_value这是一个数组
def compute_true_value():
    # true state value, just a promising guess
    # 价值尺度，到达左边是-1，到达右边是+1
    true_value = np.arange(-1001, 1003, 2) / 1001.0

    # Dynamic programming to find the true state values, based on the promising guess above
    # Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states
    while True:
        old_value = np.copy(true_value)
        for state in STATES:
            true_value[state] = 0
            # 选择动作 左边是-1，右边是+1
            for action in ACTIONS:
                # 就是文中说的：从当前位置向一侧100（1到100）个邻居中的任意一个位置移动
                for step in range(1, STEP_RANGE + 1):
                    step *= action
                    next_state = state + step
                    # 这个就是把状态值设置在左右边界上，这个语句很巧妙
                    next_state = max(min(next_state, N_STATES + 1), 0)
                    # asynchronous update for faster convergence
                    # 异步更新更快的收敛速度
                    # 1.0 / (2 * STEP_RANGE)这个应该是概率，1.0/STEP_RANGE就是选择1-100的概率，而2就是选择左边还是右边
                    true_value[state] += 1.0 / (2 * STEP_RANGE) * true_value[next_state]
        error = np.sum(np.abs(old_value - true_value))
        if error < 1e-2:
            break
    # correct the state value for terminal states to 0
    # 这个价值设置为0是什么意思
    true_value[0] = true_value[-1] = 0

    return true_value

# take an @action at @state, return new state and reward for this transition
# 从当前状态执行动作，返回下一个状态和即时回报
def step(state, action):
    # 在1-100之间随机产生一个数字，就是运动到的步长
    step = np.random.randint(1, STEP_RANGE + 1)
    step *= action
    state += step
    state = max(min(state, N_STATES + 1), 0)
    if state == 0:
        reward = -1
    elif state == N_STATES + 1:
        reward = 1
    else:
        reward = 0
    return state, reward

# get an action, following random policy
def get_action():
    if np.random.binomial(1, 0.5) == 1:
        return 1
    return -1

# a wrapper class for aggregation value function
# 聚合类的创建，创建一个聚合所用的数组
# 这里面的w就是十个，就是将一千个状态进行状态聚类，得到十个参数，方便了计算
class ValueFunction:
    # @num_of_groups: # of aggregations
    # num_of_groups就是组的数量
    def __init__(self, num_of_groups):
        self.num_of_groups = num_of_groups
        # 组的长度，这个是100
        self.group_size = N_STATES // num_of_groups

        # thetas
        # 这个就是状态聚合中所说的对于St所在的组为1，对于其他组为0
        self.params = np.zeros(num_of_groups)

    # get the value of @state
    def value(self, state):
        if state in END_STATES:
            return 0
        # 去模数，这里为什么减去1呢？因为1-100是第0组，所以100这个应该减去1，然后在除以100，得到的模数就是0
        group_index = (state - 1) // self.group_size
        return self.params[group_index]

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    def update(self, delta, state):
        group_index = (state - 1) // self.group_size
        # 这里要注意，这个就是文中说的梯度在对应的组为1，对于其他组为0
        # 所以这里面就不用加入梯度了
        # 只要是求w的，不管是MC还是TD都是这样
        self.params[group_index] += delta

# a wrapper class for tile coding value function
# 瓦片编码的类
class TilingsValueFunction:
    # @num_of_tilings: # of tilings 覆盖的数量
    # @tileWidth: each tiling has several tiles, this parameter specifies the width of each tile 瓦片的宽度
    # @tilingOffset: specifies how tilings are put together 覆盖的偏移量
    def __init__(self, numOfTilings, tileWidth, tilingOffset):
        # 覆盖的数量
        self.numOfTilings = numOfTilings
        # 瓦片宽度 每个瓦片包含200个状态,意思就是每个覆盖有五个瓦片
        self.tileWidth = tileWidth # 200
        # 偏移量
        self.tilingOffset = tilingOffset

        # To make sure that each sate is covered by same number of tiles,
        # we need one more tile for each tiling
        # //就是向下取整
        self.tilingSize = N_STATES // tileWidth + 1 # 这个是6,(N_STATES // tileWidth) + 1

        # weight for each tile
        # 每个覆盖,每个瓦片的权重,50个覆盖,每个覆盖6个瓦片
        self.params = np.zeros((self.numOfTilings, self.tilingSize))

        # For performance, only track the starting position for each tiling
        # As we have one more tile for each tiling, the starting position will be negative
        # 因为我们为每个覆盖多了一个瓦片(之前是5,现在成了6了)，所以起始位置将是负的
        # 这里就生成了五十个覆盖
        self.tilings = np.arange(-tileWidth + 1, 0, tilingOffset)

    # get the value of @state
    def value(self, state):
        stateValue = 0.0
        # go through all the tilings
        # 遍历所有的覆盖
        for tilingIndex in range(0, len(self.tilings)):
            # find the active tile in current tiling
            # 查找每个覆盖下这个状态所对应的瓦片
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            stateValue += self.params[tilingIndex, tileIndex]
        return stateValue

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    def update(self, delta, state):

        # each state is covered by same number of tilings
        # so the delta should be divided equally into each tiling (tile)
        # 这个就是书上说的/50,对于状态聚类,只有一个覆盖,而对于瓦片覆盖,这里面有50个覆盖,所以这里要除以50
        delta /= self.numOfTilings

        # go through all the tilings
        # 遍历所有的覆盖
        for tilingIndex in range(0, len(self.tilings)):
            # find the active tile in current tiling
            # 找到对应的瓦片
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            self.params[tilingIndex, tileIndex] += delta

# a wrapper class for polynomial / Fourier -based value function
# 这个就是一个类，就是多项式和傅里叶基的整合

# 如果是多项式基，标识符就是0
POLYNOMIAL_BASES = 0
# 如果是傅里叶基，标识符就是1
FOURIER_BASES = 1
class BasesValueFunction:
    # @order: # of bases, each function also has one more constant parameter (called bias in machine learning)
    # @type: polynomial bases or Fourier bases
    def __init__(self, order, type):
        self.order = order
        # order这里可以去[5 10 20]，这个就是这个函数是几级的
        # 这个就是级数所对应的权重
        self.weights = np.zeros(order + 1)

        # set up bases function
        self.bases = []
        if type == POLYNOMIAL_BASES:
            for i in range(0, order + 1):
                # g = lambda x : x**2 这个就是相当于g(x)=x**2
                # 这个真的绕，lambda重新设置两个变量，分别是si，并且i是被赋值了，是把for里面的i赋值给i，然后函数表达式就是s**i，这个是一个匿名函数
                # 这个意思就是生成一个数组，数组内容是[1 s s^2 s^3 s^4 s^5]
                self.bases.append(lambda s, i=i: pow(s, i))
        elif type == FOURIER_BASES:
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: np.cos(i * np.pi * s))

    # get the value of @state
    # 只是一个状态，不是所有的状态
    def value(self, state):
        # map the state space into [0, 1]
        # 把值映射到[0 1]
        state /= float(N_STATES)
        # get the feature vector
        feature = np.asarray([func(state) for func in self.bases])
        # 返回两个数的点积
        return np.dot(self.weights, feature)

    def update(self, delta, state):
        # map the state space into [0, 1]
        state /= float(N_STATES)
        # get derivative value
        derivative_value = np.asarray([func(state) for func in self.bases])
        # 这里面使用的就是线性方法中的对V求导，求导后就是X（s），也就是derivative_value所求的值
        self.weights += delta * derivative_value

# gradient Monte Carlo algorithm
# 梯度的蒙特卡洛算法
# @value_function: an instance of class ValueFunction ValueFunction就是一个变量，其可以调用类
# @alpha: step size
# @distribution: array to store the distribution statistics
def gradient_monte_carlo(value_function, alpha, distribution=None):
    state = START_STATE # 是从五百开始的
    trajectory = [state] # 轨迹

    # We assume gamma = 1, so return is just the same as the latest reward
    # 其实这个就是G，因为中间的回报都是0，只有终止状态的价值是-1或1，而且montecarlo就是从终止状态进行累加的当前状态的，所以这里的reward就是G
    reward = 0.0
    # 这个就是montecarlo，必须执行完一幕才可以进行下一步
    while state not in END_STATES:
        # 要么是+1要么是-1，概率都是0.5
        action = get_action()
        # 从当前状态执行动作，返回下一个状态和即时回报
        next_state, reward = step(state, action)
        # 轨迹的添加
        trajectory.append(next_state)
        state = next_state
    # print(trajectory)
    # Gradient update for each state in this trajectory
    # 更新轨迹中出现的每一个状态
    for state in trajectory[:-1]:
        # 这个就是梯度MonteCarlo算法中的alpha*（Gt-v（St，w））
        # 这里的即时回报，其实就是G，因为中间过程是0
        delta = alpha * (reward - value_function.value(state))
        value_function.update(delta, state)
        # 对出现过得状态进行加1，就是记录出现的次数，这个是显示出现的次数，也就是分布
        if distribution is not None:
            distribution[state] += 1

# semi-gradient n-step TD algorithm
# @valueFunction: an instance of class ValueFunction
# @n: # of steps TD(n)中的n
# @alpha: step size 就是步长
def semi_gradient_temporal_difference(value_function, n, alpha):
    # initial starting state
    state = START_STATE

    # arrays to store states and rewards for an episode
    # space isn't a major consideration, so I didn't use the mod trick
    states = [state]
    rewards = [0]

    # track the time
    time = 0

    # the length of this episode
    T = float('inf')
    while True:
        # go to next time step
        time += 1

        if time < T:
            # choose an action randomly
            action = get_action()
            next_state, reward = step(state, action)

            # store new state and new reward
            states.append(next_state)
            rewards.append(reward)

            if next_state in END_STATES:
                T = time

        # get the time of the state to update
        # 这个就是伪代码中的t-n+1,这里面为啥没有+1？，伪代码中下面是t+1-n<T所以要加1，而这里直接是<=T所以也就不用加1了
        # 还有就是这里的回报值就是即时的t时刻就是t时刻的，书本上的是t时刻的状态所行动的动作的回报值是t+1的，如果加上一就会出现下边索引超出范围的错误
        update_time = time - n
        if update_time >= 0:
            returns = 0.0
            # calculate corresponding rewards
            # 计算G（t:t+n-1）
            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += rewards[t]
            # add state value to the return
            # 当update_time + n > T 时，即时回报值就不会增加
            if update_time + n <= T:
                returns += value_function.value(states[update_time + n])
            state_to_update = states[update_time]
            # update the value function
            if not state_to_update in END_STATES:
                delta = alpha * (returns - value_function.value(state_to_update))
                value_function.update(delta, state_to_update)
        if update_time == T - 1:
            break
        state = next_state

# Figure 9.1, gradient Monte Carlo algorithm
def figure_9_1(true_value):
    # 幕数为100000
    episodes = int(1e5)
    # 步长alpha是书中所示
    alpha = 2e-5

    # we have 10 aggregations in this example, each has 100 states
    # 我们会有十个分组，每个分组会有100个状态
    # num_of_groups 这个就是10的变量
    # 创建一个聚合类的数组
    value_function = ValueFunction(10)
    distribution = np.zeros(N_STATES + 2)
    for ep in tqdm(range(episodes)):
        gradient_monte_carlo(value_function, alpha, distribution)

    distribution /= np.sum(distribution)
    state_values = [value_function.value(i) for i in STATES]

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plt.plot(STATES, state_values, label='Approximate MC value')
    plt.plot(STATES, true_value[1: -1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(STATES, distribution[1: -1], label='State distribution')
    plt.xlabel('State')
    plt.ylabel('Distribution')
    plt.legend()

    plt.savefig('../images/figure_9_1.png')
    plt.close()

# semi-gradient TD on 1000-state random walk
def figure_9_2_left(true_value):
    episodes = int(1e5)
    alpha = 2e-4
    value_function = ValueFunction(10)
    for ep in tqdm(range(episodes)):
        semi_gradient_temporal_difference(value_function, 1, alpha)

    stateValues = [value_function.value(i) for i in STATES]
    plt.plot(STATES, stateValues, label='Approximate TD value')
    plt.plot(STATES, true_value[1: -1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()

# different alphas and steps for semi-gradient TD
def figure_9_2_right(true_value):
    # all possible steps
    # n-TD中的n，从1 2 4 ----- 512 10个
    steps = np.power(2, np.arange(0, 10))

    # all possible alphas
    # 步长 [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1] 11个
    alphas = np.arange(0, 1.1, 0.1)

    # each run has 10 episodes
    episodes = 10

    # perform 100 independent runs
    runs = 100

    # track the errors for each (step, alpha) combination
    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(runs)):
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        for step_ind, step in zip(range(len(steps)), steps):
            for alpha_ind, alpha in zip(range(len(alphas)), alphas):
                # we have 20 aggregations in this example
                # 这里创建20个组
                value_function = ValueFunction(20)
                # 进行10幕实验，得到误差的平均值
                for ep in range(0, episodes):
                    semi_gradient_temporal_difference(value_function, step, alpha)
                    # calculate the RMS error
                    # 每50个的值是相同的
                    state_value = np.asarray([value_function.value(i) for i in STATES])
                    # 总误差除以总的状态数目
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(state_value - true_value[1: -1], 2)) / N_STATES)
    # take average
    errors /= episodes * runs
    # truncate the error
    for i in range(len(steps)):
        plt.plot(alphas, errors[i, :], label='n = ' + str(steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()

def figure_9_2(true_value):
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    figure_9_2_left(true_value)
    plt.subplot(2, 1, 2)
    figure_9_2_right(true_value)

    plt.savefig('../images/figure_9_2.png')
    plt.close()

# Figure 9.5, Fourier basis and polynomials
def figure_9_5(true_value):
    # my machine can only afford 1 run
    runs = 1

    episodes = 5000

    # # of bases
    orders = [5, 10, 20]

    alphas = [1e-4, 5e-5]
    labels = [['polynomial basis'] * 3, ['fourier basis'] * 3]

    # track errors for each episode
    errors = np.zeros((len(alphas), len(orders), episodes))
    for run in range(runs):
        # 基函数的级数
        for i in range(len(orders)):
            value_functions = [BasesValueFunction(orders[i], POLYNOMIAL_BASES), BasesValueFunction(orders[i], FOURIER_BASES)]
            # 使用哪个函数，是多项式还是傅里叶
            for j in range(len(value_functions)):
                # 然后进行每幕计算，计算5000幕
                for episode in tqdm(range(episodes)):

                    # gradient Monte Carlo algorithm 使用GMC进行计算
                    # 练习中，不同的基函数，步长是不一样的，多项式的步长是1e-4
                    # 这个就是已经对里面的权重学习完了
                    gradient_monte_carlo(value_functions[j], alphas[j])

                    # get state values under current value function
                    # 得到每一个状态所对应的value
                    state_values = [value_functions[j].value(state) for state in STATES]

                    # get the root-mean-squared error
                    # [1: -1]这个意思就是从开始到末尾，为什么不是最后一个么，因为是到不了最后一个的
                    errors[j, i, episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))

    # average over independent runs
    errors /= runs

    for i in range(len(alphas)):
        for j in range(len(orders)):
            plt.plot(errors[i, j, :], label='%s order = %d' % (labels[i][j], orders[j]))
    plt.xlabel('Episodes')
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('../images/figure_9_5.png')
    plt.close()

# Figure 9.10, it will take quite a while
def figure_9_10(true_value):

    # My machine can only afford one run, thus the curve isn't so smooth
    runs = 1

    # number of episodes
    episodes = 5000
    # 50个覆盖,这里说的是覆盖,单词却是瓦片
    num_of_tilings = 50

    # each tile will cover 200 states
    # 每个瓦片包含200个状态,意思就是每个覆盖有五个瓦片
    tile_width = 200

    # how to put so many tilings
    # 多重覆盖彼此之间的偏移量是4,这里面的4就是状态量,意思就是偏移量是每个瓦片的五十分之一
    tiling_offset = 4

    labels = ['tile coding (50 tilings)', 'state aggregation (one tiling)']

    # track errors for each episode
    errors = np.zeros((len(labels), episodes))
    for run in range(runs):
        # initialize value functions for multiple tilings and single tiling
        value_functions = [TilingsValueFunction(num_of_tilings, tile_width, tiling_offset),
                         ValueFunction(N_STATES // tile_width)]
        for i in range(len(value_functions)):
            for episode in tqdm(range(episodes)):
                # I use a changing alpha according to the episode instead of a small fixed alpha
                # With a small fixed alpha, I don't think 5000 episodes is enough for so many
                # parameters in multiple tilings.
                # The asymptotic performance for single tiling stays unchanged under a changing alpha,
                # however the asymptotic performance for multiple tilings improves significantly
                # alpha = 1.0 / (episode + 1)
                alpha = 0.0001
                # gradient Monte Carlo algorithm
                gradient_monte_carlo(value_functions[i], alpha)

                # get state values under current value function
                state_values = [value_functions[i].value(state) for state in STATES]

                # get the root-mean-squared error
                errors[i][episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))

    # average over independent runs
    errors /= runs

    for i in range(0, len(labels)):
        plt.plot(errors[i], label=labels[i])
    plt.xlabel('Episodes')
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('../images/figure_9_10.png')
    plt.close()

if __name__ == '__main__':

    # 计算的是真值，就是各个状态真正的值
    true_value = compute_true_value()

    # figure_9_1(true_value)
    # figure_9_2(true_value)
    # 图5和前面的区别就是前面是吧状态或者状态聚类作为参数变量的，也就是为什么求导数的时候会出现1的情况，
    # 而图五使用的就不是状态聚类作为的变量，而是把所有的状态作为一个变量，然后级数增大，变量数目就会增大，
    # 这样他的求导就是函数本身，这个就是线性方法的优点
    # figure_9_5(true_value)
    figure_9_10(true_value)
