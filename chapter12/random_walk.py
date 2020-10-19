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

# all states
N_STATES = 19

# all states but terminal states
# [1-19]
STATES = np.arange(1, N_STATES + 1)

# start from the middle state
START_STATE = 10

# two terminal states
# an action leading to the left terminal state has reward -1
# an action leading to the right terminal state has reward 1
END_STATES = [0, N_STATES + 1]

# true state values from Bellman equation
# 走到左边的收益是-1,走到右边的收益是1
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
# 终止状态的回报是0
TRUE_VALUE[0] = TRUE_VALUE[N_STATES + 1] = 0.0

# base class for lambda-based algorithms in this chapter
# In this example, we use the simplest linear feature function, state aggregation.
# And we use exact 19 groups, so the weights for each group is exact the value for that state
class ValueFunction:
    # @rate: lambda, as it's a keyword in python, so I call it rate
    # @stepSize: alpha, step size for update
    def __init__(self, rate, step_size):
        self.rate = rate
        self.step_size = step_size
        self.weights = np.zeros(N_STATES + 2)

    # the state value is just the weight
    def value(self, state):
        return self.weights[state]

    # feed the algorithm with new observation
    # derived class should override this function
    def learn(self, state, reward):
        return

    # initialize some variables at the beginning of each episode
    # must be called at the very beginning of each episode
    # derived class should override this function
    def new_episode(self):
        return

# Off-line lambda-return algorithm
class OffLineLambdaReturn(ValueFunction):
    # rate就是权重值,step_size就是步长
    def __init__(self, rate, step_size):
        ValueFunction.__init__(self, rate, step_size)
        # To accelerate learning, set a truncate value for power of lambda
        self.rate_truncate = 1e-3

    def new_episode(self):
        # initialize the trajectory
        # 路径的轨迹
        self.trajectory = [START_STATE]
        # only need to track the last reward in one episode, as all others are 0
        self.reward = 0.0
    # 传入进行的就是S'和Rt+1
    # 因为这里的状态是下一个时刻的状态,所以这里的时间应该减1
    # 这个函数就是将状态放进去trajectory,如果是下一个状态是终止状态,那么进行学习,每一幕都会对w进行学习更新
    def learn(self, state, reward):
        # add the new state to the trajectory
        self.trajectory.append(state)
        if state in END_STATES:
            # start off-line learning once the episode ends
            # 到这里就是结束了一幕,得到回报值
            self.reward = reward
            # 返回结束时间
            # 注意,这里是将END_STATES放进去了,所以这里应该减去1
            # 这个才是真正的时间
            self.T = len(self.trajectory) - 1
            # 离线学习,lambda回报,必须离线下才可以进行的
            self.off_line_learn()

    # get the n-step return from the given time
    # 返回未来n步的累计收益,这个就是计算G
    # 这个函数就是计算Gt:(t+n)的回报
    def n_step_return_from_time(self, n, time):
        # gamma is always 1 and rewards are zero except for the last reward
        # the formula can be simplified
        end_time = min(time + n, self.T)
        # 注意这个就是P284的公式,因为中间的即时回报是0,所以就只考虑v(st+n,wt+n-1)的值,这个值就是self.trajectory[end_time]
        returns = self.value(self.trajectory[end_time])
        if end_time == self.T:
            # 如果time + n>self.T,那么中间肯定会到达终止状态,所以即时回报就得加上R了
            returns += self.reward
        # returns就是返回这个状态所对应的值
        return returns

    # get the lambda-return from the given time
    # 得到lambda-回报
    def lambda_return_from_time(self, time):
        returns = 0.0
        lambda_power = 1
        # 这个是加上(1-lambda)*E((lambda^n-1)Gt:t+n)
        # 到达T后.n就是从1到T-1-t,
        for n in range(1, self.T - time):
            # time就是当前时刻
            # n_step_return_from_time就是返回Gt:(t+n)
            # n就是从1到T-1-t,time就是当前状态所对应的时刻
            returns += lambda_power * self.n_step_return_from_time(n, time)
            lambda_power *= self.rate
            if lambda_power < self.rate_truncate:
                # If the power of lambda has been too small, discard all the following sequences
                break
        returns *= 1 - self.rate
        # 这个就是加上lambda*Gt
        if lambda_power >= self.rate_truncate:
            # 这个就是加上lambda*Gt,为什么是self.reward而不是self.n_step_return_from_time(n, time)呢?
            # 其实这两个一样,这个没有少一个self.rate,因为lambda_power是在计算完returns才计算的,所以到达for结束 ,就是需要的lambda_power
            returns += lambda_power * self.reward
            # returns += lambda_power * self.n_step_return_from_time(self.T - time, time)
        return returns

    # perform off-line learning at the end of an episode
    # 每一幕都会对w进行学习更新
    def off_line_learn(self):
        for time in range(self.T):
            # update for each state in the trajectory
            state = self.trajectory[time]
            # time从0开始
            delta = self.lambda_return_from_time(time) - self.value(state)
            # 梯度还是跟之前一样,会成为1
            delta *= self.step_size
            self.weights[state] += delta

# TD(lambda) algorithm
class TemporalDifferenceLambda(ValueFunction):
    def __init__(self, rate, step_size):
        ValueFunction.__init__(self, rate, step_size)
        self.new_episode()

    def new_episode(self):
        # initialize the eligibility trace
        self.eligibility = np.zeros(N_STATES + 2)
        # initialize the beginning state
        self.last_state = START_STATE

    def learn(self, state, reward):
        # update the eligibility trace and weights
        self.eligibility *= self.rate
        self.eligibility[self.last_state] += 1
        delta = reward + self.value(state) - self.value(self.last_state)
        delta *= self.step_size
        self.weights += delta * self.eligibility
        self.last_state = state

# True online TD(lambda) algorithm
class TrueOnlineTemporalDifferenceLambda(ValueFunction):
    def __init__(self, rate, step_size):
        ValueFunction.__init__(self, rate, step_size)

    def new_episode(self):
        # initialize the eligibility trace
        self.eligibility = np.zeros(N_STATES + 2)
        # initialize the beginning state
        self.last_state = START_STATE
        # initialize the old state value
        self.old_state_value = 0.0

    def learn(self, state, reward):
        # update the eligibility trace and weights
        last_state_value = self.value(self.last_state)
        state_value = self.value(state)
        dutch = 1 - self.step_size * self.rate * self.eligibility[self.last_state]
        self.eligibility *= self.rate
        self.eligibility[self.last_state] += dutch
        delta = reward + state_value - last_state_value
        self.weights += self.step_size * (delta + last_state_value - self.old_state_value) * self.eligibility
        self.weights[self.last_state] -= self.step_size * (last_state_value - self.old_state_value)
        self.old_state_value = state_value
        self.last_state = state

# 19-state random walk
# 随机进行行走,并且每一幕都进行学习
def random_walk(value_function):
    value_function.new_episode()
    state = START_STATE
    while state not in END_STATES:
        next_state = state + np.random.choice([-1, 1])
        if next_state == 0:
            reward = -1
        elif next_state == N_STATES + 1:
            reward = 1
        else:
            reward = 0
        # 这个就是S'和Rt+1
        # 这个函数就是将状态放进去trajectory,如果是下一个状态是终止状态,那么进行学习,每一幕都会对w进行学习更新
        value_function.learn(next_state, reward)
        state = next_state

# general plot framework
# @valueFunctionGenerator: generate an instance of value function
# @runs: specify the number of independent runs
# @lambdas: a series of different lambda values
# @alphas: sequences of step size for each lambda
def parameter_sweep(value_function_generator, runs, lambdas, alphas):
    # play for 10 episodes for each run
    # 运行十幕
    episodes = 10
    # track the rms errors
    # 弄成一个8*10的矩阵
    errors = [np.zeros(len(alphas_)) for alphas_ in alphas]
    # 运行50次
    for run in tqdm(range(runs)):
        # 每一个lambdas对应一行alphas
        for lambdaIndex, rate in enumerate(lambdas):
            for alphaIndex, alpha in enumerate(alphas[lambdaIndex]):
                # rate这个就是权重值,alpha就是步长
                valueFunction = value_function_generator(rate, alpha)
                for episode in range(episodes):
                    # 一幕数据之后结束
                    random_walk(valueFunction)
                    stateValues = [valueFunction.value(state) for state in STATES]
                    errors[lambdaIndex][alphaIndex] += np.sqrt(np.mean(np.power(stateValues - TRUE_VALUE[1: -1], 2)))

    # average over runs and episodes
    for error in errors:
        error /= episodes * runs

    for i in range(len(lambdas)):
        plt.plot(alphas[i], errors[i], label='lambda = ' + str(lambdas[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.legend()

# Figure 12.3: Off-line lambda-return algorithm
def figure_12_3():
    # 权重值
    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    # 步长
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.55, 0.05),
              np.arange(0, 0.22, 0.02),
              np.arange(0, 0.11, 0.01)]
    parameter_sweep(OffLineLambdaReturn, 50, lambdas, alphas)

    plt.savefig('../images/figure_12_3_2.png')
    plt.close()

# Figure 12.6: TD(lambda) algorithm
def figure_12_6():
    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.99, 0.09),
              np.arange(0, 0.55, 0.05),
              np.arange(0, 0.33, 0.03),
              np.arange(0, 0.22, 0.02),
              np.arange(0, 0.11, 0.01),
              np.arange(0, 0.044, 0.004)]
    parameter_sweep(TemporalDifferenceLambda, 50, lambdas, alphas)

    plt.savefig('../images/figure_12_6.png')
    plt.close()

# Figure 12.7: True online TD(lambda) algorithm
def figure_12_8():
    lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = [np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 1.1, 0.1),
              np.arange(0, 0.88, 0.08),
              np.arange(0, 0.44, 0.04),
              np.arange(0, 0.11, 0.01)]
    parameter_sweep(TrueOnlineTemporalDifferenceLambda, 50, lambdas, alphas)

    plt.savefig('../images/figure_12_8.png')
    plt.close()

if __name__ == '__main__':
    figure_12_3()
    # figure_12_6()
    # figure_12_8()
