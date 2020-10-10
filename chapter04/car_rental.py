#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

matplotlib.use('Agg')

# maximum # of cars in each location
MAX_CARS = 20

# maximum # of cars to move during night
MAX_MOVE_OF_CARS = 5


# 泊松分布的参数λ是单位时间(这里就是天)内随机事件的平均发生次数,泊松分布的期望和方差均为λ


# expectation for rental requests in first location
# 对第一个位置的租赁请求的期望 租车期望
RENTAL_REQUEST_FIRST_LOC = 3
# expectation for rental requests in second location
RENTAL_REQUEST_SECOND_LOC = 4

# 还车期望
# expectation for # of cars returned in first location
RETURNS_FIRST_LOC = 3
# expectation for # of cars returned in second location
RETURNS_SECOND_LOC = 2

# 折损系数
DISCOUNT = 0.9

# credit earned by a car 回报
RENTAL_CREDIT = 10

# cost of moving a car 调车费用
MOVE_CAR_COST = 2

# all possible actions
# 借车和还车是概率问题，能够控制的动作只有移动车辆，从第一个地点移除车辆为正，移入车辆为负
# actions = [-5 5]
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)

# An up bound for poisson distribution 泊松分布的一个上界
# If n is greater than this value, then the probability of getting n is truncated to 0
# 如果n大于这个值，那么得到n的概率被截断为0
POISSON_UPPER_BOUND = 11

# Probability for poisson distribution 泊松分布的概率
# @lam: lambda should be less than 10 for this function
# 泊松分布的字典{状态（数量*10+λ）：概率}
poisson_cache = dict() #字典


# 返回n, lam所对应的概率
def poisson_probability(n, lam):
    global poisson_cache
    # 这里是一个技巧，也可以用别的方法，这个是为了防止重复，如果用n+λ，假设第二个地点租0辆车，n+λ就是4;第二个地点还车2,n+λ就是2+2,他们就是一样的，这样就会产生奇异
    key = n * 10 + lam # 这里的lamda就是lam
    # 超出范围也设置概率
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    # 返回n, lam所对应的概率
    return poisson_cache[key]

# 根据给定策略进行策略评估
# 根据当前状态、动作（移动车辆）、上一个时刻价值函数、返还车辆是否为固定值;返回这个时刻的这个状态这个动作的回报值
def expected_return(state, action, state_value, constant_returned_cars):
    """
    @state: [# of cars in first location, # of cars in second location]
    @action: positive+ if moving cars from first location to second location,
            negative- if moving cars from second location to first location
    @stateValue: state value matrix
    @constant_returned_cars:  if set True, model is simplified such that
    the # of cars returned in daytime becomes constant
    rather than a random value from poisson distribution, which will reduce calculation time
    and leave the optimal policy/value state matrix almost the same
    """
    # initailize total return
    returns = 0.0

    # cost for moving cars
    # 只要移动车辆 就是扣钱
    returns -= MOVE_CAR_COST * abs(action)

    # moving cars 就是移动车辆后，两个地方的车的数量
    NUM_OF_CARS_FIRST_LOC = min(state[0] - action, MAX_CARS)
    NUM_OF_CARS_SECOND_LOC = min(state[1] + action, MAX_CARS)

    # go through all possible rental requests
    # 产生11个参数，就是0---10
    # 就是遍历下一个状态，就是可能到达的所有状态
    #  遍历租用的所有车的数量
    for rental_request_first_loc in range(POISSON_UPPER_BOUND):
        for rental_request_second_loc in range(POISSON_UPPER_BOUND):
            # probability for current combination of rental requests
            prob = poisson_probability(rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC) * \
                poisson_probability(rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC)

            #就是存储现在两个地点应该有的车的数目
            num_of_cars_first_loc = NUM_OF_CARS_FIRST_LOC
            num_of_cars_second_loc = NUM_OF_CARS_SECOND_LOC

            # valid rental requests should be less than actual # of cars
            # 考虑到租车请求在车辆不足的情况下无法被满足，因而使用有效租车数来记录实际租出去的数量
            valid_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)
            valid_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)

            # get credits for renting
            reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT
            num_of_cars_first_loc -= valid_rental_first_loc
            num_of_cars_second_loc -= valid_rental_second_loc
            # 还车不扣钱
            if constant_returned_cars: #如果采用定值还车的方法
                # get returned cars, those cars can be used for renting tomorrow
                # 得到还车的期望 就是每天还车的数量
                # 泊松分布的参数λ是单位时间(这里就是天)内随机事件的平均发生次数,泊松分布的期望和方差均为λ
                returned_cars_first_loc = RETURNS_FIRST_LOC
                returned_cars_second_loc = RETURNS_SECOND_LOC #两个车场每天收到的还车数量为定值
                #还完车后，车的数量
                num_of_cars_first_loc = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                num_of_cars_second_loc = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                # 根据书本P79迭代策略评估中策略评估部分V(s)更新部分公式，这只是求和式内的一条，所有循环结束后的returns为求和式的总值
                returns += prob * (reward + DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])
            else:
                #还车
                for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
                    for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
                        prob_return = poisson_probability(
                            returned_cars_first_loc, RETURNS_FIRST_LOC) * poisson_probability(returned_cars_second_loc, RETURNS_SECOND_LOC)
                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                        num_of_cars_second_loc_ = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                        prob_ = prob_return * prob #还车数量进行遍历
                        returns += prob_ * (reward + DISCOUNT *
                                            state_value[num_of_cars_first_loc_, num_of_cars_second_loc_])
    return returns


def figure_4_2(constant_returned_cars=True):
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros(value.shape, dtype=np.int)

    iterations = 0
    _, axes = plt.subplots(2, 4, figsize=(40, 20))
    # wspace、hspace分别表示子图之间左右、上下的间距
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)

        # policy evaluation (in-place)
        # 策略评估 价值函数稳定到一个值
        while True:
            old_value = value.copy()
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    new_state_value = expected_return([i, j], policy[i, j], value, constant_returned_cars)
                    value[i, j] = new_state_value
            max_value_change = abs(old_value - value).max()
            print('max value change {}'.format(max_value_change))
            if max_value_change < 1e-4:
                break

        # policy improvement
        # 策略改进
        policy_stable = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action = policy[i, j]
                action_returns = []
                # actions = [-5 5]
                # 遍历所有的动作，然后下面找到最大值
                for action in actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(expected_return([i, j], action, value, constant_returned_cars))
                    else:
                        action_returns.append(-np.inf)
                new_action = actions[np.argmax(action_returns)]
                policy[i, j] = new_action
                # 收敛到一个动作，就是每一个动作都进行找最优
                if policy_stable and old_action != new_action:
                    policy_stable = False
        print('policy stable {}'.format(policy_stable))

        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[5])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('policy {}'.format(5), fontsize=30)
            break

        iterations += 1

    plt.savefig('../images/figure_4_2.png')
    plt.close()


if __name__ == '__main__':
    figure_4_2()
