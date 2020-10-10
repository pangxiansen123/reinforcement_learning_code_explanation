#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Nicky van Foreest(vanforeest@gmail.com)                        #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
# usable_ace手头是否有ace牌，且能叫为11点而不爆牌
# 手头牌面值和（12-21）0-11不需考虑，因为无论抽到什么牌怎么都不可能爆牌，故一定是hit
# 庄家的明牌。(1,…10)（J Q K都是10）
# 所以共有2*10*10 200个state，故policy表和value表就是2*10*10的
#######################################################################
# 21点游戏
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# actions: hit or stand
ACTION_HIT = 0 # 就是拿牌
ACTION_STAND = 1  #  "strike" in the book 停牌不拿
ACTIONS = [ACTION_HIT, ACTION_STAND]



# policy for player
POLICY_PLAYER = np.zeros(22, dtype=np.int)
# 只在自己牌面和为 20 or 21 时stand，其余一律hit，不考虑其他因素
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT
POLICY_PLAYER[20] = ACTION_STAND
POLICY_PLAYER[21] = ACTION_STAND


###########这俩policy是为off-policy算法准备
# 其中一个就是在12-20的时候拿牌
# 另一个方案就是随机的决定到底是拿牌还是停牌
# function form of target policy of player
def target_policy_player(usable_ace_player, player_sum, dealer_card):
    return POLICY_PLAYER[player_sum]
# function form of behavior policy of player
def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
    # 实验次数为1，赢得概率是0.5，下面就是一半一半的决定到底是拿牌还是停牌
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STAND
    return ACTION_HIT

# policy for dealer
# 这个就是书上说的固定策略，点数等于或者超过17的时候就停牌
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STAND

# get a new card
# 拿牌的时候随机得到一个牌，只能是1-10
def get_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# get the value of a card (11 for ace).
# 这里返回的都是11
def card_value(card_id):
    return 11 if card_id == 1 else card_id

# play a game 这个函数就是玩游戏，然后
# 返回初始化的状态，就是相对于玩家来说的，就是我知道自己手中的牌和庄家亮的一张牌，还有ace是否是用来作为11
# 然后收益，1 0 -1
# 最后是玩家的策略，就是每执行一个动作，那么这个动作下的状态
# @policy_player: specify policy for player
# @initial_state: [whether player has a usable Ace, sum of player's cards, one card of dealer]
# @initial_action: the initial action
# policy_player即玩家策略，由于庄家策略是固定的POLICY_DEALER，
# 所以可操作的策略为玩家策略。
# 这里可以选用固定的目标策略POLICY_PLAYER，
# 也可以自定义行动策略b来保证对状态空间的探索。
def play(policy_player, initial_state=None, initial_action=None):
    # player status

    # sum of player
    # 你手中的点数总和
    player_sum = 0

    # trajectory of player
    player_trajectory = []

    # whether player uses Ace as 11
    usable_ace_player = False

    # dealer status
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False
    # 这个就是书上的规定：
    #       玩家和庄家各两张牌，然后如果不是天和，那么玩家就可以一直要牌，这里假设玩家会要到12左右
    #       所以就会出现下面的while判断。然而庄家就只能要两张，因为是发完两张之后就是玩家进行要牌了
    if initial_state is None:
        # generate a random initial state
        # 这个while就是随机产生一个状态，或者下面的else，自己给定一个状态
        while player_sum < 12:
            # if sum of player is less than 12, always hit
            # 这里得到的牌数中只要是ace就是11
            card = get_card()
            player_sum += card_value(card)

            # If the player's sum is larger than 21, he may hold one or two aces.
            # 如果总和比21大，那么这个可能就是有一个或者两个ace
            if player_sum > 21:
                # 断言语句 如果player_sum等于22，就执行下面的语句，减去10，如果比22还大，那么就是错误
                assert player_sum == 22
                # last card must be ace
                player_sum -= 10
            else:
                #就是把ace使用为11的时候的标志位设置为True
                usable_ace_player |= (1 == card)

        # initialize cards of dealer, suppose dealer will show the first card he gets
        dealer_card1 = get_card()
        dealer_card2 = get_card()

    else:
        # use specified initial state
        # 就是使用了自己给定的初始状态，dealer_card1就是庄家亮的一个牌
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()

    # initial state of the game
    # 把初始化的状态放进到state中，就是相对于玩家来说的，就是我知道自己手中的牌和庄家亮的一张牌，还有ace是否是用来作为11
    state = [usable_ace_player, player_sum, dealer_card1]

    # initialize dealer's sum
    # 初始化的时候，里面的ace就是11，所以下面in就是看看庄家是否抽到了ace，抽到的话就把usable_ace_dealer为真
    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2)
    # if the dealer's sum is larger than 21, he must hold two aces.
    if dealer_sum > 21:
        assert dealer_sum == 22
        # use one Ace as 1 rather than 11
        dealer_sum -= 10
    assert dealer_sum <= 21 # 判断庄家时候爆牌
    assert player_sum <= 21 # 判断玩家时候爆牌

    # 游戏开始，就是前面的一堆程序，就是为了得到玩家手中的牌，以及庄家手中的牌
    # game starts!

    # player's turn 这个就是玩家回合
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            # get action based on current sum
            # policy_player就是输入参数里面的值
            action = policy_player(usable_ace_player, player_sum, dealer_card1)

        # track player's trajectory for importance sampling
        # (usable_ace_player, player_sum, dealer_card1)这个就构成了那200个状态
        player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])
        # 停止拿牌，然后退出while
        if action == ACTION_STAND:
            break
        # if hit, get new card
        card = get_card()
        # Keep track of the ace count. the usable_ace_player flag is insufficient alone as it cannot
        # distinguish between having one ace or two.
        # 记录自己手里面有几张被用的ace（就是牌是11）
        ace_count = int(usable_ace_player)
        if card == 1:
            ace_count += 1
        player_sum += card_value(card)
        # If the player has a usable ace, use it as 1 to avoid busting and continue.
        # 如果点数超过21，切有备用的ace，那么就将点数减去10，得到再判断，如此循环
        while player_sum > 21 and ace_count:
            player_sum -= 10
            ace_count -= 1
        # player busts
        # 如果里面的ace都成了1，但是点数还是超过了21，那么就直接爆牌，回报为-1
        if player_sum > 21:
            return state, -1, player_trajectory
        assert player_sum <= 21
        # 这个没有问题，都是一张牌一张牌进行判断的
        usable_ace_player = (ace_count == 1)

    # dealer's turn 庄家回合
    while True:
        # get action based on current sum
        action = POLICY_DEALER[dealer_sum]
        # 停止拿牌，然后退出while
        if action == ACTION_STAND:
            break
        # if hit, get a new card
        new_card = get_card()
        ace_count = int(usable_ace_dealer)
        if new_card == 1:
            ace_count += 1
        dealer_sum += card_value(new_card)
        # If the dealer has a usable ace, use it as 1 to avoid busting and continue.
        while dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1
        # dealer busts
        if dealer_sum > 21:
            return state, 1, player_trajectory
        usable_ace_dealer = (ace_count == 1)

    # compare the sum between player and dealer
    assert player_sum <= 21 and dealer_sum <= 21
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory

###------ 这个就是同策略每次访问型MC预测算法-------#############
# Monte Carlo Sample with On-Policy
# 同策略的蒙特卡洛采样 采样和训练使用的策略一样
# 该方法是对无完整过程模型p（状态转移矩阵），无法使用DP，而仅从交互序列中进行值函数v(s)估计的方法。
# 可分为first-visit 和 every-visit两种，
# 其区别在于first-visit仅处理每一个交互序列中某state的第一次出现，
# 而every-visit对每一个交互序列中某state的每次出现一视同仁。
# 返回每个状态下使用ace的收益和不适用ace的收益

# 该函数对前面定义的简单policy（就是一幕数据）进行v(s)评估。
# 将200个state按有无ace分两类分别返回（100+100）。
# 此函数使用every-visit，实践中绝大部分都是every-visit，因为实现更方便，不用验证是不是first-visit。
# 拿到一个交互序列后，遍历其中每一step，将交互序列的reward对应加到各个state上。
# 整个过程重复episodes次，最后对value表用state count求平均。
# 按此方法即可在未知过程模型的情况下，仅用policy 与 环境的交互结果对policy对应的v(s)进行估计。
def monte_carlo_on_policy(episodes):
    # 庄家十个状态，玩家十个状态
    states_usable_ace = np.zeros((10, 10)) # 在这个状态使用ace时候的收益
    # initialze counts to 1 to avoid 0 being divided
    states_usable_ace_count = np.ones((10, 10)) # 就是这个状态用ace的次数
    states_no_usable_ace = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    states_no_usable_ace_count = np.ones((10, 10))
    for i in tqdm(range(0, episodes)):
        # 就是进行一幕实验
        # 按照习惯，_表示某个变量是临时的或无关紧要的。
        _, reward, player_trajectory = play(target_policy_player)
        # 返回初始化的状态，就是相对于玩家来说的，就是我知道自己手中的牌和庄家亮的一张牌，还有ace是否是用来作为11
        # 然后收益，1 0 -1
        # 最后player_trajectory是玩家的策略，就是在这个状态的时候玩家拿牌的策略
        # player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])
        # 这个就不是首次遍历，而是全部遍历相加，这个只是一幕的数据
        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            # 因为玩家手中的牌是12-21，所以这里减去12，就是从0开始，9结束，代表着十个状态，方便计算
            player_sum -= 12
            # 同样的，庄家手中的牌是从1-10的，这里减去1，就是从0开始，9结束，代表着十个状态，方便计算
            dealer_card -= 1
            if usable_ace:
                states_usable_ace_count[player_sum, dealer_card] += 1
                states_usable_ace[player_sum, dealer_card] += reward
            else:
                states_no_usable_ace_count[player_sum, dealer_card] += 1
                states_no_usable_ace[player_sum, dealer_card] += reward
    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count


###------ 试探性策略评估，这里返回的是状态价值函数-------#############
# 我们要想将Monte Carlo应用到DP中的 policy iteration算法上求optimal policy，需
# 要估计state_action_values 即q(a,s)而非v(s)，
# 这是因为没有过程模型p时，仅有v(s)是无法求optimal action的（DP中可以）。
# 下面的算法采用greedy policy（贪婪算法），并用Exploring Starts弥补探索的缺失。
# 所谓Exploring Starts就是随机选取交互的init，
# 这样在当进行的episodes足够多的时候，就可以保证每个state都被探索到了。
# 显然Exploring Starts在很多实际问题中并不现实，因为init态很多时候是定死的，
# 导致Exploring Starts无法进行，之后会讨论其他保证探索的方法。
# Monte Carlo with Exploring Starts
def monte_carlo_es(episodes):
    # (playerSum, dealerCard, usableAce, action)
    # 四维空间
    state_action_values = np.zeros((10, 10, 2, 2))
    # initialze counts to 1 to avoid division by 0
    state_action_pair_count = np.ones((10, 10, 2, 2))

    # behavior policy is greedy
    # 这个函数的定义就是，根据usable_ace, player_sum, dealer_card所对应的状态，
    # 来计算该state下面各个动作所对应的value然后计算最大值，并且返回最大值所对应的action
    def behavior_policy(usable_ace, player_sum, dealer_card):
        usable_ace = int(usable_ace)
        player_sum -= 12
        dealer_card -= 1
        # get argmax of the average returns(s, a)
        values_ = state_action_values[player_sum, dealer_card, usable_ace, :] / \
                  state_action_pair_count[player_sum, dealer_card, usable_ace, :]
        # 这个是列表推导 就是得出value_最大值所对应的action_
        # np.random.choice这个意思就是可能最有动作有多个，随机选择一个就可以了
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # play for several episodes
    # tqdm就是进度条，就是进行多少幕实验
    for episode in tqdm(range(episodes)):
        # for each episode, use a randomly initialized state and action
        # state = [usable_ace_player, player_sum, dealer_card1]，就是随机选择，尽力使得所有的状态可以被遍历
        initial_state = [bool(np.random.choice([0, 1])),
                       np.random.choice(range(12, 22)),
                       np.random.choice(range(1, 11))]
        # 动作可以被遍历
        initial_action = np.random.choice(ACTIONS)
        # 如果是行为策略，那么就是最优策略,目标策略就是普通的策略
        current_policy = behavior_policy if episode else target_policy_player
        # 就是玩了一把游戏，返回汇报和策略π
        _, reward, trajectory = play(current_policy, initial_state, initial_action)
        # 创建集合
        first_visit_check = set()
        for (usable_ace, player_sum, dealer_card), action in trajectory:
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            state_action = (usable_ace, player_sum, dealer_card, action)
            if state_action in first_visit_check:
                continue
            first_visit_check.add(state_action)
            # update values of state-action pairs，这个是首次访问型，计算q(s,a),然后吧很多幕数据中出现这个状态进行相加
            state_action_values[player_sum, dealer_card, usable_ace, action] += reward
            state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1

    return state_action_values / state_action_pair_count

# Monte Carlo Sample with Off-Policy 这个是异策略
# 返回的是价值评估函数
def monte_carlo_off_policy(episodes):
    # state = [usable_ace_player, player_sum, dealer_card1]
    # 就是题目的例题5.4，玩家的和为13，庄家露牌2
    initial_state = [True, 13, 2]
    # 这个就是重要度采样比
    rhos = []
    # 回报
    returns = []

    for i in range(0, episodes):
        # behavior_policy_player一直不懂这个到底是啥意思
        _, reward, player_trajectory = play(behavior_policy_player, initial_state=initial_state)

        # get the importance ratio
        numerator = 1.0 # 分子
        denominator = 1.0 # 分母
        for (usable_ace, player_sum, dealer_card), action in player_trajectory:
            if action == target_policy_player(usable_ace, player_sum, dealer_card):
                denominator *= 0.5
            else:
                numerator = 0.0
                break
        # 就是求重要度采样比
        rho = numerator / denominator
        # 加入到列表
        rhos.append(rho)
        returns.append(reward)
    # 将列表转换为数组
    rhos = np.asarray(rhos)
    returns = np.asarray(returns)
    weighted_returns = rhos * returns
    # 回报权重的累加
    weighted_returns = np.add.accumulate(weighted_returns)
    # 权重的累加
    rhos = np.add.accumulate(rhos)
    # 普通重要度采样
    ordinary_sampling = weighted_returns / np.arange(1, episodes + 1)

    with np.errstate(divide='ignore',invalid='ignore'):
        # 满足条件(rhos != 0)，输出weighted_returns / rhos，不满足输出0。
        weighted_sampling = np.where(rhos != 0, weighted_returns / rhos, 0)

    return ordinary_sampling, weighted_sampling

def figure_5_1():
    states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(500000)

    states = [states_usable_ace_1,
              states_usable_ace_2,
              states_no_usable_ace_1,
              states_no_usable_ace_2]

    titles = ['Usable Ace, 10000 Episodes',
              'Usable Ace, 500000 Episodes',
              'No Usable Ace, 10000 Episodes',
              'No Usable Ace, 500000 Episodes']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    for state, title, axis in zip(states, titles, axes):
        # 热力图 因为xy是反的，左右这里应该进行转换
        fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('../images/figure_5_1.png')
    plt.close()

def figure_5_2():
    state_action_values = monte_carlo_es(500000)

    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)

    # get the optimal policy state_action_values这个是四维的，，axis=-1这个意思就是按照最大值所对应的action
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)

    images = [action_usable_ace,
              state_value_usable_ace,
              action_no_usable_ace,
              state_value_no_usable_ace]

    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('../images/figure_5_2.png')
    plt.close()

def figure_5_3():
    true_value = -0.27726
    episodes = 10000
    runs = 100
    error_ordinary = np.zeros(episodes)
    error_weighted = np.zeros(episodes)
    for i in tqdm(range(0, runs)):
        ordinary_sampling_, weighted_sampling_ = monte_carlo_off_policy(episodes)
        # get the squared error
        error_ordinary += np.power(ordinary_sampling_ - true_value, 2)
        error_weighted += np.power(weighted_sampling_ - true_value, 2)
    error_ordinary /= runs
    error_weighted /= runs

    plt.plot(error_ordinary, label='Ordinary Importance Sampling')
    plt.plot(error_weighted, label='Weighted Importance Sampling')
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Mean square error')
    plt.xscale('log')
    plt.legend()

    plt.savefig('../images/figure_5_3.png')
    plt.close()


if __name__ == '__main__':
    # # 给个图就是各个状态所对应的回报
    # figure_5_1()
    # 每个状态所对应的价值和动作
    # figure_5_2()
    figure_5_3()

