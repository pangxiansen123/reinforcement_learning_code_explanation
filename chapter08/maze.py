#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import turtle
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import deepcopy


class PriorityQueue:
    def __init__(self):
        # 定义一个优先队列的列表
        self.pq = []
        # 是一个字典
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0
    # item是不可改变的，所以这里是用元组
    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        # item就是键，entry就是值
        self.entry_finder[item] = entry
        # 这里是使用堆，数据结构堆（heap）是一种优先队列。
        # 将entry列表堆入到堆中，并且按照priority进行优先级排列
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED
    # 这个就是从优先队列中弹出一个元素
    def pop_item(self):
        while self.pq:
            # 堆弹出，并且删除这个元素
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        # 空就返回真True
        return not self.entry_finder

# A wrapper class for a maze, containing all the information about the maze.
# 一个类的封装，包含全部关于迷宫的信息
# Basically it's initialized to DynaMaze by default, however it can be easily adapted
# to other maze
# 基本上它被初始化为DynaMaze默认，但是它可以很容易地适应其他迷宫
class Maze:
    def __init__(self):
        # maze width
        self.WORLD_WIDTH = 9

        # maze height
        self.WORLD_HEIGHT = 6

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state
        self.START_STATE = [0, 0]

        # goal state
        self.GOAL_STATES = [[0, 8]]

        # all obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.old_obstacles = None
        self.new_obstacles = None

        # time to change obstacles
        # 就是改变障碍物的时间
        self.obstacle_switch_time = None

        # initial state action pair values
        # self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # the size of q value
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))

        # max steps
        self.max_steps = float('inf')

        # track the resolution for this maze
        self.resolution = 1

    # extend a state to a higher resolution maze
    # @state: state in lower resoultion maze
    # @factor: extension factor, one state will become factor^2 states after extension
    # 这个函数就是将矩阵中的一个元素进行扩充，之前是2*2，假设factor为2，扩充到4*4
    def extend_state(self, state, factor):
        new_state = [state[0] * factor, state[1] * factor]
        new_states = []
        for i in range(0, factor):
            for j in range(0, factor):
                new_states.append([new_state[0] + i, new_state[1] + j])
        return new_states

    # extend a state into higher resolution
    # one state in original maze will become @factor^2 states in @return new maze
    def extend_maze(self, factor):
        new_maze = Maze()
        new_maze.WORLD_WIDTH = self.WORLD_WIDTH * factor
        new_maze.WORLD_HEIGHT = self.WORLD_HEIGHT * factor
        # 这个表明开始点还是开始点，没有扩充成矩阵
        new_maze.START_STATE = [self.START_STATE[0] * factor, self.START_STATE[1] * factor]
        # 但是目的地却扩充成矩阵了
        new_maze.GOAL_STATES = self.extend_state(self.GOAL_STATES[0], factor)
        new_maze.obstacles = []
        for state in self.obstacles:
            # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            new_maze.obstacles.extend(self.extend_state(state, factor))
        new_maze.q_size = (new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions))
        # new_maze.stateActionValues = np.zeros((new_maze.WORLD_HEIGHT, new_maze.WORLD_WIDTH, len(new_maze.actions)))
        # 这个分辨率有问题，应该是factor的倒数
        new_maze.resolution = factor
        return new_maze

    # take @action in @state
    # @return: [new state, reward]
    # 根据动作和状态，得到下一个动作，和即时回报
    def step(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        # 如果动作碰到了障碍物，那么就回到原处
        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
        else:
            reward = 0.0
        return [x, y], reward

# a wrapper class for parameters of dyna algorithms
# 就是一堆参数
class DynaParams:
    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.1

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.time_weight = 0

        # n-step planning
        self.planning_steps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']

        # threshold for priority queue
        # 优先级队列阈值
        self.theta = 0


# choose an action based on epsilon-greedy algorithm
# 就是epsilon-greedy贪心算法
def choose_action(state, q_value, maze, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        # 最大值所对应的下标就是动作
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

# Trivial model for planning in Dyna-Q
# 训练模型model
class TrivialModel:
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict() # dictionary 字典
        self.rand = rand

    # feed the model with previous experience
    # 根据之前的数据对模型进行训练，就是对字典进行更新
    def feed(self, state, action, next_state, reward):
        # 如果要复制的列表中包含了列表，那就使用copy.deepcopy()函数来代替。deepcopy()函数将同时复制它们内部的列表。
        # 就是深层次的复制，不仅仅是复制引用
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        # tuple(state)就是将列表转化成元组
        # 值可以取任何数据类型，但键必须是不可变的，如字符串，数字或元组。所以这里面将键设置为元组。
        if tuple(state) not in self.model.keys():
            # 所以tuple(state)就是key（键）
            # 然后这个键所对应的值是一个字典
            # 就是字典里面还有一个字典
            self.model[tuple(state)] = dict()
        # 更新这个状态下的的回报及下一个状态(下一个状态是列表)
        # 字典中的值也是一个字典，字典的键就是动作，值就是一个列表
        self.model[tuple(state)][action] = [list(next_state), reward]

    # randomly sample from previous experience
    # 从之前的状态进行采样
    # 就是随机采样，得到现在的状态和动作，得到下一个状态和即时回报
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        # 将字典的key转换成列表
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        # 因为值也是一个字典，所以值字典的key也成为一个字典
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward

# Time-based model for planning in Dyna-Q+
# 基于时间的模型，就是将时间考虑进去，很久没有访问的动作，会有更加大的额外概率进行访问（只是说的额外的，本身的回报可能比较小）
class TimeModel:
    # @maze: the maze instance. Indeed it's not very reasonable to give access to maze to the model.
    # @timeWeight: also called kappa, the weight for elapsed time in sampling reward, it need to be small
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, maze, time_weight=1e-4, rand=np.random):
        self.rand = rand
        self.model = dict()

        # track the total time
        self.time = 0

        self.time_weight = time_weight
        self.maze = maze

    # feed the model with previous experience
    # 根据之前的实验经验进行模型的学习训练
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        self.time += 1
        # 字典的键是不可变化的，所以这里将状态作为元组，作为键
        if tuple(state) not in self.model.keys():
            # 字典的值是字典
            self.model[tuple(state)] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            # 这个动作之前是从未被遍历过的，被考虑到计划步骤中
            # 允许在规划步骤中考虑以前从未尝试过的行动，那么他的回报就是0，并且时间设置成1
            for action_ in self.maze.actions:
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    # 没有访问过得的动作的时间全部设置为1，为了计算额外的回报方便
                    self.model[tuple(state)][action_] = [list(state), 0, 1]

        self.model[tuple(state)][action] = [list(next_state), reward, self.time]

    # randomly sample from previous experience
    # 从之前的经验里面随机产生样本
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward, time = self.model[state][action]

        # adjust reward with elapsed time since last vist
        # 自从上一次拜访所过去的时间作为调节回报，就是对回报进行额外的试探收益
        # 这个就是Dyna-Q+的优点，就是文中说的，在Dyna-Q的基础上增加额外的试探收益来鼓励试探性动作
        # 就是很久没遍历的动作，增加的额外回报会更大
        reward += self.time_weight * np.sqrt(self.time - time)

        state = deepcopy(state)
        next_state = deepcopy(next_state)

        return list(state), action, list(next_state), reward

# Model containing a priority queue for Prioritized Sweeping
# 这个模型中包含一个优先队列
class PriorityModel(TrivialModel):
    def __init__(self, rand=np.random):
        TrivialModel.__init__(self, rand)
        # maintain a priority queue
        self.priority_queue = PriorityQueue()
        # track predecessors for every state
        # 这个就是追踪每一个状态的前向状态，就是什么状态和动作可以到达这个状态
        self.predecessors = dict()

    # add a @state-@action pair into the priority queue with priority @priority
    def insert(self, priority, state, action):
        # note the priority queue is a minimum heap, so we use -priority
        # 返回的就是实际中最大的元素
        # (tuple(state), action)就是一个item
        self.priority_queue.add_item((tuple(state), action), -priority)

    # @return: whether the priority queue is empty
    def empty(self):
        return self.priority_queue.empty()

    # get the first item in the priority queue
    # 得到优先队列的第一个元素，这个也不叫采样了，这个就是根据优先队列，得到一个item
    def sample(self):
        (state, action), priority = self.priority_queue.pop_item()
        # 模型是一个字典，字典里面还是一个字典，所以这里用两个中括号
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return -priority, list(state), action, list(next_state), reward

    # feed the model with previous experience
    # 就是进行模型的学习
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        TrivialModel.feed(self, state, action, next_state, reward)
        if tuple(next_state) not in self.predecessors.keys():
            # set()是创建一个空集合，元素之间无序，每个元素唯一，不存在相同元素
            self.predecessors[tuple(next_state)] = set()
        self.predecessors[tuple(next_state)].add((tuple(state), action))

    # get all seen predecessors of a state @state
    def predecessor(self, state):
        if tuple(state) not in self.predecessors.keys():
            return []
        predecessors = []
        for state_pre, action_pre in list(self.predecessors[tuple(state)]):
            # 将前一个状态，动作，和达到本次状态的回报加入到predecessors中
            predecessors.append([list(state_pre), action_pre, self.model[state_pre][action_pre][1]])
        return predecessors

#####--------这个model相当于父类，可以调用子类中的函数
# play for an episode for Dyna-Q algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
# 返回步数，就是运行一幕结束后的步数
def dyna_q(q_value, model, maze, dyna_params):
    # START_STATE = [2, 0]
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        # track the steps
        steps += 1

        # get action
        # 就是epsilon-greedy贪心算法
        # dyna_params这里面存储的就是各种所需的参数
        # turtle.penup()
        # turtle.goto((state[0])*20,(state[1])*20)
        # turtle.dot()
        action = choose_action(state, q_value, maze, dyna_params)

        # take action
        # 根据动作和状态，得到下一个动作，和即时回报
        # 这个应该就是真实经验得到的数据
        next_state, reward = maze.step(state, action)

        # Q-Learning update
        # 就是Q学习，Q(S,A)=Q(S,A)+alpha*(R+gamma*(max(Q(S',A))-Q(S,A)))
        q_value[state[0], state[1], action] += \
            dyna_params.alpha * (reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                                 q_value[state[0], state[1], action])

        # feed the model with experience
        # 就是将得到的数据进行更新，更新到model里面
        # 实际经验的模型学习
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        # 从模型之中进行样本学习，这个就是规划，这个意思就是在真实经验得到后，会进行模型的更新，
        # 然后在更新后的矩阵中进行更新，就是随机采样，进行更新
        for t in range(0, dyna_params.planning_steps):
            # 就是随机采样，得到现在的状态和动作，得到下一个状态和即时回报
            state_, action_, next_state_, reward_ = model.sample()
            q_value[state_[0], state_[1], action_] += \
                dyna_params.alpha * (reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) -
                                     q_value[state_[0], state_[1], action_])

        state = next_state
        # plt.plot(state)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        #plt.legend()
        # check whether it has exceeded the step limit
        if steps > maze.max_steps:
            break
    return steps

# play for an episode for prioritized sweeping algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
# @return: # of backups during this episode
def prioritized_sweeping(q_value, model, maze, dyna_params):
    state = maze.START_STATE

    # track the steps in this episode
    steps = 0

    # track the backups in planning phase
    backups = 0

    while state not in maze.GOAL_STATES:
        steps += 1

        # get action
        # 这里的策略都是epsilon-greedy贪心算法
        action = choose_action(state, q_value, maze, dyna_params)

        # take action
        next_state, reward = maze.step(state, action)

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # get the priority for current state action pair
        # 这个就是文中说的变化越大，优先级越高
        priority = np.abs(reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                          q_value[state[0], state[1], action])

        if priority > dyna_params.theta:
            # 就是把这个状态和动作放入到优先队列里面
            model.insert(priority, state, action)

        # start planning
        planning_step = 0

        # planning for several steps,
        # although keep planning until the priority queue becomes empty will converge much faster
        while planning_step < dyna_params.planning_steps and not model.empty():
            # get a sample with highest priority from the model
            priority, state_, action_, next_state_, reward_ = model.sample()

            # update the state action value for the sample
            # 根据书上说的，先更新Q(S,A)
            delta = reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) - \
                    q_value[state_[0], state_[1], action_]
            q_value[state_[0], state_[1], action_] += dyna_params.alpha * delta

            # deal with all the predecessors of the sample state
            for state_pre, action_pre, reward_pre in model.predecessor(state_):
                priority = np.abs(reward_pre + dyna_params.gamma * np.max(q_value[state_[0], state_[1], :]) -
                                  q_value[state_pre[0], state_pre[1], action_pre])
                if priority > dyna_params.theta:
                    model.insert(priority, state_pre, action_pre)
            planning_step += 1

        state = next_state

        # update the # of backups
        # planning_step + 1到这里，意思就是5
        # backups就是反向回溯的总步数
        backups += planning_step + 1

    return backups


def show_path(dyna_maze,q_value):
    mystate = dyna_maze.START_STATE
    x, y = mystate
    mysteps = 0
    optimal_policy_x = []
    optimal_policy_y = []
    optimal_policy_x.append(y)
    optimal_policy_y.append(-x)
    while [x, y] not in dyna_maze.GOAL_STATES:
        bestAction = np.argmax(q_value[x, y, :])
        mysteps += 1
        if bestAction == dyna_maze.ACTION_UP:
            x = max(x - 1, 0)
        elif bestAction == dyna_maze.ACTION_DOWN:
            x = min(x + 1, dyna_maze.WORLD_HEIGHT - 1)
        elif bestAction == dyna_maze.ACTION_LEFT:
            y = max(y - 1, 0)
        elif bestAction == dyna_maze.ACTION_RIGHT:
            y = min(y + 1, dyna_maze.WORLD_WIDTH - 1)
            # 如果动作碰到了障碍物，那么就回到原处
        if [x, y] in dyna_maze.obstacles:
            break
        optimal_policy_x.append(y)
        optimal_policy_y.append(-x)
        plt.plot(optimal_policy_x, optimal_policy_y)
        plt.draw()
        plt.pause(0.01)
        plt.clf()
    print("使用了{}步".format(mysteps))
# Figure 8.2, DynaMaze, use 10 runs instead of 30 runs
def figure_8_2():
    # set up an instance for DynaMaze
    # 生成一个矩阵
    plt.ion()  # 开启interactive mode 成功的关键函数
    dyna_maze = Maze()
    dyna_params = DynaParams()

    runs = 10
    episodes = 50
    # 这个就是几步规划的问题
    planning_steps = [0, 5, 50]
    steps = np.zeros((len(planning_steps), episodes))

    for run in tqdm(range(runs)):
        for i, planning_step in enumerate(planning_steps):
            dyna_params.planning_steps = planning_step
            q_value = np.zeros(dyna_maze.q_size)

            # generate an instance of Dyna-Q model
            # 就是创建一个模型
            model = TrivialModel()
            for ep in range(episodes):
                # print('run:', run, 'planning step:', planning_step, 'episode:', ep)
                # 运行十次，每次步数都会进行叠加
                steps[i, ep] += dyna_q(q_value, model, dyna_maze, dyna_params)
            show_path(dyna_maze, q_value)
    # averaging over runs
    steps /= runs

    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], label='%d planning steps' % (planning_steps[i]))
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()

    plt.savefig('../images/figure_8_2.png')
    plt.close()

# wrapper function for changing maze
# @maze: a maze instance
# @dynaParams: several parameters for dyna algorithms
# 就是返回累计回报值，就是成功了几次，最后的步数就是总的回报值
def changing_maze(maze, dyna_params):

    # set up max steps 3000
    max_steps = maze.max_steps

    # track the cumulative rewards 跟踪累积奖励
    # dyna_params.runs这里是20,2应该意思就是两个方法，max_steps是3000
    rewards = np.zeros((dyna_params.runs, 2, max_steps))

    for run in tqdm(range(dyna_params.runs)):
        # set up models 设置模型
        # TrivialModel()训练模型model
        # models里面有两个模型，一个是Dyna-Q的模型,一个是Dyna-Q+的模型
        models = [TrivialModel(), TimeModel(maze, time_weight=dyna_params.time_weight)]

        # initialize state action values
        # maze.q_size这个是三维向量，那么q_values就是四维向量
        q_values = [np.zeros(maze.q_size), np.zeros(maze.q_size)]

        # methods = ['Dyna-Q', 'Dyna-Q+']
        for i in range(len(dyna_params.methods)):
            # print('run:', run, dyna_params.methods[i])

            # set old obstacles for the maze
            maze.obstacles = maze.old_obstacles

            steps = 0
            last_steps = steps
            # 就是运行的最大步数
            while steps < max_steps:
                # play for an episode
                # 返回步数，就是运行一幕结束后的步数
                steps += dyna_q(q_values[i], models[i], maze, dyna_params)
                # update cumulative rewards
                # 进行一幕需要很多步，但是我们只处理最后一步，也就是最后一步加1，其他步保持不变
                rewards[run, i, last_steps: steps] = rewards[run, i, last_steps]
                rewards[run, i, min(steps, max_steps - 1)] = rewards[run, i, last_steps] + 1
                last_steps = steps

                if steps > maze.obstacle_switch_time:
                    # change the obstacles
                    maze.obstacles = maze.new_obstacles
            show_path(maze, q_values[i])

    # averaging over runs 压缩run，就是把run去掉
    rewards = rewards.mean(axis=0)

    return rewards

# Figure 8.4, BlockingMaze
# 壁障迷宫，就是障碍物发生改变时候，学习路线的改变
def figure_8_4():
    # set up a blocking maze instance
    blocking_maze = Maze()
    blocking_maze.START_STATE = [5, 3]
    blocking_maze.GOAL_STATES = [[0, 8]]
    # 旧的障碍物
    blocking_maze.old_obstacles = [[3, i] for i in range(0, 8)]

    # new obstalces will block the optimal path 阻塞
    # 新的障碍物
    blocking_maze.new_obstacles = [[3, i] for i in range(1, 9)]

    # step limit
    blocking_maze.max_steps = 3000

    # obstacles will change after 1000 steps
    # the exact step for changing will be different
    # However given that 1000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    # 在步数时间是1000的时候改变障碍物
    blocking_maze.obstacle_switch_time = 1000

    # set up parameters
    dyna_params = DynaParams()
    dyna_params.alpha = 1.0
    dyna_params.planning_steps = 10
    dyna_params.runs = 20

    # kappa must be small, as the reward for getting the goal is only 1
    dyna_params.time_weight = 1e-4

    # play
    # 就是返回累计回报值，就是成功了几次，最后的步数就是总的回报值的平均值
    rewards = changing_maze(blocking_maze, dyna_params)

    for i in range(len(dyna_params.methods)):
        plt.plot(rewards[i, :], label=dyna_params.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.savefig('../images/figure_8_4.png')
    plt.close()

# Figure 8.5, ShortcutMaze
def figure_8_5():
    # set up a shortcut maze instance
    shortcut_maze = Maze()
    shortcut_maze.START_STATE = [5, 3]
    shortcut_maze.GOAL_STATES = [[0, 8]]
    shortcut_maze.old_obstacles = [[3, i] for i in range(1, 9)]

    # new obstacles will have a shorter path
    shortcut_maze.new_obstacles = [[3, i] for i in range(1, 8)]

    # step limit
    shortcut_maze.max_steps = 600

    # obstacles will change after 3000 steps
    # the exact step for changing will be different
    # However given that 3000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    shortcut_maze.obstacle_switch_time = 300

    # set up parameters
    dyna_params = DynaParams()

    # 50-step planning
    dyna_params.planning_steps = 50
    dyna_params.runs = 5
    dyna_params.time_weight = 1e-3
    dyna_params.alpha = 1.0

    # 就是返回累计回报值，就是成功了几次，最后的步数就是总的回报值
    rewards = changing_maze(shortcut_maze, dyna_params)

    for i in range(len(dyna_params.methods)):
        plt.plot(rewards[i, :], label=dyna_params.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.savefig('../images/figure_8_5.png')
    plt.close()

# Check whether state-action values are already optimal
def check_path(q_values, maze):
    # get the length of optimal path
    # 14 is the length of optimal path of the original maze
    # 1.2 means it's a relaxed optifmal path
    max_steps = 14 * maze.resolution * 1.2
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        action = np.argmax(q_values[state[0], state[1], :])
        state, _ = maze.step(state, action)
        steps += 1
        if steps > max_steps:
            return False
    return True

# Example 8.4, mazes with different resolution
def example_8_4():
    # get the original 6 * 9 maze
    original_maze = Maze()

    # set up the parameters for each algorithm
    params_dyna = DynaParams()
    params_dyna.planning_steps = 5
    params_dyna.alpha = 0.5
    params_dyna.gamma = 0.95

    params_prioritized = DynaParams()
    params_prioritized.theta = 0.0001
    params_prioritized.planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params = [params_prioritized, params_dyna]

    # set up models for planning
    models = [PriorityModel, TrivialModel]
    method_names = ['Prioritized Sweeping', 'Dyna-Q']

    # due to limitation of my machine, I can only perform experiments for 5 mazes
    # assuming the 1st maze has w * h states, then k-th maze has w * h * k * k states
    num_of_mazes = 3

    # build all the mazes
    mazes = [original_maze.extend_maze(i) for i in range(1, num_of_mazes + 1)]
    methods = [prioritized_sweeping, dyna_q]

    # My machine cannot afford too many runs...
    runs = 5

    # track the # of backups
    # 运行的次数，两个方法，矩阵的维度
    backups = np.zeros((runs, 2, num_of_mazes))

    for run in range(0, runs):
        for i in range(0, len(method_names)):
            # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表，元素个数与最短的列表一致
            for mazeIndex, maze in zip(range(0, len(mazes)), mazes):
                print('run %d, %s, maze size %d' % (run, method_names[i], maze.WORLD_HEIGHT * maze.WORLD_WIDTH))

                # initialize the state action values
                q_value = np.zeros(maze.q_size)

                # track steps / backups for each episode
                steps = []

                # generate the model
                model = models[i]()

                # play for an episode
                # 这里面就是进行一幕一幕的真实实验的进行
                while True:
                    # 返回一幕实验反向回溯的步数
                    # 然后将每幕实验的回溯步数相加
                    steps.append(methods[i](q_value, model, maze, params[i]))

                    # print best actions w.r.t. current state-action values
                    # printActions(currentStateActionValues, maze)

                    # check whether the (relaxed) optimal path is found
                    if check_path(q_value, maze):
                        break
                show_path(maze, q_value)
                # update the total steps / backups for this maze
                backups[run, i, mazeIndex] = np.sum(steps)
    # 对run进行平均值的求解
    backups = backups.mean(axis=0)

    # Dyna-Q performs several backups per step
    # Dyna - Q每个步骤执行多个反向回溯
    # 这里有个疑问，为什么要加1呢
    backups[1, :] *= params_dyna.planning_steps + 1

    for i in range(0, len(method_names)):
        plt.plot(np.arange(1, num_of_mazes + 1), backups[i, :], label=method_names[i])
    plt.xlabel('maze resolution factor')
    plt.ylabel('backups until optimal solution')
    plt.yscale('log')
    plt.legend()

    plt.savefig('../images/example_8_4.png')
    plt.close()

if __name__ == '__main__':
    # figure_8_2()
    # figure_8_4()
    # figure_8_5()
    example_8_4()

