import gym
import numpy as np
import matplotlib.pyplot as plt

#       3
#       ^
#       |
# 0< —— o —— > 2
#       |
#       v
#       1
# 动作空间=4，分别是上下左右移动，同时湖面上可能刮风，使agent随机移动

# S Starting point,safe
# F Frozen surface,safe
# H Hole,end of game,bad
# G Goal,end of game,good

# 状态空间=16,4x4的矩阵
# SFFF
# FHFH
# FFFH
# HFFG

# 在左侧边缘，无法向左移动，要么原地不动，要么向下移动
# 在上侧边缘，无法向上移动，要么原地不动，要么向右移动

env = gym.make('FrozenLake-v0')
env.reset()

def value_iteration(env, gamma = 1.0, no_of_iterations = 2000):
    '''
    值迭代函数，目的是准确评估每一个状态的好坏
    '''
    # 状态价值函数，向量维度等于游戏状态的个数
    value_table = np.zeros(env.observation_space.n)
    # 随着迭代，记录算法的收敛性
    error = []  # 价值函数的差
    index = []  # 价值函数的第一个元素
    threshold = 1e-20
    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)
        for state in range(env.env.nS):
            Q_value = [sum(p*(r + gamma*updated_value_table[s_]) for p, s_, r, _ in env.env.P[state][action]) for action in range(env.env.nA)]
            value_table[state] = max(Q_value)
        index.append(value_table[0])
        error.append(np.sum(np.fabs(updated_value_table - value_table)))
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    # 画出价值函数的收敛曲线
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].plot(range(1, len(error)+1),error)
    # 画出价值函数第一个元素的值的收敛情况
    ax[1].plot(range(1, len(index)+1),index)
    plt.show()
    return updated_value_table

def extract_policy(value_table, gamma=1.0):
    '''
    在一个收敛的、能够对状态进行准确评估的状态值函数的基础上，推导出策略函数，即在每一个状态下应该采取什么动作最优的
    '''

    # policy代表处于状态t时应该采取的最佳动作是上/下/左/右,policy长度16
    policy = np.zeros(env.observation_space.n)

    for state in range(env.env.nS):
        Q_table = [sum([p*(r + gamma*value_table[s_]) for p, s_, r, _ in env.env.P[state][action]]) for action in range(env.env.nA)]
        policy[state] = np.argmax(Q_table)
    return policy

optimal_value_function = value_iteration(env=env)
print('\nthe best value function:\n',optimal_value_function,'\n')

optimal_policy = extract_policy(optimal_value_function, gamma=1.0)
print('the best policy:\n',optimal_policy)



