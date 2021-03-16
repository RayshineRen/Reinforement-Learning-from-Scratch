import gym
import numpy as np

env = gym.make('FrozenLake-v0')

def compute_value_function(env, policy, gamma = 1.0, threshold=1e-20):
    value_table = np.zeros(env.env.nS)
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.env.nS):
            action = policy[state]
            value_table[state] = sum([p*(r + gamma*updated_value_table[s_]) for p, s_, r, _ in env.env.P[state][action]])
        if(np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break
    return value_table

def extract_policy(value_table,gamma=1.0):
    policy = np.zeros(env.env.nS)
    for state in range(env.observation_space.n):
        Q_table = [sum([p*(r+gamma*value_table[s_]) for p, s_, r, _ in env.env.P[state][action]]) for action in range(env.action_space.n)]
        policy[state] = np.argmax(Q_table)
    return policy

def policy_iteration(env,gamma = 1.0, no_of_iterations = 200000):
    '''
    状态值估计和策略函数的优化是交替进行的，从随机策略出发，估计状态价值
    再从收敛的状态值函数出发，优化之前的随机策略。由此往复，直至收敛
    '''
    gamma = 1.0

    random_policy = np.random.choice(env.env.nA, size=(env.env.nS))
    for i in range(no_of_iterations):
        new_value_func = compute_value_function(env, random_policy, gamma)
        new_policy = extract_policy(new_value_func)
        if(np.all(random_policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        random_policy = new_policy
    return new_policy

env.reset()
print(policy_iteration(env))











