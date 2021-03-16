import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np

register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery': False},
        max_episode_steps=1000,
        reward_threshold=0.78, # optimum = .8196
    )
env = gym.make('FrozenLakeNotSlippery-v0')

def run_episode(env, policy, gamma=1.0, render=False):
    obs = env.reset()
    total_reward = 0.0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += gamma**step_idx*reward
        step_idx += 1
        if done:
             break
    return total_reward

def evaluate_policy(env, policy, gamma=1.0, n=100):
    scores = [
        run_episode(env, policy, gamma, False) for _ in range(n)
    ]
    return np.mean(scores)

def extract_policy(env, v, gamma=1.0):
    policy = np.zeros(env.env.nS)
    for state in range(env.env.nS):
        q_sa = [sum([p*(r+gamma*v[s_]) for p, s_, r, _ in env.env.P[state][action]]) for action in range(env.env.nA)]
        policy[state] = np.argmax(q_sa)
    return policy

def value_iteration(env, gamma=1.0, n=5000):
    error = []  # 价值函数的差
    index = []  # 价值函数的第一个元素
    eps = 1e-20
    v = np.zeros(env.env.nS)
    for i in range(n):
        pre_v = np.copy(v)
        for state in range(env.env.nS):
            q_sa = [sum([p*(r+gamma*pre_v[s_]) for p, s_, r, _ in env.env.P[state][action]]) for action in range(env.env.nA)]
            v[state] = max(q_sa)
        index.append(v[0])
        error.append(np.sum(np.fabs(pre_v - v)))
        if(np.sum(np.fabs(pre_v - v)) <= eps):
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    # 画出价值函数的收敛曲线
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].plot(range(1, len(error) + 1), error)
    # 画出价值函数第一个元素的值的收敛情况
    ax[1].plot(range(1, len(index) + 1), index)
    plt.show()
    return v

if __name__ == '__main__':
    gamma = 0.5 # gamma=1.0时，状态0的价值与周围一致，最佳策略可能是留在原地
    optimal_v = value_iteration(env, gamma)
    policy = extract_policy(env, optimal_v, gamma)
    print('the best policy:\n', policy)
    run_episode(env, policy, gamma, True)
    policy_score = evaluate_policy(env, policy, gamma, 100)
    print('Policy average score = ', policy_score)












