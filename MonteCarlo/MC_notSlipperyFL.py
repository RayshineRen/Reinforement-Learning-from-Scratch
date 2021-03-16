import gym
from gym.envs.registration import register
import random
import numpy as np
from tqdm import *

register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery': False},
        max_episode_steps=1000,
        reward_threshold=0.78, # optimum = .8196
    )

env = gym.make('FrozenLakeNotSlippery-v0')
env.reset()
env.render()

def run_episode(env, policy, gamma=1.0, render=False):
    states = []
    actions = []
    rewards = []
    obs = env.reset()
    while True:
        states.append(obs)
        if render:
            env.render()
        action = np.random.choice(np.array([i for i in range(env.env.nA)]), p=policy[obs])
        obs, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        if done:
            break
    return states, actions, rewards

def evaluate_policy(env, policy, gamma=1.0, n=100):
    scores = [
        np.mean(run_episode(env, policy, gamma, False)[2]) for _ in range(n)
    ]
    return np.mean(scores)

def eps_greedy(env, eps, Q_table):
    policy = {}
    for state in range(env.env.nS):
        Q_sa = Q_table[state]
        prob = np.zeros(env.action_space.n)
        # a_star = np.argmax(Q_table[state])
        a_star_value = np.max(Q_table[state])
        for action in range(env.env.nA):
            if Q_sa[action] == a_star_value:
                prob[action] = eps / env.action_space.n + 1 - eps
            else:
                prob[action] = eps / env.action_space.n
        if np.sum(prob) != 1:
            prob /= np.sum(prob)
        policy[state] = prob
    return policy

def monte_calro_qFunc(env, gamma=1.0, epslion=0.15, episodes=50000):
    Q_table = np.zeros((env.env.nS, env.env.nA))
    N_table = np.zeros((env.env.nS, env.env.nA))
    policy = eps_greedy(env, 1, Q_table)
    eps = 0.15
    for k in tqdm(range(2, episodes+2)):
        states, actions, rewards = run_episode(env, policy, gamma, False)
        G = 0.0
        N = len(states)
        for step in range(N-1, -1, -1):
            G *= gamma
            G += rewards[step]
        for step in range(N):
            N_table[states[step]][actions[step]] += 1
            # if states[step] not in states[:step]:
            # 在第一个状态路过两次。e.g. [0, 0, 4, 8, 9, 13, 14] 只会记录第一次动作的价值，而且是错误的！
            Q_table[states[step]][actions[step]] += 1/N_table[states[step]][actions[step]] * (G - Q_table[states[step]][actions[step]])
            G -= rewards[step]
            G /= gamma
        # eps = 1 / k
        policy = eps_greedy(env, eps, Q_table)
        # print(policy)
        # print(Q_table)
    return Q_table, policy

def show_policy(policy):
    OutPolicy = []
    for state in policy.keys():
        OutPolicy.append(np.argmax(policy[state]))
    return OutPolicy

# 不使用epslion-greedy算法，计算更快
Qtable, policy = monte_calro_qFunc(env, gamma=0.85)
policy = show_policy(policy)
print(Qtable)
print(policy)
obs = env.reset()
while True:
    env.render()
    action = policy[obs]
    obs, reward, done, _ = env.step(action)
    if done:
        env.render()
        break














