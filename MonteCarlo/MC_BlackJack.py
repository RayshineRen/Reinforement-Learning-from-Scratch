import numpy as np
import gym
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from functools import partial

env = gym.make('Blackjack-v0')

## 庄家Dealer Score只显示一张牌的值，实际它有两张牌,玩家停止叫牌后，庄家可以选择叫牌或者停止
def print_observation(observation, reward=0):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}, Reward:{}".format(score, usable_ace, dealer_score, reward))

def sample_policy(observation):
    '''
    如果当前牌面值>=20，就不再叫牌（Stand）
    如果当前牌面值<20,继续叫牌（Hit）
    '''
    # observation len=3,分别是闲家分数、庄家分数、是否将A当做11
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

def test_env():
    for i_episode in range(7):
        observation = env.reset()
        reward = 0
        for t in range(100):
            print_observation(observation, reward)
            action = sample_policy(observation) # 采取的措施
            print("Taking action: {}".format( ["Stick(不要)", "Hit(要)"][action]))
            observation, reward, done, _ = env.step(action) # 环境的反馈
            if done: # 判断是否结束
                print_observation(observation, reward)
                if reward == 1:
                    result = 'Win'
                elif reward == 0:
                    result = 'Draw'
                else:
                    result = 'Loss'
                print("Game end. Reward: {}, Result:{}\n".format(float(reward), result))
                break

def generate_episode(policy, env):
    '''
    玩一局游戏，收集状态信息、动作信息和reward
    '''
    states, actions, rewards = [], [], []
    observation = env.reset()
    while True:
        states.append(observation)
        # 根据策略函数，确定采取的动作
        action = policy(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            # states.append(observation)
            break
    return states, actions, rewards

# for _ in range(5):
#     print("Game%i"%(_))
#     states, actions, rewards = generate_episode(sample_policy, env)
#     print(states)
#     print(actions)
#     print(rewards)
#     print()

def testDefaultdict():
    s = 'mississippi'
    d = defaultdict(int)
    for k in s:
        d[k] += 1
    print(d['w'])

def first_visit_mc_prediction(policy, env, n_episodes):
    value_table = defaultdict(float)
    N = defaultdict(int)
    gamma = 0.85
    for _ in range(n_episodes):
        states, _, rewards = generate_episode(policy, env)
        returns = 0.0
        for step in range(len(states)-1, -1, -1):
            returns *= gamma
            returns += rewards[step]
        for t in range(len(states)):
            S = states[t]
            R = rewards[t]
            if S not in states[:t]:
                N[S] += 1
                value_table[S] += (returns - value_table[S]) / N[S]
            returns -= R
            returns /= gamma
    return value_table

def plot_blackjack(V, ax1, ax2):
    player_sum = np.arange(12, 21 + 1)
    dealer_show = np.arange(1, 10 + 1)
    usable_ace = np.array([False, True])
    state_values = np.zeros((len(player_sum),
                                len(dealer_show),
                                len(usable_ace)))
    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = V[player, dealer, ace]
    print(np.shape(state_values))
    X, Y = np.meshgrid(player_sum, dealer_show)
    ax1.plot_wireframe(X, Y, state_values[:, :, 0])
    ax2.plot_wireframe(X, Y, state_values[:, :, 1])
    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('player sum')
        ax.set_xlabel('dealer showing')
        ax.set_zlabel('state-value')
    pyplot.show()

fig, axes = pyplot.subplots(nrows=2, figsize=(5, 8), subplot_kw={'projection': '3d'})
axes[0].set_title('value function without usable ace')
axes[1].set_title('value function with usable ace')

value = first_visit_mc_prediction(sample_policy, env, n_episodes=500000)

plot_blackjack(value, axes[0], axes[1])







