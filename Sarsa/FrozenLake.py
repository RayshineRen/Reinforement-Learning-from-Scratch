import gym, sys, numpy as np
from gym.envs.registration import register
from gridworld import FrozenLakeWapper
import time

no_slippery = True
render_last = True # whether to visualize the last episode in testing

episodes = 10000
num_iter = 100
learning_rate = 0.01
gamma = 0.8
eps = 0.3

if no_slippery == True:
    # the simplified frozen lake without slippery (so the transition is deterministic)
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery': False},
        max_episode_steps=1000,
        reward_threshold=0.78, # optimum = .8196
    )
    env = gym.make('FrozenLakeNotSlippery-v0')
else:
    # the standard slippery frozen lake
    env = gym.make('FrozenLake-v0')

def eps_greedy(state, eps):
    if np.random.uniform(0, 1) < eps:
        action = np.random.choice(env.action_space.n)
    else:
        action = np.argmax(Q_table[state])
    return action

Q_table = np.zeros((env.observation_space.n, env.action_space.n))

for episode in range(episodes):
    state = env.reset()
    action = eps_greedy(state, eps)
    for iter in range(num_iter):
        state_new, reward, done, _ = env.step(action)
        action_new = eps_greedy(state_new, eps)
        Q_table[state][action] += learning_rate*(reward+gamma*Q_table[state_new][action_new] - Q_table[state][action])
        state = state_new
        action = action_new
        if done:
            break
if no_slippery == True:
    print('---Frozenlake without slippery move-----')
else:
    print('---Standard frozenlake------------------')

num_episode = 500
rewards = 0.0
for epi in range(num_episode):
    s = env.reset()
    for _ in range(100):
        action = np.argmax(Q_table[s])
        s_new, reward, done, _ = env.step(action)
        if epi == num_episode - 1 and render_last:
            env.render()
        s = s_new
        if done:
            if reward == 1:
                rewards += 1
            break

print('---Success rate=%.3f'%(rewards*1.0 / num_episode))
print('-------------------------------')

def test_alg(env):
    env = FrozenLakeWapper(env)
    s = env.reset()
    for _ in range(100):
        env.render()
        time.sleep(1)
        action = np.argmax(Q_table[s])
        s_new, reward, done, _ = env.step(action)
        s = s_new
        if done:
            env.render()
            time.sleep(2)
            break

test_alg(env)