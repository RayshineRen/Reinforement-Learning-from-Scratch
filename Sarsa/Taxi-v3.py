import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import copy

env = gym.make('Taxi-v3')
alpha = 0.3
gamma = 0.85
epslion = 0.02
episodes = 2000

Q_table = np.zeros((env.observation_space.n, env.action_space.n))

def epslion_greedy(state, epslion):
    if random.uniform(0, 1) < epslion:
        return env.action_space.sample()
    else:
        return np.argmax(Q_table[state])

def OutPolicy(env, Q_table):
    policy = []
    for state in range(env.observation_space.n):
        action = np.argmax(Q_table[state])
        policy.append(action)
    return policy

def evaluatePolicy(env, policy):
    obs = env.reset()
    env.render()
    while True:
        obs, reward, done, _ = env.step(int(policy[obs]))
        env.render()
        if done:
            break

r_record = []
error_Qtable = []
Q_old = copy.deepcopy(Q_table)

for episode in range(episodes):
    r = 0.0
    state = env.reset()
    while True:
        action = epslion_greedy(state, epslion)
        state_prime, reward, done, _ = env.step(action)
        action_prime = epslion_greedy(state_prime, epslion)
        Q_table[state][action] += alpha*(reward + gamma*Q_table[state_prime][action_prime] - Q_table[state][action])
        state = state_prime
        action = action_prime
        r += reward
        if done:
            break
    print("total reward: ", r)
    r_record.append(r)
    error_Qtable.append(np.sum(np.fabs(Q_table - Q_old)))
    Q_old = copy.deepcopy(Q_table)

env.close()

plt.figure(1, figsize=(10, 10))
plt.plot(list(range(episodes)), r_record[:episodes], linewidth=0.8)
plt.title('Reword Convergence Curve',fontsize=15)
plt.xlabel('Iteration',fontsize=15)
plt.ylabel('Total Reword of one episode',fontsize=15)

plt.figure(2, figsize=(10, 10))
plt.plot(list(range(episodes)), error_Qtable[:2000], linewidth=0.8)
plt.title('Q-table difference Convergence Curve',fontsize=15)
plt.xlabel('Iteration',fontsize=15)
plt.ylabel('Error between old and new Q-table',fontsize=15)

plt.show()

evaluatePolicy(env, OutPolicy(env, Q_table))





