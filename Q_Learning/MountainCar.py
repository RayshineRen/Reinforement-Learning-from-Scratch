import numpy as np
import gym
from gym import wrappers

off_policy = True # if True use off-policy q-learning update, if False, use on-policy SARSA update
n_states = 40
iter_max = 5000
initial_lr = 1.0 # Learning rate
min_lr = 0.003
gamma = 1.0
t_max = 10000
eps = 0.1

def obs_to_state(env, obs):
    """ Maps an observation to state """
    # we quantify the continous state space into discrete space
    env_low = env.observation_space.low
    env_high= env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    position = int((obs[0] - env_low[0]) / env_dx[0])
    velocity = int((obs[1] - env_low[1]) / env_dx[1])
    position = position if position < n_states else n_states - 1
    velocity = velocity if velocity < n_states else n_states - 1
    return position, velocity

def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            posi, velo = obs_to_state(env, obs)
            action = policy[posi][velo]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

if __name__ == '__main__':
    '''
    State: [position, velocity]，position 范围 [-0.6, 0.6]，velocity 范围 [-0.1, 0.1]
    Action: 0(向左推) 或 1(不动) 或 2(向右推)
    Reward: -1
    Done: 小车到达山顶或已花费200回合
    '''
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)
    if off_policy == True:
        print ('----- using Q Learning -----')
    else:
        print('------ using SARSA Learning ---')
    q_table = np.zeros((n_states, n_states, env.action_space.n))
    for i in range(iter_max):
        obs = env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eta = max(min_lr, initial_lr * (0.85 ** (i//100)))
        for j in range(t_max):
            a, b = obs_to_state(env, obs)
            if np.random.uniform(0, 1) < eps:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q_table[a][b])
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            a_, b_ = obs_to_state(env, obs)
            if off_policy == True:
                q_table[a][b][action] += eta*(reward + gamma*np.max(q_table[a_][b_]) - q_table[a][b][action])
            else:
                if np.random.uniform(0,1) < eps:
                    action_ = np.random.choice(env.action_space.n)
                else:
                    action_ = np.argmax(q_table[a_][b_])
                q_table[a][b][action] += eta*(reward + gamma*q_table[a_][b_][action_] - q_table[a][b][action])
            if done:
                break
        if i % 200 == 0:
            print('Iteration #%d -- Total reward = %d.' % (i + 1, total_reward))
    solution_policy = np.argmax(q_table, axis=2)
    print(solution_policy)
    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    for _ in range(2):
        run_episode(env, solution_policy, True)
    env.close()


