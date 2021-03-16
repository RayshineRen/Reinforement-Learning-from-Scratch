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
env.reset()

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

def compute_policy_v(env, policy, gamma=1.0):
    v = np.zeros(env.env.nS)
    eps = 1e-20
    while True:
        pre_v = np.copy(v)
        for s in range(env.env.nS):
            policy_a = policy[s]
            v[s] = sum([p*(r+gamma*pre_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
        if(np.sum(np.fabs(v - pre_v)) <= eps):
            break
    return v

def policy_iteration(env, gamma=1.0):
    policy = np.random.choice(env.env.nA, size=(env.env.nS))
    max_iterations = 200000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if(np.all(policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        policy = new_policy
    return policy

if __name__ == '__main__':
    gamma = 0.5
    optimal_policy = policy_iteration(env, gamma)
    print(optimal_policy)
    run_episode(env, optimal_policy, gamma, True)
    scores = evaluate_policy(env, optimal_policy, gamma)
    print('Average scores = ', np.mean(scores))
















