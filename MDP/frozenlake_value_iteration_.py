import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register
import matplotlib.pyplot as plt

def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.

    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.

    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0.0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma**step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward

def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
        run_episode(env, policy, gamma, False) for _ in range(n)
    ]
    return np.mean(scores)

def extract_policy(env, v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] += sum(p*(r + gamma*v[s_]) for p, s_, r, _ in env.env.P[s][a])
        policy[s] = np.argmax(q_sa)
    return policy

evaluatedPolicy = []
def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.env.nS)
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            q_sa = [sum([p*(r+gamma*prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
            v[s] = max(q_sa)
        evaluatedPolicy.append(evaluate_policy(env, extract_policy(env, v, gamma), gamma, 100))
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    return v

if __name__ == '__main__':
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    gamma = 1.0
    optimal_v = value_iteration(env, gamma)
    policy = extract_policy(env, optimal_v, gamma)
    policy_score = evaluate_policy(env, policy, gamma, 1000)
    print('Policy average score = ', policy_score)
    plt.plot(np.arange(1, len(evaluatedPolicy)+1), evaluatedPolicy)
    plt.show()












