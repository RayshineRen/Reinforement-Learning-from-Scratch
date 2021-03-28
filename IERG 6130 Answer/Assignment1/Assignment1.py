# Import some packages that we need to use
from utils import *
import gym
import numpy as np
from collections import deque


# Solve the TODOs and remove `pass`

def _render_helper(env):
    env.render()
    wait(sleep=0.2)


def evaluate(policy, num_episodes, seed=0, env_name='FrozenLake8x8-v0', render=False):
    """[TODO] You need to implement this function by yourself. It
    evaluate the given policy and return the mean episode reward.
    We use `seed` argument for testing purpose.
    You should pass the tests in the next cell.

    :param policy: a function whose input is an interger (observation)
    :param num_episodes: number of episodes you wish to run
    :param seed: an interger, used for testing.
    :param env_name: the name of the environment
    :param render: a boolean flag. If true, please call _render_helper
    function.
    :return: the averaged episode reward of the given policy.
    """

    # Create environment (according to env_name, we will use env other than 'FrozenLake8x8-v0')
    env = gym.make(env_name)

    # Seed the environment
    env.seed(seed)

    # Build inner loop to run.
    # For each episode, do not set the limit.
    # Only terminate episode (reset environment) when done = True.
    # The episode reward is the sum of all rewards happen within one episode.
    # Call the helper function `render(env)` to render
    rewards = []
    for i in range(num_episodes):
        # reset the environment
        obs = env.reset()
        act = policy(obs)
        ep_reward = 0
        while True:
            # [TODO] run the environment and terminate it if done, collect the
            # reward at each step and sum them to the episode reward.
            obs, reward, done, info = env.step(act)
            act = policy(obs)
            ep_reward += reward
            if done:
                break

        rewards.append(ep_reward)

    return np.mean(rewards)

class TabularRLTrainerAbstract:
    """This is the abstract class for tabular RL trainer. We will inherent the specify
    algorithm's trainer from this abstract class, so that we can reuse the codes like
    getting the dynamic of the environment (self._get_transitions()) or rendering the
    learned policy (self.render())."""

    def __init__(self, env_name='FrozenLake8x8-v0', model_based=True):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.action_dim = self.env.action_space.n
        self.obs_dim = self.env.observation_space.n

        self.model_based = model_based

    def _get_transitions(self, state, act):
        """Query the environment to get the transition probability,
        reward, the next state, and done given a pair of state and action.
        We implement this function for you. But you need to know the
        return format of this function.
        """
        self._check_env_name()
        assert self.model_based, "You should not use _get_transitions in " \
                                 "model-free algorithm!"

        # call the internal attribute of the environments.
        # `transitions` is a list contain all possible next states and the
        # probability, reward, and termination indicater corresponding to it
        transitions = self.env.env.P[state][act]

        # Given a certain state and action pair, it is possible
        # to find there exist multiple transitions, since the
        # environment is not deterministic.
        # You need to know the return format of this function: a list of dicts
        ret = []
        for prob, next_state, reward, done in transitions:
            ret.append({
                "prob": prob,
                "next_state": next_state,
                "reward": reward,
                "done": done
            })
        return ret

    def _check_env_name(self):
        assert self.env_name.startswith('FrozenLake')

    def print_table(self):
        """print beautiful table, only work for FrozenLake8X8-v0 env. We
        write this function for you."""
        self._check_env_name()
        print_table(self.table)

    def train(self):
        """Conduct one iteration of learning."""
        raise NotImplementedError("You need to override the "
                                  "Trainer.train() function.")

    def evaluate(self):
        """Use the function you write to evaluate current policy.
        Return the mean episode reward of 1000 episodes when seed=0."""
        result = evaluate(self.policy, 1000, env_name=self.env_name)
        return result

    def render(self):
        """Reuse your evaluate function, render current policy
        for one episode when seed=0"""
        evaluate(self.policy, 1, render=True, env_name=self.env_name)


# Solve the TODOs and remove `pass`

class PolicyIterationTrainer(TabularRLTrainerAbstract):
    def __init__(self, gamma=1.0, eps=1e-10, env_name='FrozenLake8x8-v0'):
        super(PolicyIterationTrainer, self).__init__(env_name)

        # discount factor
        self.gamma = gamma

        # value function convergence criterion
        self.eps = eps

        # build the value table for each possible observation
        self.table = np.zeros((self.obs_dim,))

        # [TODO] you need to implement a random policy at the beginning.
        # It is a function that take an integer (state or say observation)
        # as input and return an interger (action).
        # remember, you can use self.action_dim to get the dimension (range)
        # of the action, which is an integer in range
        # [0, ..., self.action_dim - 1]
        # hint: generating random action at each call of policy may lead to
        #  failure of convergence, try generate random actions at initializtion
        #  and fix it during the training.
        #         self.policy = np.random.choice(self.action_dim, size=(self.obs_dim))
        def random_obs(obs):
            return np.random.choice(range(self.action_dim))

        policy_table = np.random.choice(self.action_dim, [self.obs_dim, ])
        self.policy = lambda s: policy_table[s]
        # test your random policy
        test_random_policy(self.policy, self.env)

    def train(self):
        """Conduct one iteration of learning."""
        # [TODO] value function may be need to be reset to zeros.
        # if you think it should, then do it. If not, then move on.
        # hint: the value function is equivalent to self.table,
        #  a numpy array with length 64.
        # self.table = 0*self.table
        # self.table = np.zeros((self.obs_dim,))
        self.update_value_function()
        self.update_policy()

    def update_value_function(self):
        count = 0  # count the steps of value updates
        while True:
            old_table = self.table.copy()

            for state in range(self.obs_dim):
                act = self.policy(state)
                transition_list = self._get_transitions(state, act)

                state_value = 0
                for transition in transition_list:
                    prob = transition['prob']
                    reward = transition['reward']
                    next_state = transition['next_state']
                    done = transition['done']

                    # [TODO] what is the right state value?
                    # hint: you should use reward, self.gamma, old_table, prob,
                    # and next_state to compute the state value
                    state_value += prob * (reward + self.gamma * old_table[next_state])

                # update the state value
                self.table[state] = state_value

            # [TODO] Compare the old_table and current table to
            #  decide whether to break the value update process.
            # hint: you should use self.eps, old_table and self.table
            should_break = None
            # if self.eps >= (np.sum(old_table) - np.sum(self.table)):
            #     should_break = True
            should_break = (np.abs(old_table - self.table) < self.eps).all()
            if should_break:
                break
            count += 1
            if count % 200 == 0:
                # disable this part if you think debug message annoying.
                print("[DEBUG]\tUpdated values for {} steps. "
                      "Difference between new and old table is: {}".format(
                    count, np.sum(np.abs(old_table - self.table))
                ))
            if count > 4000:
                print("[HINT] Are you sure your codes is OK? It shouldn't be "
                      "so hard to update the value function. You already "
                      "use {} steps to update value function within "
                      "single iteration.".format(count))
            if count > 6000:
                raise ValueError("Clearly your code has problem. Check it!")

    def update_policy(self):
        """You need to define a new policy function, given current
        value function. The best action for a given state is the one that
        has greatest expected return.

        To optimize computing efficiency, we introduce a policy table,
        which take state as index and return the action given a state.
        """
        policy_table = np.zeros([self.obs_dim, ], dtype=np.int)
        for state in range(self.obs_dim):
            state_action_values = [0] * self.action_dim
            # [TODO] assign the action with greatest "value"
            # to policy_table[state]
            # hint: what is the proper "value" here?
            #  you should use table, gamma, reward, prob,
            #  next_state and self._get_transitions() function
            #  as what we done at self.update_value_function()
            #  Bellman equation may help.
            for act in range(self.action_dim):
                transition_list = self._get_transitions(state, act)
                action_value = 0
                for transition in transition_list:
                    prob = transition['prob']
                    reward = transition['reward']
                    next_state = transition['next_state']
                    action_value += prob * (reward + self.gamma * self.table[next_state])
                state_action_values[act] = action_value
            best_action = np.argmax(state_action_values)
            policy_table[state] = best_action

        self.policy = lambda obs: policy_table[obs]


# Solve the TODOs and remove `pass`

# Managing configurations of your experiments is important for your research.
default_pi_config = dict(
    max_iteration=1000,
    evaluate_interval=1,
    gamma=1.0,
    eps=1e-10
)


def policy_iteration(train_config=None):
    config = default_pi_config.copy()
    if train_config is not None:
        config.update(train_config)

    trainer = PolicyIterationTrainer(gamma=config['gamma'], eps=config['eps'])

    old_policy_result = {
        obs: -1 for obs in range(trainer.obs_dim)
    }

    for i in range(config['max_iteration']):
        # train the agent
        trainer.train()  # [TODO] please uncomment this line
        # [TODO] compare the new policy with old policy to check whether
        #  should we stop. If new and old policy have same output given any
        #  observation, them we consider the algorithm is converged and
        #  should be stopped.
        new_policy = trainer.policy
        new_policy_result = {obs: new_policy(obs) for obs in range(trainer.obs_dim)}
        should_stop = True
        for obs in range(trainer.obs_dim):
            if new_policy_result[obs] != old_policy_result[obs]:
                should_stop = False
                break
        if should_stop:
            print("We found policy is not changed anymore at "
                  "iteration {}. Current mean episode reward "
                  "is {}. Stop training.".format(i, trainer.evaluate()))
            break
        old_policy_result = new_policy_result

        # evaluate the result
        if i % config['evaluate_interval'] == 0:
            print(
                "[INFO]\tIn {} iteration, current mean episode reward is {}."
                "".format(i, trainer.evaluate()))

            if i > 20:
                print("You sure your codes is OK? It shouldn't take so many "
                      "({}) iterations to train a policy iteration "
                      "agent.".format(i))

    assert trainer.evaluate() > 0.8, \
        "We expect to get the mean episode reward greater than 0.8. " \
        "But you get: {}. Please check your codes.".format(trainer.evaluate())

    return trainer

# pi_agent = policy_iteration()

# Solve the TODOs and remove `pass`


class ValueIterationTrainer(PolicyIterationTrainer):
    """Note that we inherate Policy Iteration Trainer, to resue the
    code of update_policy(). It's same since it get optimal policy from
    current state-value table (self.table).
    """

    def __init__(self, gamma=1.0, env_name='FrozenLake8x8-v0'):
        super(ValueIterationTrainer, self).__init__(gamma, None, env_name)

    def train(self):
        """Conduct one iteration of learning."""
        # [TODO] value function may be need to be reset to zeros.
        # if you think it should, then do it. If not, then move on.

        # In value iteration, we do not explicit require a
        # policy instance to run. We update value function
        # directly based on the transitions. Therefore, we
        # don't need to run self.update_policy() in each step.
        self.update_value_function()

    def update_value_function(self):
        old_table = self.table.copy()
        for state in range(self.obs_dim):
            state_value = 0
            # [TODO] what should be de right state value?
            # hint: try to compute the state_action_values first
            state_action_value = [0] * self.action_dim
            for act in range(self.action_dim):
                transition_list = self._get_transitions(state, act)
                action_value = 0
                for transition in transition_list:
                    prob = transition['prob']
                    reward = transition['reward']
                    next_state = transition['next_state']
                    done = transition['done']
                    action_value += prob * (reward + self.gamma * old_table[next_state])
                state_action_value[act] = action_value
            state_value = np.max(state_action_value)
            self.table[state] = state_value

        # Till now the one step value update is finished.
        # You can see that we do not use a inner loop to update
        # the value function like what we did in policy iteration.
        # This is because to compute the state value, which is
        # a expectation among all possible action given by a
        # specified policy, we **pretend** already own the optimal
        # policy (the max operation).

    def evaluate(self):
        """Since in value iteration we do not maintain a policy function,
        so we need to retrieve it when we need it."""
        self.update_policy()
        return super().evaluate()

    def render(self):
        """Since in value iteration we do not maintain a policy function,
        so we need to retrieve it when we need it."""
        self.update_policy()
        return super().render()


# Solve the TODOs and remove `pass`

# Managing configurations of your experiments is important for your research.
default_vi_config = dict(
    max_iteration=10000,
    evaluate_interval=100,  # don't need to update policy each iteration
    gamma=1.0,
    eps=1e-10
)


def value_iteration(train_config=None):
    config = default_vi_config.copy()
    if train_config is not None:
        config.update(train_config)

    # [TODO] initialize Value Iteration Trainer. Remember to pass
    #  config['gamma'] to it.
    trainer = ValueIterationTrainer(gamma=config['gamma'])
    old_state_value_table = trainer.table.copy()
    old_policy_result = {
        obs: -1 for obs in range(trainer.obs_dim)
    }
    for i in range(config['max_iteration']):
        # train the agent
        trainer.train()  # [TODO] please uncomment this line

        # evaluate the result
        if i % config['evaluate_interval'] == 0:
            print("[INFO]\tIn {} iteration, current "
                  "mean episode reward is {}.".format(
                i, trainer.evaluate()
            ))

            # [TODO] compare the new policy with old policy to check should
            #  we stop.
            # [HINT] If new and old policy have same output given any
            #  observation, them we consider the algorithm is converged and
            #  should be stopped.
            should_stop = True
            trainer.update_policy()
            new_policy_result = {
                obs: trainer.policy(obs) for obs in range(trainer.obs_dim)
            }
            for obs in range(trainer.obs_dim):
                if new_policy_result[obs] != old_policy_result[obs]:
                    should_stop = False
                    break
            old_policy_result = new_policy_result
            if should_stop:
                print("We found policy is not changed anymore at "
                      "iteration {}. Current mean episode reward "
                      "is {}. Stop training.".format(i, trainer.evaluate()))
                break

            if i > 3000:
                print("You sure your codes is OK? It shouldn't take so many "
                      "({}) iterations to train a policy iteration "
                      "agent.".format(
                    i))

    assert trainer.evaluate() > 0.8, \
        "We expect to get the mean episode reward greater than 0.8. " \
        "But you get: {}. Please check your codes.".format(trainer.evaluate())

    return trainer

# vi_agent = value_iteration()

# Solve the TODOs and remove `pass`

class SARSATrainer(TabularRLTrainerAbstract):
    def __init__(self,
                 gamma=1.0,
                 eps=0.1,
                 learning_rate=1.0,
                 max_episode_length=100,
                 env_name='FrozenLake8x8-v0'
                 ):
        super(SARSATrainer, self).__init__(env_name, model_based=False)

        # discount factor
        self.gamma = gamma

        # epsilon-greedy exploration policy parameter
        self.eps = eps

        # maximum steps in single episode
        self.max_episode_length = max_episode_length

        # the learning rate
        self.learning_rate = learning_rate

        # build the Q table
        # [TODO] uncomment the next line, pay attention to the shape
        self.table = np.zeros((self.obs_dim, self.action_dim))

    def policy(self, obs):
        """Implement epsilon-greedy policy

        It is a function that take an integer (state / observation)
        as input and return an interger (action).
        """

        # [TODO] You need to implement the epsilon-greedy policy here.
        # hint: We have self.eps probability to choose a unifomly random
        #  action in range [0, 1, .., self.action_dim - 1],
        #  otherwise choose action that maximize the Q value
        prob = np.random.random()
        if prob < self.eps:
            action = np.random.choice(range(self.action_dim))
        else:
            action = np.argmax(self.table[obs])
        return action
        # raise NotImplementedError()

    def train(self):
        """Conduct one iteration of learning."""
        # [TODO] Q table may be need to be reset to zeros.
        # if you think it should, then do it. If not, then move on.
        obs = self.env.reset()
        act = self.policy(obs)
        for t in range(self.max_episode_length):
            next_obs, reward, done, _ = self.env.step(act)
            next_act = self.policy(next_obs)
            # [TODO] compute the TD error, based on the next observation and
            #  action.
            td_error = reward + self.gamma*self.table[next_obs, next_act] - self.table[obs, act]
            # [TODO] compute the new Q value
            # hint: use TD error, self.learning_rate and old Q value
            new_value = self.learning_rate*td_error
            self.table[obs][act] += new_value
            # [TODO] Implement (1) break if done. (2) update obs for next
            #  self.policy(obs) call
            obs = next_obs
            act = next_act
            if done:
                break

# Run this cell without modification

# Managing configurations of your experiments is important for your research.
default_sarsa_config = dict(
    max_iteration=20000,
    max_episode_length=200,
    learning_rate=0.01,
    evaluate_interval=1000,
    gamma=0.8,
    eps=0.3,
    env_name='FrozenLakeNotSlippery-v0'
)


def sarsa(train_config=None):
    config = default_sarsa_config.copy()
    if train_config is not None:
        config.update(train_config)

    trainer = SARSATrainer(
        gamma=config['gamma'],
        eps=config['eps'],
        learning_rate=config['learning_rate'],
        max_episode_length=config['max_episode_length'],
        env_name=config['env_name']
    )

    for i in range(config['max_iteration']):
        # train the agent
        trainer.train()  # [TODO] please uncomment this line

        # evaluate the result
        if i % config['evaluate_interval'] == 0:
            print(
                "[INFO]\tIn {} iteration, current mean episode reward is {}."
                "".format(i, trainer.evaluate()))

    if trainer.evaluate() < 0.6:
        print("We expect to get the mean episode reward greater than 0.6. " \
        "But you get: {}. Please check your codes.".format(trainer.evaluate()))

    return trainer

# sarsa_trainer = sarsa()

# Solve the TODOs and remove `pass`

class QLearningTrainer(TabularRLTrainerAbstract):
    def __init__(self,
                 gamma=1.0,
                 eps=0.1,
                 learning_rate=1.0,
                 max_episode_length=100,
                 env_name='FrozenLake8x8-v0'
                 ):
        super(QLearningTrainer, self).__init__(env_name, model_based=False)
        self.gamma = gamma
        self.eps = eps
        self.max_episode_length = max_episode_length
        self.learning_rate = learning_rate

        # build the Q table
        self.table = np.zeros((self.obs_dim, self.action_dim))

    def policy(self, obs):
        """Implement epsilon-greedy policy

        It is a function that take an integer (state / observation)
        as input and return an interger (action).
        """

        # [TODO] You need to implement the epsilon-greedy policy here.
        # hint: Just copy your codes in SARSATrainer.policy()
        prob = np.random.random()
        if prob < self.eps:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(self.table[obs])
        return action

    def train(self):
        """Conduct one iteration of learning."""
        # [TODO] Q table may be need to be reset to zeros.
        # if you think it should, then do it. If not, then move on.
        obs = self.env.reset()
        for t in range(self.max_episode_length):
            act = self.policy(obs)

            next_obs, reward, done, _ = self.env.step(act)

            # [TODO] compute the TD error, based on the next observation
            # hint: we do not need next_act anymore.
            td_error = reward + self.gamma * np.max(self.table[next_obs]) - self.table[obs, act]
            # [TODO] compute the new Q value
            # hint: use TD error, self.learning_rate and old Q value
            new_value = self.table[obs,act]+self.learning_rate*td_error
            self.table[obs][act] = new_value
            obs = next_obs
            if done:
                break

# Run this cell without modification

# Managing configurations of your experiments is important for your research.
default_q_learning_config = dict(
    max_iteration=20000,
    max_episode_length=200,
    learning_rate=0.01,
    evaluate_interval=1000,
    gamma=0.8,
    eps=0.3,
    env_name='FrozenLakeNotSlippery-v0'
)


def q_learning(train_config=None):
    config = default_q_learning_config.copy()
    if train_config is not None:
        config.update(train_config)

    trainer = QLearningTrainer(
        gamma=config['gamma'],
        eps=config['eps'],
        learning_rate=config['learning_rate'],
        max_episode_length=config['max_episode_length'],
        env_name=config['env_name']
    )

    for i in range(config['max_iteration']):
        # train the agent
        trainer.train()  # [TODO] please uncomment this line

        # evaluate the result
        if i % config['evaluate_interval'] == 0:
            print(
                "[INFO]\tIn {} iteration, current mean episode reward is {}."
                "".format(i, trainer.evaluate()))

    if trainer.evaluate() < 0.6:
        print("We expect to get the mean episode reward greater than 0.6. " \
        "But you get: {}. Please check your codes.".format(trainer.evaluate()))

    return trainer

# q_learning_trainer = q_learning()

# Solve the TODOs and remove `pass`

class MCControlTrainer(TabularRLTrainerAbstract):
    def __init__(self,
                 gamma=1.0,
                 eps=0.3,
                 max_episode_length=100,
                 env_name='FrozenLake8x8-v0'
                 ):
        super(MCControlTrainer, self).__init__(env_name, model_based=False)
        self.gamma = gamma
        self.eps = eps
        self.max_episode_length = max_episode_length
        # build the dict of lists
        self.returns = {}
        for obs in range(self.obs_dim):
            for act in range(self.action_dim):
                self.returns[(obs, act)] = []
        # build the Q table
        self.table = np.zeros((self.obs_dim, self.action_dim))

    def policy(self, obs):
        """Implement epsilon-greedy policy
        It is a function that take an integer (state / observation)
        as input and return an interger (action).
        """

        # [TODO] You need to implement the epsilon-greedy policy here.
        # hint: Just copy your codes in SARSATrainer.policy()
        prob = np.random.random()
        if prob < self.eps:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(self.table[obs])
        return action

    def train(self):
        """Conduct one iteration of learning."""
        observations = []
        actions = []
        rewards = []
        # [TODO] rollout for one episode, store data in three lists create
        #  above.
        # hint: we do not need to store next observation.
        done = False
        obs = self.env.reset()
        while not done:
            observations.append(obs)
            act = self.policy(obs)
            actions.append(act)
            obs, reward, done, info = self.env.step(act)
            rewards.append(reward)
        assert len(actions) == len(observations)
        assert len(actions) == len(rewards)

        occured_state_action_pair = set()
        length = len(actions)
        value = 0
        for i in reversed(range(length)):
            # if length = 10, then i = 9, 8, ..., 0
            obs = observations[i]
            act = actions[i]
            reward = rewards[i]
            # [TODO] compute the value reversely
            # hint: value(t) = gamma * value(t+1) + r(t)
            value = self.gamma*value + reward
            if (obs, act) not in occured_state_action_pair:
                occured_state_action_pair.add((obs, act))
            # [TODO] append current return (value) to dict
            # hint: `value` represents the future return due to
            #  current (obs, act), so we need to store this value
            #  in trainer.returns with index (obs, act)
            self.returns[(obs, act)].append(value)
            # [TODO] compute the Q value from self.returns and write it
            #  into self.table
            # hint: the Q value is the mean of previous saw state-action value
            #  self.table is a 2D array so we can use table[obs, act] to get
            #  the Q value of state-action pair.
            self.table[obs, act] = np.mean(self.returns[(obs, act)])
            # we don't need to update the policy since it is
            # automatically adjusted with self.table

# Run this cell without modification

# Managing configurations of your experiments is important for your research.
default_mc_control_config = dict(
    max_iteration=20000,
    max_episode_length=200,
    evaluate_interval=1000,
    gamma=0.8,
    eps=0.3,
    env_name='FrozenLakeNotSlippery-v0'
)


def mc_control(train_config=None):
    config = default_mc_control_config.copy()
    if train_config is not None:
        config.update(train_config)

    trainer = MCControlTrainer(
        gamma=config['gamma'],
        eps=config['eps'],
        max_episode_length=config['max_episode_length'],
        env_name=config['env_name']
    )

    for i in range(config['max_iteration']):
        # train the agent
        trainer.train()

        # evaluate the result
        if i % config['evaluate_interval'] == 0:
            print(
                "[INFO]\tIn {} iteration, current mean episode reward is {}."
                "".format(i, trainer.evaluate()))

    if trainer.evaluate() < 0.6:
        print("We expect to get the mean episode reward greater than 0.6. " \
        "But you get: {}. Please check your codes.".format(trainer.evaluate()))

    return trainer

# mc_control_trainer = mc_control()
#
# sarsa_trainer = sarsa()




