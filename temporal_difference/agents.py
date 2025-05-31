from q_learning import *
from sarsa import SARSA
from expected_sarsa import ExpectedSARSA
from double_q_learning import DoubleQLearning
from collections import deque
from sklearn.preprocessing import KBinsDiscretizer

class FrozenLakeQLearning(QLearning):
    """
    Description
    --------------
    Class describing an agent operating in the FrozenLake environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(FrozenLakeQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class CliffWalkingQLearning(QLearning):
    """
    Description
    --------------
    Class describing an agent operating in the CliffWalking environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(CliffWalkingQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class TaxiQLearning(QLearning):
    """
    Description
    --------------
    Class describing an agent operating in the Taxi environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(TaxiQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class BlackJackQLearning(QLearning):
    """
    Description
    --------------
    Class describing an agent operating in the BlackJack environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(BlackJackQLearning, self).__init__(env, gamma)
        self.n_actions = env.action_space.n
        self.reset()

class CartPoleQLearning(QLearning):
    """
    Description
    --------------
    Class describing an agent operating in the CartPole environment.
    """
    
    def __init__(self, env, gamma=0.9, n_bins=5, strategy='uniform'):
        """
        Description
        --------------
        Constructor of class CartPoleQLearning.
        
        Arguments
        --------------
        env          : CartPole-v1 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(CartPoleQLearning, self).__init__(env, gamma)
        self.encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        self.n_actions = env.action_space.n
        self.reset()

    def pretrain(self, n_pretrain=100):
        """
        Description
        --------------
        Run a pretraining phase to discretise the state space.
        
        Arguments
        --------------
        """
        
        states = []
        for episode in range(n_pretrain):
            state, _ = self.env.reset()
            states.append(state)
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = (terminated or truncated)
                state = next_state
                states.append(state)

        states = np.vstack(states)
        self.encoder.fit(states)

    def unroll(self, alpha, epsilon):
        """
        Description
        --------------
        Unroll the current epsilon-greedy policy from state and update the q-values at each step.
        
        Arguments
        --------------
        alpha   : Float in ]0, 1[, learning rate.
        epsilon : Float in ]0, 1[, parameter of the epsilon-greedy policy.
        
        Returns
        --------------
        """

        state, _ = self.env.reset()
        state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
        reward_episode = 0
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
            reward_episode += reward
            done = (terminated or truncated)
            while not done and state == next_state:
                state = next_state
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                reward_episode += reward
                done = (terminated or truncated)

            self.update_q_value(state, action, reward, next_state, alpha)
            state = next_state

        return reward_episode

    def train(self, alpha=0.1, epsilon_start=1, epsilon_stop=0.1, decay_rate=1e-3, n_train=1000, print_iter=100, reward_stop=300):
        """
        Description
        --------------
        Train a Q-Learning algorithm.
        
        Arguments
        --------------
        alpha         : Float in ]0, 1[, learning rate.
        epsilon_start : Float in ]0, 1[, initial value of epsilon.
        epsilon_stopt : Float in ]0, 1[, final value of epsilon.
        decay_rate    : Float, decay rate of epsilon from epsilon_start to epsilon_stop.
        n_train       : Int, total number of iterations.
        print_iter    : Int, number of iterations between two successive prints.
        
        Returns
        --------------
        """

        rewards = deque(maxlen=100)
        reward_mean = None
        for i in range(n_train):
            epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*i)
            reward_episode = self.unroll(alpha, epsilon)
            if len(rewards) < rewards.maxlen:
                rewards.append(reward_episode)

            else:
                if reward_mean is None:
                    reward_mean = np.mean(rewards)

                else:
                    reward_mean += (reward_episode - rewards[0])/rewards.maxlen

                rewards.append(reward_episode)
                if reward_mean >= reward_stop:
                    print('Iteration : %d' %i)
                    print('Epsilon   : %.5f' %epsilon)
                    print('Reward    : %.3f' %reward_mean)
                    return

            if i%print_iter == 0:
                print('Iteration : %d' %i)
                print('Epsilon   : %.5f' %epsilon)
                if reward_mean is not None: print('Reward    : %.3f' %reward_mean)
                print('\n')

    def test(self, env, n_episodes=1000, verbose=False):
        """
        Description
        --------------
        Test the agent.
        
        Arguments
        --------------
        env        : gymnasium environment.
        n_episodes : Int, number of test episodes.
        verbose    : Boolean, if True, print the episode index and its corresponding length and return.
        
        Returns
        --------------
        """
        
        returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            state, _ = env.reset()
            state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
            done = False
            R = 0
            n_steps = 0
            while not done:
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                done = (terminated or truncated)
                while not done and state == next_state:
                    state = next_state
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                    done = (terminated or truncated)

                state = next_state
                R += reward
                n_steps += 1
                
            returns[episode] = R
            if verbose:
                print('Episode : %d, length : %d, return : %.3F' %(episode, n_steps, R))

        return_avg, return_std = returns.mean(), returns.std()
        print('avg : %.3f, std : %.3f' %(return_avg, return_std))
        return return_avg, return_std
            
    def save_gif(self, env, file_name, n_episodes=1, duration=20):
        """
        Description
        --------------
        Test the agent and save a gif.
        
        Arguments
        --------------
        env       : gymnasium environment.
        file_name : String, path to the saved gif.
        
        Returns
        --------------
        """
        
        frames = []
        for i in range(n_episodes):
            state, _ = env.reset()
            state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
            done = False
            R = 0
            n_steps = 0
            while not done:
                frames.append(Image.fromarray(env.render(), mode='RGB'))
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                done = (terminated or truncated)
                while not done and state == next_state:
                    state = next_state
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                    done = (terminated or truncated)
                    
                state = next_state
                R += reward
                n_steps += 1

            frames.append(Image.fromarray(env.render(), mode='RGB'))
            
        frames[0].save(file_name, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)

class FrozenLakeSARSA(SARSA):
    """
    Description
    --------------
    Class describing an agent operating in the FrozenLake environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(FrozenLakeSARSA, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class CliffWalkingSARSA(SARSA):
    """
    Description
    --------------
    Class describing an agent operating in the CliffWalking environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(CliffWalkingSARSA, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class TaxiSARSA(SARSA):
    """
    Description
    --------------
    Class describing an agent operating in the Taxi environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(TaxiSARSA, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class BlackJackSARSA(SARSA):
    """
    Description
    --------------
    Class describing an agent operating in the BlackJack environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(BlackJackSARSA, self).__init__(env, gamma)
        self.n_actions = env.action_space.n
        self.reset()

class CartPoleSARSA(SARSA):
    """
    Description
    --------------
    Class describing an agent operating in the CartPole environment.
    """
    
    def __init__(self, env, gamma=0.9, n_bins=5, strategy='uniform'):
        """
        Description
        --------------
        Constructor of class CartPoleQLearning.
        
        Arguments
        --------------
        env          : CartPole-v1 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(CartPoleSARSA, self).__init__(env, gamma)
        self.encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        self.n_actions = env.action_space.n
        self.reset()

    def pretrain(self, n_pretrain=100):
        """
        Description
        --------------
        Run a pretraining phase to discretise the state space.
        
        Arguments
        --------------
        """
        
        states = []
        for episode in range(n_pretrain):
            state, _ = self.env.reset()
            states.append(state)
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = (terminated or truncated)
                state = next_state
                states.append(state)

        states = np.vstack(states)
        self.encoder.fit(states)

    def unroll(self, alpha, epsilon):
        """
        Description
        --------------
        Unroll the current epsilon-greedy policy from state and update the q-values at each step.
        
        Arguments
        --------------
        alpha   : Float in ]0, 1[, learning rate.
        epsilon : Float in ]0, 1[, parameter of the epsilon-greedy policy.
        
        Returns
        --------------
        """

        state, _ = self.env.reset()
        state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
        reward_episode = 0
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
            reward_episode += reward
            done = (terminated or truncated)
            while not done and state == next_state:
                state = next_state
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                reward_episode += reward
                done = (terminated or truncated)

            self.update_q_value(state, action, reward, next_state, alpha, epsilon)
            state = next_state

        return reward_episode

    def train(self, alpha=0.1, epsilon_start=1, epsilon_stop=0.1, decay_rate=1e-3, n_train=1000, print_iter=100, reward_stop=300):
        """
        Description
        --------------
        Train a Q-Learning algorithm.
        
        Arguments
        --------------
        alpha         : Float in ]0, 1[, learning rate.
        epsilon_start : Float in ]0, 1[, initial value of epsilon.
        epsilon_stop  : Float in ]0, 1[, final value of epsilon.
        decay_rate    : Float, decay rate of epsilon from epsilon_start to epsilon_stop.
        n_train       : Int, total number of iterations.
        print_iter    : Int, number of iterations between two successive prints.
        
        Returns
        --------------
        """

        rewards = deque(maxlen=100)
        reward_mean = None
        for i in range(n_train):
            epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*i)
            reward_episode = self.unroll(alpha, epsilon)
            if len(rewards) < rewards.maxlen:
                rewards.append(reward_episode)

            else:
                if reward_mean is None:
                    reward_mean = np.mean(rewards)

                else:
                    reward_mean += (reward_episode - rewards[0])/rewards.maxlen

                rewards.append(reward_episode)
                if reward_mean >= reward_stop:
                    print('Iteration : %d' %i)
                    print('Epsilon   : %.5f' %epsilon)
                    print('Reward    : %.3f' %reward_mean)
                    return

            if i%print_iter == 0:
                print('Iteration : %d' %i)
                print('Epsilon   : %.5f' %epsilon)
                if reward_mean is not None: print('Reward    : %.3f' %reward_mean)
                print('\n')

    def test(self, env, n_episodes=1000, verbose=False):
        """
        Description
        --------------
        Test the agent.
        
        Arguments
        --------------
        env        : gymnasium environment.
        n_episodes : Int, number of test episodes.
        verbose    : Boolean, if True, print the episode index and its corresponding length and return.
        
        Returns
        --------------
        """
        
        returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            state, _ = env.reset()
            state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
            done = False
            R = 0
            n_steps = 0
            while not done:
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                done = (terminated or truncated)
                while not done and state == next_state:
                    state = next_state
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                    done = (terminated or truncated)

                state = next_state
                R += reward
                n_steps += 1
                
            returns[episode] = R
            if verbose:
                print('Episode : %d, length : %d, return : %.3F' %(episode, n_steps, R))

        return_avg, return_std = returns.mean(), returns.std()
        print('avg : %.3f, std : %.3f' %(return_avg, return_std))
        return return_avg, return_std
            
    def save_gif(self, env, file_name, n_episodes=1, duration=20):
        """
        Description
        --------------
        Test the agent and save a gif.
        
        Arguments
        --------------
        env       : gymnasium environment.
        file_name : String, path to the saved gif.
        
        Returns
        --------------
        """
        
        frames = []
        for i in range(n_episodes):
            state, _ = env.reset()
            state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
            done = False
            R = 0
            n_steps = 0
            while not done:
                frames.append(Image.fromarray(env.render(), mode='RGB'))
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                done = (terminated or truncated)
                while not done and state == next_state:
                    state = next_state
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                    done = (terminated or truncated)
                    
                state = next_state
                R += reward
                n_steps += 1

            frames.append(Image.fromarray(env.render(), mode='RGB'))
            
        frames[0].save(file_name, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)

class FrozenLakeExpectedSarsa(ExpectedSARSA):
    """
    Description
    --------------
    Class describing an agent operating in the FrozenLake environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(FrozenLakeExpectedSarsa, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class CliffWalkingExpectedSARSA(ExpectedSARSA):
    """
    Description
    --------------
    Class describing an agent operating in the CliffWalking environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(CliffWalkingExpectedSARSA, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class TaxiExpectedSARSA(ExpectedSARSA):
    """
    Description
    --------------
    Class describing an agent operating in the Taxi environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(TaxiExpectedSARSA, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class BlackJackExpectedSARSA(ExpectedSARSA):
    """
    Description
    --------------
    Class describing an agent operating in the BlackJack environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(BlackJackExpectedSARSA, self).__init__(env, gamma)
        self.n_actions = env.action_space.n
        self.reset()

class CartPoleExpectedSARSA(ExpectedSARSA):
    """
    Description
    --------------
    Class describing an agent operating in the CartPole environment.
    """
    
    def __init__(self, env, gamma=0.9, n_bins=5, strategy='uniform'):
        """
        Description
        --------------
        Constructor of class CartPoleQLearning.
        
        Arguments
        --------------
        env          : CartPole-v1 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(CartPoleExpectedSARSA, self).__init__(env, gamma)
        self.encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        self.n_actions = env.action_space.n
        self.reset()

    def pretrain(self, n_pretrain=100):
        """
        Description
        --------------
        Run a pretraining phase to discretise the state space.
        
        Arguments
        --------------
        """
        
        states = []
        for episode in range(n_pretrain):
            state, _ = self.env.reset()
            states.append(state)
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = (terminated or truncated)
                state = next_state
                states.append(state)

        states = np.vstack(states)
        self.encoder.fit(states)

    def unroll(self, alpha, epsilon):
        """
        Description
        --------------
        Unroll the current epsilon-greedy policy from state and update the q-values at each step.
        
        Arguments
        --------------
        alpha   : Float in ]0, 1[, learning rate.
        epsilon : Float in ]0, 1[, parameter of the epsilon-greedy policy.
        
        Returns
        --------------
        """

        state, _ = self.env.reset()
        state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
        reward_episode = 0
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
            reward_episode += reward
            done = (terminated or truncated)
            while not done and state == next_state:
                state = next_state
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                reward_episode += reward
                done = (terminated or truncated)

            self.update_q_value(state, action, reward, next_state, alpha, epsilon)
            state = next_state

        return reward_episode

    def train(self, alpha=0.1, epsilon_start=1, epsilon_stop=0.1, decay_rate=1e-3, n_train=1000, print_iter=100, reward_stop=300):
        """
        Description
        --------------
        Train a Q-Learning algorithm.
        
        Arguments
        --------------
        alpha         : Float in ]0, 1[, learning rate.
        epsilon_start : Float in ]0, 1[, initial value of epsilon.
        epsilon_stop  : Float in ]0, 1[, final value of epsilon.
        decay_rate    : Float, decay rate of epsilon from epsilon_start to epsilon_stop.
        n_train       : Int, total number of iterations.
        print_iter    : Int, number of iterations between two successive prints.
        
        Returns
        --------------
        """

        rewards = deque(maxlen=100)
        reward_mean = None
        for i in range(n_train):
            epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*i)
            reward_episode = self.unroll(alpha, epsilon)
            if len(rewards) < rewards.maxlen:
                rewards.append(reward_episode)

            else:
                if reward_mean is None:
                    reward_mean = np.mean(rewards)

                else:
                    reward_mean += (reward_episode - rewards[0])/rewards.maxlen

                rewards.append(reward_episode)
                if reward_mean >= reward_stop:
                    print('Iteration : %d' %i)
                    print('Epsilon   : %.5f' %epsilon)
                    print('Reward    : %.3f' %reward_mean)
                    return

            if i%print_iter == 0:
                print('Iteration : %d' %i)
                print('Epsilon   : %.5f' %epsilon)
                if reward_mean is not None: print('Reward    : %.3f' %reward_mean)
                print('\n')

    def test(self, env, n_episodes=1000, verbose=False):
        """
        Description
        --------------
        Test the agent.
        
        Arguments
        --------------
        env        : gymnasium environment.
        n_episodes : Int, number of test episodes.
        verbose    : Boolean, if True, print the episode index and its corresponding length and return.
        
        Returns
        --------------
        """
        
        returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            state, _ = env.reset()
            state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
            done = False
            R = 0
            n_steps = 0
            while not done:
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                done = (terminated or truncated)
                while not done and state == next_state:
                    state = next_state
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                    done = (terminated or truncated)

                state = next_state
                R += reward
                n_steps += 1
                
            returns[episode] = R
            if verbose:
                print('Episode : %d, length : %d, return : %.3F' %(episode, n_steps, R))

        return_avg, return_std = returns.mean(), returns.std()
        print('avg : %.3f, std : %.3f' %(return_avg, return_std))
        return return_avg, return_std
            
    def save_gif(self, env, file_name, n_episodes=1, duration=20):
        """
        Description
        --------------
        Test the agent and save a gif.
        
        Arguments
        --------------
        env       : gymnasium environment.
        file_name : String, path to the saved gif.
        
        Returns
        --------------
        """
        
        frames = []
        for i in range(n_episodes):
            state, _ = env.reset()
            state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
            done = False
            R = 0
            n_steps = 0
            while not done:
                frames.append(Image.fromarray(env.render(), mode='RGB'))
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                done = (terminated or truncated)
                while not done and state == next_state:
                    state = next_state
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                    done = (terminated or truncated)
                    
                state = next_state
                R += reward
                n_steps += 1

            frames.append(Image.fromarray(env.render(), mode='RGB'))
            
        frames[0].save(file_name, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)

class FrozenLakeDoubleQLearning(DoubleQLearning):
    """
    Description
    --------------
    Class describing an agent operating in the FrozenLake environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(FrozenLakeDoubleQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class CliffWalkingDoubleQLearning(DoubleQLearning):
    """
    Description
    --------------
    Class describing an agent operating in the CliffWalking environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(CliffWalkingDoubleQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class TaxiDoubleQLearning(DoubleQLearning):
    """
    Description
    --------------
    Class describing an agent operating in the Taxi environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(TaxiDoubleQLearning, self).__init__(env, gamma)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.reset()

class BlackJackDoubleQLearning(DoubleQLearning):
    """
    Description
    --------------
    Class describing an agent operating in the BlackJack environment.
    """
    
    def __init__(self, env, gamma=0.9):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : CliffWalking-v0 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(BlackJackDoubleQLearning, self).__init__(env, gamma)
        self.n_actions = env.action_space.n
        self.reset()

class CartPoleDoubleQLearning(DoubleQLearning):
    """
    Description
    --------------
    Class describing an agent operating in the CartPole environment.
    """
    
    def __init__(self, env, gamma=0.9, n_bins=5, strategy='uniform'):
        """
        Description
        --------------
        Constructor of class CartPoleQLearning.
        
        Arguments
        --------------
        env          : CartPole-v1 environment.
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        q_values     : np.array of shape (n_states, n_actions) or None, q-values.
        """
        
        super(CartPoleDoubleQLearning, self).__init__(env, gamma)
        self.encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        self.n_actions = env.action_space.n
        self.reset()

    def pretrain(self, n_pretrain=100):
        """
        Description
        --------------
        Run a pretraining phase to discretise the state space.
        
        Arguments
        --------------
        """
        
        states = []
        for episode in range(n_pretrain):
            state, _ = self.env.reset()
            states.append(state)
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = (terminated or truncated)
                state = next_state
                states.append(state)

        states = np.vstack(states)
        self.encoder.fit(states)

    def unroll(self, alpha, epsilon):
        """
        Description
        --------------
        Unroll the current epsilon-greedy policy from state and update the q-values at each step.
        
        Arguments
        --------------
        alpha   : Float in ]0, 1[, learning rate.
        epsilon : Float in ]0, 1[, parameter of the epsilon-greedy policy.
        
        Returns
        --------------
        """

        state, _ = self.env.reset()
        state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
        reward_episode = 0
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
            reward_episode += reward
            done = (terminated or truncated)
            while not done and state == next_state:
                state = next_state
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                reward_episode += reward
                done = (terminated or truncated)

            self.update_q_value(state, action, reward, next_state, alpha)
            state = next_state

        return reward_episode

    def train(self, alpha=0.1, epsilon_start=1, epsilon_stop=0.1, decay_rate=1e-3, n_train=1000, print_iter=100, reward_stop=300):
        """
        Description
        --------------
        Train a Q-Learning algorithm.
        
        Arguments
        --------------
        alpha         : Float in ]0, 1[, learning rate.
        epsilon_start : Float in ]0, 1[, initial value of epsilon.
        epsilon_stopt : Float in ]0, 1[, final value of epsilon.
        decay_rate    : Float, decay rate of epsilon from epsilon_start to epsilon_stop.
        n_train       : Int, total number of iterations.
        print_iter    : Int, number of iterations between two successive prints.
        
        Returns
        --------------
        """

        rewards = deque(maxlen=100)
        reward_mean = None
        for i in range(n_train):
            epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*i)
            reward_episode = self.unroll(alpha, epsilon)
            if len(rewards) < rewards.maxlen:
                rewards.append(reward_episode)

            else:
                if reward_mean is None:
                    reward_mean = np.mean(rewards)

                else:
                    reward_mean += (reward_episode - rewards[0])/rewards.maxlen

                rewards.append(reward_episode)
                if reward_mean >= reward_stop:
                    print('Iteration : %d' %i)
                    print('Epsilon   : %.5f' %epsilon)
                    print('Reward    : %.3f' %reward_mean)
                    return

            if i%print_iter == 0:
                print('Iteration : %d' %i)
                print('Epsilon   : %.5f' %epsilon)
                if reward_mean is not None: print('Reward    : %.3f' %reward_mean)
                print('\n')

    def test(self, env, n_episodes=1000, verbose=False):
        """
        Description
        --------------
        Test the agent.
        
        Arguments
        --------------
        env        : gymnasium environment.
        n_episodes : Int, number of test episodes.
        verbose    : Boolean, if True, print the episode index and its corresponding length and return.
        
        Returns
        --------------
        """
        
        returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            state, _ = env.reset()
            state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
            done = False
            R = 0
            n_steps = 0
            while not done:
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                done = (terminated or truncated)
                while not done and state == next_state:
                    state = next_state
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                    done = (terminated or truncated)

                state = next_state
                R += reward
                n_steps += 1
                
            returns[episode] = R
            if verbose:
                print('Episode : %d, length : %d, return : %.3F' %(episode, n_steps, R))

        return_avg, return_std = returns.mean(), returns.std()
        print('avg : %.3f, std : %.3f' %(return_avg, return_std))
        return return_avg, return_std
            
    def save_gif(self, env, file_name, n_episodes=1, duration=20):
        """
        Description
        --------------
        Test the agent and save a gif.
        
        Arguments
        --------------
        env       : gymnasium environment.
        file_name : String, path to the saved gif.
        
        Returns
        --------------
        """
        
        frames = []
        for i in range(n_episodes):
            state, _ = env.reset()
            state = tuple(self.encoder.transform(state.reshape((1, -1))).flatten())
            done = False
            R = 0
            n_steps = 0
            while not done:
                frames.append(Image.fromarray(env.render(), mode='RGB'))
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                done = (terminated or truncated)
                while not done and state == next_state:
                    state = next_state
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = tuple(self.encoder.transform(next_state.reshape((1, -1))).flatten())
                    done = (terminated or truncated)
                    
                state = next_state
                R += reward
                n_steps += 1

            frames.append(Image.fromarray(env.render(), mode='RGB'))
            
        frames[0].save(file_name, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)