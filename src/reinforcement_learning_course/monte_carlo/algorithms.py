import numpy as np
from typing import List
from reinforcement_learning_course.core import Agent
from gymnasium import Env


class MonteCarlo(Agent[int, int]):
    """Monte-Carlo reinforcement learning agent.
    
    Implements the Monte-Carlo method for learning optimal policies through
    episode sampling and value estimation from complete trajectories.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99) -> None:
        """Initialize the Monte-Carlo agent.
        
        Args:
            env: Gymnasium environment wrapper.
            gamma: Discount factor, typically in [0, 1]. Defaults to 0.99.
        
        Attributes:
            env: The gymnasium environment.
            n_states: Number of states in the environment.
            n_actions: Number of actions available.
            q_values: Dictionary mapping states to arrays of state-action values.
            visits: Dictionary mapping states to visit counts for each action.
        """

        super().__init__(env, gamma)
        self.n_states, self.n_actions = self.set_n_states_actions()
        self.q_values = {}
        self.visits = {}

    def reset(self) -> None:
        """Reset the Q-values and visit counts.
        
        Clears all learned state-action values and visit statistics for training from scratch.
        """

        self.q_values = {}
        self.visits = {}

    def set_n_states_actions(self) -> tuple[int, int]:
        """Get the number of states and actions from the environment.
        
        Returns:
            A tuple containing:
                - n_states: Number of states in the environment.
                - n_actions: Number of actions available.
        """

        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        return n_states, n_actions
    
    def action_explore(self, state: int, epsilon: float) -> int:
        """Select an action using epsilon-greedy exploration.
        
        Args:
            state: The current state.
            epsilon: Exploration probability in (0, 1) - probability of taking a suboptimal action.
        
        Returns:
            The action to perform.
        """

        action_max = self.action(state)
        bern = np.random.binomial(1, 1 - epsilon)
        if bern == 1:
            return action_max
        
        return np.random.choice(self.n_actions)
        
    def action(self, state: int) -> int:
        """Select the best action according to estimated Q-values.
        
        If the state has been visited, returns the action with highest Q-value.
        Otherwise, returns a random action.
        
        Args:
            state: The current state.
        
        Returns:
            The estimated optimal action, or a random action if state is unseen.
        """

        if state in self.q_values:
            return self.q_values[state].argmax()
        
        return self.env.action_space.sample()
    
    def unroll(self, epsilon: float) -> tuple[List, List, List]:
        """Run a single episode using the epsilon-greedy policy.
        
        Generates a sequence of states, actions, and rewards from an initial state
        until a terminal state is reached.
        
        Args:
            epsilon: Exploration probability for the epsilon-greedy policy.
        
        Returns:
            A tuple containing:
                - states: List of visited states.
                - actions: List of actions taken.
                - rewards: List of rewards received.
        """

        states, actions, rewards = [], [], []
        state, _ = self.env.reset()
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated or truncated)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        return states, actions, rewards
    
    def train(self, epsilon_start: float = 1., epsilon_stop: float = 0.1, decay_rate: float = 1e-3, 
              n_train: int = 1000, first_visit: bool = True, print_iter: int = 10) -> None:
        """Train using the Monte-Carlo algorithm.
        
        Iteratively samples episodes and updates Q-values based on returns observed
        in each episode. Uses either first-visit or every-visit Monte-Carlo.
        
        Args:
            epsilon_start: Initial exploration rate. Defaults to 1.0.
            epsilon_stop: Final exploration rate. Defaults to 0.1.
            decay_rate: Exponential decay rate for epsilon. Defaults to 1e-3.
            n_train: Number of training episodes. Defaults to 1000.
            first_visit: If True, use first-visit MC (only first occurrence of state counts).
              If False, use every-visit MC. Defaults to True.
            print_iter: Print progress every print_iter episodes. Defaults to 10.
        """

        for i in range(n_train):
            epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*i)
            states, actions, rewards = self.unroll(epsilon)
            G = 0
            for t in range(-1, -len(states)-1, -1):
                state, action, reward = states[t], actions[t], rewards[t]
                G = self.gamma*G + reward
                if (first_visit and state not in set(states[:t])) or (not first_visit):
                    # If state has already been visited, update its estimated q-values and number of visits
                    if state in self.q_values:
                        self.q_values[state][action] = (self.q_values[state][action]*self.visits[state][action] + G)/(self.visits[state][action] + 1)
                        self.visits[state][action] = self.visits[state][action] + 1

                    # If state has never been visited, initialize its estimated q-values and number of visits and update them.
                    else:
                        self.q_values[state] = np.zeros(self.n_actions)
                        self.visits[state] = np.zeros(self.n_actions)
                        self.q_values[state][action] = G
                        self.visits[state][action] = 1

            if i%print_iter == 0:
                print('Iteration : %d , Epsilon : %.5f' %(i, epsilon))
                
