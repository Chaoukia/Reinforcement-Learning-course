import numpy as np
from reinforcement_learning_course.core import Agent
from gymnasium import Env


class TemporalDifference(Agent[int, int]):
    """Base class for Temporal Difference learning agents.
    
    Implements the core structure for TD-based methods which learn from
    bootstrapping using single-step transitions.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99) -> None:
        """Initialize the Temporal Difference agent.
        
        Args:
            env: Gymnasium environment wrapper.
            gamma: Discount factor, typically in [0, 1]. Defaults to 0.99.
        
        Attributes:
            n_actions: Number of actions available.
            q_values: Dictionary mapping states to action values.
            visits: Dictionary tracking visit counts for each state-action pair.
        """
        super().__init__(env, gamma)
        self.n_actions = self.env.action_space.n
        self.q_values = {}
        self.visits = {}

    def reset(self) -> None:
        """Reset the Q-values and visit counts.
        
        Clears all learned state-action values and visit statistics for training from scratch.
        """

        self.q_values = {}
        self.visits = {}

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
        
        return self.env.action_space.sample()
        
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
    
    def unroll(self, alpha: float, epsilon: float) -> None:
        """Run a single episode with temporal difference updates.
        
        Must be implemented by subclasses with specific TD variant (SARSA, Q-learning, etc).
        
        Args:
            alpha: Learning rate in (0, 1).
            epsilon: Exploration parameter for epsilon-greedy policy.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError
    
    def train(self, 
              alpha: float = 0.1, 
              epsilon_start: float = 1., 
              epsilon_stop: float = 0.1, 
              decay_rate: float = 1e-3, 
              n_train: int = 1000, 
              print_iter: int = 10
              ) -> None:
        """Train using Temporal Difference learning.
        
        Iteratively samples episodes and updates Q-values using TD bootstrapping.
        
        Args:
            alpha: Learning rate in (0, 1). Defaults to 0.1.
            epsilon_start: Initial exploration rate. Defaults to 1.0.
            epsilon_stop: Final exploration rate. Defaults to 0.1.
            decay_rate: Exponential decay rate for epsilon. Defaults to 1e-3.
            n_train: Number of training episodes. Defaults to 1000.
            print_iter: Print progress every print_iter episodes. Defaults to 10.
        """

        for i in range(n_train):
            epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*i)
            self.unroll(alpha, epsilon)
            if i%print_iter == 0:
                print('Iteration : %d' %i)
                print('Epsilon   : %.5f' %epsilon)
                print('\n')


class SARSA(TemporalDifference):
    """SARSA (State-Action-Reward-State-Action) on-policy TD learning agent.
    
    Implements the SARSA algorithm which learns the value of the policy being
    followed by using the next state-action pair to bootstrap.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initialize the SARSA agent.
        
        Args:
            env: Gymnasium environment wrapper.
            gamma: Discount factor. Defaults to 0.99.
        """
        super().__init__(env, gamma)

    def update_q_value(self, 
                       state: int, 
                       action: int, 
                       reward: float, 
                       next_state: int, 
                       alpha: float, 
                       epsilon: float
                       ) -> None:
        """Update Q-value using SARSA update rule.
        
        Updates Q(state, action) using the next action sampled from the policy.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            alpha: Learning rate.
            epsilon: Exploration parameter for selecting next action.
        """

        if next_state in self.q_values:
            action_next = self.action_explore(next_state, epsilon)
            q_next_state = self.q_values[next_state][action_next]

        else:
            self.q_values[next_state], self.visits[next_state] = np.zeros(self.n_actions), np.zeros(self.n_actions)
            q_next_state = 0

        if state in self.q_values:
            q_state = self.q_values[state][action]

        else:
            self.q_values[state], self.visits[state] = np.zeros(self.n_actions), np.zeros(self.n_actions)
            q_state = 0

        td = reward + self.gamma*q_next_state - q_state
        self.visits[state][action] += 1
        if alpha is None:
            alpha = 1/(self.visits[state][action])

        self.q_values[state][action] += alpha*td
        
    def unroll(self, alpha: float, epsilon: float):
        """Run a single SARSA episode with Q-value updates.
        
        Args:
            alpha: Learning rate.
            epsilon: Exploration parameter.
        """

        state, _ = self.env.reset()
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated or truncated)
            self.update_q_value(state, action, reward, next_state, alpha, epsilon)
            state = next_state


class ExpectedSARSA(TemporalDifference):
    """Expected SARSA on-policy TD learning agent.
    
    Similar to SARSA but uses expected value of next action instead of sampled action,
    providing more stable learning.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initialize the Expected SARSA agent.
        
        Args:
            env: Gymnasium environment wrapper.
            gamma: Discount factor. Defaults to 0.99.
        """
        super().__init__(env, gamma)

    def update_q_value(self, 
                       state: int, 
                       action: int, 
                       reward: float, 
                       next_state: int, 
                       alpha: float, 
                       epsilon: float
                       ) -> None:
        """Update Q-value using Expected SARSA update rule.
        
        Updates Q(state, action) using expected value of epsilon-greedy policy
        at the next state.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            alpha: Learning rate.
            epsilon: Exploration parameter for computing expected value.
        """

        if next_state in self.q_values:
            weights = np.full_like(self.q_values[next_state], epsilon/(self.n_actions))
            weights[self.q_values[next_state].argmax()] = 1 - epsilon + epsilon/(self.n_actions)
            q_next_state = (self.q_values[next_state]*weights).sum()

        else:
            self.q_values[next_state], self.visits[next_state] = np.zeros(self.n_actions), np.zeros(self.n_actions)
            q_next_state = 0
            
        if state in self.q_values:
            q_state = self.q_values[state][action]

        else:
            self.q_values[state], self.visits[state] = np.zeros(self.n_actions), np.zeros(self.n_actions)
            q_state = 0
            
        td = reward + self.gamma*q_next_state - q_state
        self.visits[state][action] += 1
        if alpha is None:
            alpha = 1/(self.visits[state][action])

        self.q_values[state][action] += alpha*td
        
    def unroll(self, alpha: float, epsilon: float):
        """Run a single Expected SARSA episode with Q-value updates.
        
        Args:
            alpha: Learning rate.
            epsilon: Exploration parameter.
        """

        state, _ = self.env.reset()
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated or truncated)
            self.update_q_value(state, action, reward, next_state, alpha, epsilon)
            state = next_state


class QLearning(TemporalDifference):
    """Q-Learning off-policy TD learning agent.
    
    Implements Q-learning which learns the value of the optimal policy while
    following an exploratory epsilon-greedy policy.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initialize the Q-Learning agent.
        
        Args:
            env: Gymnasium environment wrapper.
            gamma: Discount factor. Defaults to 0.99.
        """
        super().__init__(env, gamma)

    def update_q_value(self, 
                       state: int, 
                       action: int, 
                       reward: float, 
                       next_state: int, 
                       alpha: float, 
                       ) -> None:
        """Update Q-value using Q-learning update rule.
        
        Updates Q(state, action) using the maximum Q-value at next state.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            alpha: Learning rate.
        """

        if next_state in self.q_values:
            q_next_state = self.q_values[next_state].max()

        else:
            self.q_values[next_state], self.visits[next_state] = np.zeros(self.n_actions), np.zeros(self.n_actions)
            q_next_state = 0

        if state in self.q_values:
            q_state = self.q_values[state][action]

        else:
            self.q_values[state], self.visits[state] = np.zeros(self.n_actions), np.zeros(self.n_actions)
            q_state = 0

        td = reward + self.gamma*q_next_state - q_state
        self.visits[state][action] += 1
        if alpha is None:
            alpha = 1/(self.visits[state][action])

        self.q_values[state][action] += alpha*td
        
    def unroll(self, alpha: float, epsilon: float):
        """Run a single Q-learning episode with Q-value updates.
        
        Args:
            alpha: Learning rate.
            epsilon: Exploration parameter (only used for action selection).
        """

        state, _ = self.env.reset()
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated or truncated)
            self.update_q_value(state, action, reward, next_state, alpha)
            state = next_state


class DoubleQLearning(TemporalDifference):
    """Double Q-Learning off-policy TD learning agent.
    
    Reduces overestimation bias in Q-learning by using two separate Q-value
    estimators and alternating which one is used for bootstrap updates.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initialize the Double Q-Learning agent.
        
        Args:
            env: Gymnasium environment wrapper.
            gamma: Discount factor. Defaults to 0.99.
        """

        super().__init__(env, gamma)
        self.q_values_1, self.q_values_2, self.visits = {}, {}, {}

    def reset(self) -> None:
        """Reset both Q-value networks and visit counts.
        
        Clears learned state-action values for training from scratch.
        """

        self.q_values_1, self.q_values_2, self.visits = {}, {}, {}

    def action(self, state: int):
        """Select best action using average of both Q-value networks.
        
        If state has been visited, returns action with max average Q-value.
        Otherwise returns random action.
        
        Args:
            state: The current state.
        
        Returns:
            The best action according to averaged Q-values, or random action if unseen.
        """

        if state in self.q_values_1:
            return (self.q_values_1[state] + self.q_values_2[state]).argmax()
        
        return self.env.action_space.sample()

    def update_q_value(self, 
                       state: int, 
                       action: int, 
                       reward: float, 
                       next_state: int, 
                       alpha: float, 
                       ) -> None:
        """Update Q-values using Double Q-learning update rule.
        
        Randomly selects which Q-value network to update and which to use for bootstrap.
        This reduces overestimation bias compared to single Q-learning.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            alpha: Learning rate.
        """
        
        bern = np.random.binomial(1, 0.5)
        if next_state in self.q_values_1:
            if bern == 0:
                action_next = self.q_values_1[next_state].argmax()
                q_next_state = self.q_values_2[next_state][action_next]

            else:
                action_next = self.q_values_2[next_state].argmax()
                q_next_state = self.q_values_1[next_state][action_next]
                
        else:
            self.q_values_1[next_state], self.q_values_2[next_state], self.visits[next_state] = np.zeros(self.n_actions), np.zeros(self.n_actions), np.zeros(self.n_actions)
            q_next_state = 0
            
        if state in self.q_values_1:
            if bern == 0:
                q_state = self.q_values_1[state][action]

            else:
                q_state = self.q_values_2[state][action]

        else:
            self.q_values_1[state], self.q_values_2[state], self.visits[state] = np.zeros(self.n_actions), np.zeros(self.n_actions), np.zeros(self.n_actions)
            q_state = 0

        td = reward + self.gamma*q_next_state - q_state
        self.visits[state][action] += 1
        if alpha is None:
            alpha = 1/(self.visits[state][action])

        if bern == 0:
            self.q_values_1[state][action] += alpha*td

        else:
            self.q_values_2[state][action] += alpha*td
        
    def unroll(self, alpha: float, epsilon: float):
        """Run a single Double Q-learning episode with Q-value updates.
        
        Args:
            alpha: Learning rate.
            epsilon: Exploration parameter.
        """

        state, _ = self.env.reset()
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated or truncated)
            self.update_q_value(state, action, reward, next_state, alpha)
            state = next_state


