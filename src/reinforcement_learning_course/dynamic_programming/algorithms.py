import numpy as np
from reinforcement_learning_course.core import Agent
from gymnasium import Env


class ValueIteration(Agent[int, int]):
    """Value Iteration algorithm for finding optimal policies.
    
    Implements value iteration, a dynamic programming algorithm for solving
    Markov Decision Processes (MDPs) to find optimal policies.
    """

    def __init__(self, env: Env[int, int], gamma: float = 1.0) -> None:
        """Initialize the Value Iteration agent.
        
        Args:
            env: Gymnasium environment wrapper with integer state and action spaces.
            gamma: Discount factor in [0, 1], typically close to 1. Defaults to 1.0.
        
        Attributes:
            env: The gymnasium environment.
            n_states: Number of states in the environment.
            n_actions: Number of actions available.
            p_transition: Array of shape (n_states, n_actions, n_states) with transition probabilities.
            r_transition: Array of shape (n_states, n_actions, n_states) with transition rewards.
            policy: Array of shape (n_states,) storing the best action for each state.
            value: Array of shape (n_states,) storing the estimated value of each state.
        """

        super().__init__(env, gamma)
        self.n_states, self.n_actions = self.set_n_states_actions()
        self.p_transition, self.r_transition = self.make_transition_matrices()
        self.policy = np.zeros(self.n_states, dtype=int)
        self.value = np.zeros(self.n_states)

    def reset(self) -> None:
        """Reset the value function and policy to zero.
        
        Clears learned values and policy for training from scratch.
        """

        self.value = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=int)

    def set_n_states_actions(self) -> tuple[int, int]:
        """Define the number of states and actions in the environment.
        
        Must be implemented by subclasses to specify the problem dimensions.
        
        Returns:
            A tuple containing:
                - n_states: Number of states in the environment.
                - n_actions: Number of actions available.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError
    
    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Construct transition probability and reward matrices.
        
        Must be implemented by subclasses to define the environment's dynamics.
        
        Returns:
            A tuple containing:
                - p_transition: Array of shape (n_states, n_actions, n_states) with
                  transition probabilities P[s][a][s'] = Pr(s' | s, a).
                - r_transition: Array of shape (n_states, n_actions, n_states) with
                  transition rewards R[s][a][s'] = r(s, a, s').
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        
        raise NotImplementedError
    
    def action(self, state: int) -> int:
        """Select an action according to the learned policy.
        
        Args:
            state: The current state.
        
        Returns:
            The action prescribed by the policy for this state.
        """
        
        return self.policy[state]
    
    def train(self, n: int = 1000, epsilon: float = 1e-12) -> None:
        """Train the agent using Value Iteration.
        
        Iteratively updates value estimates using the Bellman optimality equation
        until convergence or maximum iterations reached.
        
        Args:
            n: Maximum number of iterations. Defaults to 1000.
            epsilon: Convergence threshold - stops if value norm change is below this.
              Defaults to 1e-12.
        """

        i=0
        delta = 1
        while i < n:
            value_next = ((self.p_transition*(self.r_transition + self.gamma*self.value.reshape((1, 1, -1)))).sum(axis=-1)).max(axis=-1)
            delta = np.linalg.norm(value_next - self.value)
            self.value = value_next
            i += 1
            if delta < epsilon:
                self.policy = ((self.p_transition*(self.r_transition + self.gamma*self.value.reshape((1, 1, -1)))).sum(axis=-1)).argmax(axis=-1)
                print('Termination condition achieved after %d iterations.' %i)
                return
            
        self.policy = ((self.p_transition*(self.r_transition + self.gamma*self.value.reshape((1, 1, -1)))).sum(axis=-1)).argmax(axis=-1)
        print('The termination condition has not been achieved after %d iterations.' %i)
    

class QIteration(Agent[int, int]):
    """Q-Iteration algorithm for finding optimal policies.
    
    Implements Q-iteration, a dynamic programming algorithm that directly learns
    state-action values to find optimal policies.
    """

    def __init__(self, env: Env[int, int], gamma: float = 1.0) -> None:
        """Initialize the Q-Iteration agent.
        
        Args:
            env: Gymnasium environment wrapper with integer state and action spaces.
            gamma: Discount factor in [0, 1], typically close to 1. Defaults to 1.0.
        
        Attributes:
            env: The gymnasium environment.
            n_states: Number of states in the environment.
            n_actions: Number of actions available.
            p_transition: Array of shape (n_states, n_actions, n_states) with transition probabilities.
            r_transition: Array of shape (n_states, n_actions, n_states) with transition rewards.
            policy: Array of shape (n_states,) storing the best action for each state.
            q_value: Array of shape (n_states, n_actions) storing state-action values.
        """

        super().__init__(env, gamma)
        self.n_states, self.n_actions = self.set_n_states_actions()
        self.p_transition, self.r_transition = self.make_transition_matrices()
        self.policy = np.zeros(self.n_states, dtype=int)
        self.q_value = np.zeros((self.n_states, self.n_actions))

    def reset(self) -> None:
        """Reset the Q-values and policy to zero.
        
        Clears learned state-action values and policy for training from scratch.
        """

        self.q_value = np.zeros((self.n_states, self.n_actions))
        self.policy = np.zeros(self.n_states, dtype=int)

    def set_n_states_actions(self) -> tuple[int, int]:
        """Define the number of states and actions in the environment.
        
        Must be implemented by subclasses to specify the problem dimensions.
        
        Returns:
            A tuple containing:
                - n_states: Number of states in the environment.
                - n_actions: Number of actions available.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Construct transition probability and reward matrices.
        
        Must be implemented by subclasses to define the environment's dynamics.
        
        Returns:
            A tuple containing:
                - p_transition: Array of shape (n_states, n_actions, n_states) with
                  transition probabilities P[s][a][s'] = Pr(s' | s, a).
                - r_transition: Array of shape (n_states, n_actions, n_states) with
                  transition rewards R[s][a][s'] = r(s, a, s').
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        
        raise NotImplementedError

    def action(self, state: int) -> int:
        """Select an action according to the learned policy.
        
        Args:
            state: The current state.
        
        Returns:
            The action prescribed by the policy for this state.
        """
        
        return self.policy[state]

    def train(self, n: int = 1000, epsilon: float = 1e-12) -> None:
        """Train the agent using Q-Iteration.
        
        Iteratively updates Q-values using the Bellman equation until convergence
        or maximum iterations reached. Policy is extracted from learned Q-values.
        
        Args:
            n: Maximum number of iterations. Defaults to 1000.
            epsilon: Convergence threshold - stops if Q-value norm change is below this.
              Defaults to 1e-12.
        """
        
        i=0
        delta = 1
        while i < n:
            p_transition_reshape = np.expand_dims(self.p_transition, axis=3)
            r_transition_reshape = np.expand_dims(self.r_transition, axis=3)
            q_value_reshape = np.expand_dims(self.q_value, axis=(0, 1))
            q_value_next = ((p_transition_reshape*(r_transition_reshape + self.gamma*q_value_reshape)).sum(axis=-2)).max(axis=-1)
            delta = np.linalg.norm(q_value_next - self.q_value)
            self.q_value = q_value_next
            i += 1
            if delta < epsilon:
                self.policy = self.q_value.argmax(axis=-1)
                print('Termination condition achieved after %d iterations.' %i)
                return
            
        self.policy = self.q_value.argmax(axis=-1)
        print('The termination condition has not been achieved after %d iterations.' %i)
        

class PolicyIteration(Agent[int, int]):
    """Policy Iteration algorithm for finding optimal policies.
    
    Implements policy iteration, a dynamic programming algorithm that alternates
    between policy evaluation and policy improvement until convergence.
    """

    def __init__(self, env: Env[int, int], gamma: float = 1.0) -> None:
        """Initialize the Policy Iteration agent.
        
        Args:
            env: Gymnasium environment wrapper with integer state and action spaces.
            gamma: Discount factor in [0, 1], typically close to 1. Defaults to 1.0.
        
        Attributes:
            env: The gymnasium environment.
            n_states: Number of states in the environment.
            n_actions: Number of actions available.
            p_transition: Array of shape (n_states, n_actions, n_states) with transition probabilities.
            r_transition: Array of shape (n_states, n_actions, n_states) with transition rewards.
            policy: Array of shape (n_states,) storing the current policy.
            value: Array of shape (n_states,) storing the estimated value of each state.
        """

        super().__init__(env, gamma)
        self.n_states, self.n_actions = self.set_n_states_actions()
        self.p_transition, self.r_transition = self.make_transition_matrices()
        self.policy = np.zeros(self.n_states, dtype=int)
        self.value = np.zeros(self.n_states)

    def reset(self) -> None:
        """Reset the value function and policy to zero.
        
        Clears learned values and policy for training from scratch.
        """

        self.value = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=int)

    def set_n_states_actions(self) -> tuple[int, int]:
        """Define the number of states and actions in the environment.
        
        Must be implemented by subclasses to specify the problem dimensions.
        
        Returns:
            A tuple containing:
                - n_states: Number of states in the environment.
                - n_actions: Number of actions available.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Construct transition probability and reward matrices.
        
        Must be implemented by subclasses to define the environment's dynamics.
        
        Returns:
            A tuple containing:
                - p_transition: Array of shape (n_states, n_actions, n_states) with
                  transition probabilities P[s][a][s'] = Pr(s' | s, a).
                - r_transition: Array of shape (n_states, n_actions, n_states) with
                  transition rewards R[s][a][s'] = r(s, a, s').
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        
        raise NotImplementedError

    def action(self, state: int) -> int:
        """Select an action according to the learned policy.
        
        Args:
            state: The current state.
        
        Returns:
            The action prescribed by the policy for this state.
        """
        
        return self.policy[state]

    def train(self, n: int = 1000, epsilon: float = 1e-12) -> None:
        """Train the agent using Policy Iteration.
        
        Alternates between policy evaluation (computing values under current policy)
        and policy improvement (updating policy to be greedy w.r.t. computed values)
        until the policy stabilizes or maximum iterations reached.
        
        Args:
            n: Maximum number of iterations. Defaults to 1000.
            epsilon: Convergence threshold - stops if value norm change is below this.
              Defaults to 1e-12.
        """
        
        i=0
        delta = 1
        while i < n:
            p_transition_policy = self.p_transition[np.arange(self.n_states), self.policy, :]
            r_transition_policy = self.r_transition[np.arange(self.n_states), self.policy, :]
            p_r_policy = (p_transition_policy*r_transition_policy).sum(axis=-1)
            value_policy = np.linalg.solve(np.eye(self.n_states) - self.gamma*p_transition_policy, p_r_policy)
            policy_next = (((self.p_transition*(self.r_transition + self.gamma*value_policy.reshape((1, 1, -1)))).sum(axis=-1)).argmax(axis=-1)).astype(int)
            delta = np.linalg.norm(value_policy - self.value)
            delta_policy = (policy_next != self.policy).sum()
            self.policy = policy_next
            self.value = value_policy
            i += 1
            if delta_policy == 0 or delta < epsilon:
                print('Termination condition achieved after %d iterations.' %i)
                return
            
        print('The termination condition has not been achieved after %d iterations.' %i)
