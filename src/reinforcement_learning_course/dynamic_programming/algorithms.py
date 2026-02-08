import numpy as np
from reinforcement_learning_course.core import Agent
from gymnasium import Env


class ValueIteration(Agent[int, int]):
    """
    Value Iteration algorithm.
    """

    def __init__(self, env: Env[int, int], gamma: float = 1.0) -> None:
        """
        Description
        --------------------------------------------
        Constructor.

        Parameters & Attributes
        --------------------------------------------
        env                      : gymnasium environment Wrapper
        make_transition_matrices : function generating the transirion matrices.
        gamma                    : Float in [0, 1] generally close to 1, discount factor.
        p_transition             : np.array of shape (n_state, n_actions, n_states), transition probabilities matrix.
        r_transition             : np.array of shape (n_state, n_actions, n_states), transition rewards matrix.
        n_states                 : Int, number of states.
        n_actions                : Int, number of n_actions.
        policy                   : np.array of shape (n_states,), the policy function.
        value                    : np.array of shape (n_states,), the state value function.

        Returns
        --------------------------------------------
        """

        super().__init__(env, gamma)
        self.n_states, self.n_actions = self.set_n_states_actions()
        self.p_transition, self.r_transition = self.make_transition_matrices()
        self.policy = np.zeros(self.n_states, dtype=int)
        self.value = np.zeros(self.n_states)

    def reset(self) -> None:
        """
        Description
        --------------------------------------------
        Reset the agent's value and policy.

        Parameters
        --------------------------------------------

        Returns
        --------------------------------------------
        """

        self.value = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=int)

    def set_n_states_actions(self) -> tuple[int, int]:
        """
        Description
        --------------------------------------------
        Define the number of states and actions.

        Parameters
        --------------------------------------------

        Returns
        --------------------------------------------
        """

        raise NotImplementedError
    
    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """
        Description
        --------------------------------------------
        Construct p_transition and r_transition the probability and reward transition matrices (respectively).
        
        Parameters
        --------------------------------------------
        
        Returns
        --------------------------------------------
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        n_states     : Int, number of states.
        n_actions    : Int, number of actions.
        """
        
        raise NotImplementedError
    
    def action(self, state: int) -> int:
        """
        Description
        --------------
        Choose an action according to the agent's policy.
        
        Parameters
        --------------
        state : int, state where to choose the policy's action.
        
        Returns
        --------------
        int, action chosen by the agent's policy.
        """
        
        return self.policy[state]
    
    def train(self, n: int = 1000, epsilon: float = 1e-12) -> None:
        """
        Description
        --------------
        Train with Value Iteration.

        Parameters
        --------------
        n       : Int, maximum number of iterations.
        epsilon : Float, small threshold 0 < epsilon << 1. Stop training if the norm difference between two consecutive values drops below epsilon.

        Returns
        --------------
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
    """
    Q-Iteration algorithm.
    """

    def __init__(self, env: Env[int, int], gamma: float = 1.0) -> None:
        """
        Description
        --------------------------------------------
        Constructor.

        Parameters & Attributes
        --------------------------------------------
        env          : gymnasium environment Wrapper
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        p_transition : np.array of shape (n_state, n_actions, n_states), transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), transition rewards matrix.
        n_states     : Int, number of states.
        n_actions    : Int, number of actions.
        policy       : np.array of shape (n_states,), the policy function.
        q_value      : np.array of shape (n_states, n_actions), the state-action value function.

        Returns
        --------------------------------------------
        """

        super().__init__(env, gamma)
        self.n_states, self.n_actions = self.set_n_states_actions()
        self.p_transition, self.r_transition = self.make_transition_matrices()
        self.policy = np.zeros(self.n_states, dtype=int)
        self.q_value = np.zeros((self.n_states, self.n_actions))

    def reset(self) -> None:
        """
        Description
        --------------------------------------------
        Reset the agent's state-action value and policy.

        Parameters
        ----------------------

        Returns
        ----------------------
        """

        self.q_value = np.zeros((self.n_states, self.n_actions))
        self.policy = np.zeros(self.n_states, dtype=int)

    def set_n_states_actions(self) -> tuple[int, int]:
        """
        Description
        --------------------------------------------
        Define the number of states and actions.

        Parameters
        --------------------------------------------

        Returns
        --------------------------------------------
        """

        raise NotImplementedError

    def make_transition_matrices(self) -> tuple[np.array, np.array, int, int]:
        """
        Description
        --------------
        Construct p_transition and r_transition the probability and reward transition matrices (respectively).
        
        Parameters
        --------------
        
        Returns
        --------------
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        """
        
        raise NotImplementedError

    def action(self, state: int) -> int:
        """
        Description
        --------------
        Choose an action according to the agent's policy.
        
        Parameters
        --------------
        state : int, state where to choose the policy's action.
        
        Returns
        --------------
        int, action chosen by the agent's policy.
        """
        
        return self.policy[state]

    def train(self, n: int = 1000, epsilon: float = 1e-12) -> None:
        """
        Description
        --------------
        Train with Q-iteration.
        
        Arguments
        --------------
        epsilon : Float, small threshold 0 < epsilon << 1, stop the algorithm if the norm difference between two consecutive values drops below epsilon.
        n       : Int, number of iterations.
        
        Returns
        --------------
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
    """
    Policy Iteration algorithm.
    """

    def __init__(self, env: Env[int, int], gamma: float = 1.0) -> None:
        """
        Description
        --------------------------------------------
        Constructor.

        Parameters & Attributes
        --------------------------------------------
        env          : gymnasium environment Wrapper
        gamma        : Float in [0, 1] generally close to 1, discount factor.
        p_transition : np.array of shape (n_state, n_actions, n_states), transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), transition rewards matrix.
        n_states     : Int, number of states.
        n_actions    : Int, number of actions.
        policy       : np.array of shape (n_states,), the policy function.
        value        : np.array of shape (n_states,), the state value function.

        Returns
        --------------------------------------------
        """

        super().__init__(env, gamma)
        self.n_states, self.n_actions = self.set_n_states_actions()
        self.p_transition, self.r_transition = self.make_transition_matrices()
        self.policy = np.zeros(self.n_states, dtype=int)
        self.value = np.zeros(self.n_states)

    def reset(self) -> None:
        """
        Description
        --------------------------------------------
        Reset the agent's value and policy.

        Parameters
        ----------------------

        Returns
        ----------------------
        """

        self.value = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=int)

    def set_n_states_actions(self) -> tuple[int, int]:
        """
        Description
        --------------------------------------------
        Define the number of states and actions.

        Parameters
        --------------------------------------------

        Returns
        --------------------------------------------
        """

        raise NotImplementedError

    def make_transition_matrices(self) -> tuple[np.array, np.array, int, int]:
        """
        Description
        --------------
        Construct p_transition and r_transition the probability and reward transition matrices (respectively).
        
        Parameters
        --------------
        
        Returns
        --------------
        p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
        r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
        """
        
        raise NotImplementedError

    def action(self, state: int) -> int:
        """
        Description
        --------------
        Choose an action according to the agent's policy.
        
        Parameters
        --------------
        state : int, state where to choose the policy's action.
        
        Returns
        --------------
        int, action chosen by the agent's policy.
        """
        
        return self.policy[state]

    def train(self, n: int = 1000, epsilon: float = 1e-12) -> None:
        """
        Description
        --------------
        Train with Policy Iteration.
        
        Arguments
        --------------
        n       : Int, maximum number of iterations.
        epsilon : Float, small threshold 0 < epsilon << 1. Stop training if the norm difference between two consecutive values drops below epsilon.
        
        Returns
        --------------
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
