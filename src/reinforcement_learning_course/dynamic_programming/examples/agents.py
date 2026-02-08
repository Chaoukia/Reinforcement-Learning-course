import numpy as np
import reinforcement_learning_course.dynamic_programming.algorithms as algs
import reinforcement_learning_course.dynamic_programming.examples.utils as utils
from gymnasium.core import Env


class ValueIterationFrozenLake(algs.ValueIteration):
    """
    Value Iteration agent for the frozen lake environment.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        return utils.frozen_lake_transition_matrices(self.env)
    
    
class QIterationFrozenLake(algs.QIteration):
    """
    Q-Iteration agent for the frozen lake environment.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        return utils.frozen_lake_transition_matrices(self.env)
    
    
class PolicyIterationFrozenLake(algs.PolicyIteration):
    """
    Policy Iteration agent for the frozen lake environment.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        return utils.frozen_lake_transition_matrices(self.env)
    

class ValueIterationCliffWalking(algs.ValueIteration):
    """
    Value Iteration agent for the cliff walking environment.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        return utils.cliff_walking_transition_matrices(self.env)

    
class QIterationCliffWalking(algs.QIteration):
    """
    Q-Iteration agent for the cliff walking environment.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        return utils.cliff_walking_transition_matrices(self.env)

    
class PolicyIterationCliffWalking(algs.PolicyIteration):
    """
    Policy Iteration agent for the cliff walking environment.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        return utils.cliff_walking_transition_matrices(self.env)


class ValueIterationTaxi(algs.ValueIteration):
    """
    Value Iteration agent for the Taxi environment.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        n_states, n_actions = self.env.observation_space.n + 1, self.env.action_space.n
        return n_states, n_actions
        
    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        return utils.taxi_transition_matrices(self.env)
    
    
class QIterationTaxi(algs.QIteration):
    """
    Q-Iteration agent for the Taxi environment.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        n_states, n_actions = self.env.observation_space.n + 1, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        return utils.taxi_transition_matrices(self.env)
    
    
class PolicyIterationTaxi(algs.PolicyIteration):
    """
    Policy Iteration agent for the Taxi environment.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        super().__init__(env, gamma)
        
    def set_n_states_actions(self) -> None:
        n_states, n_actions = self.env.observation_space.n + 1, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        return utils.taxi_transition_matrices(self.env)
