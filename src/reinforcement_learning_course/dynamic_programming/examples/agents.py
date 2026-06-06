import numpy as np
import reinforcement_learning_course.dynamic_programming.algorithms as algs
import reinforcement_learning_course.dynamic_programming.examples.utils as utils
from gymnasium.core import Env


class ValueIterationFrozenLake(algs.ValueIteration):
    """Value Iteration agent for the frozen lake environment."""

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initializes ValueIterationFrozenLake.

        Args:
            env: The frozen lake Gymnasium environment.
            gamma: Discount factor for future rewards.
        """
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        """Returns the number of states and actions in the environment.

        Returns:
            A tuple of:
                - n_states: Number of discrete states.
                - n_actions: Number of discrete actions.
        """
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Builds the transition probability and reward matrices for the frozen lake environment.

        Returns:
            A tuple of:
                - transition_matrix: State transition probability matrix.
                - reward_matrix: Expected reward matrix.
        """
        return utils.frozen_lake_transition_matrices(self.env)


class QIterationFrozenLake(algs.QIteration):
    """Q-Iteration agent for the frozen lake environment."""

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initializes QIterationFrozenLake.

        Args:
            env: The frozen lake Gymnasium environment.
            gamma: Discount factor for future rewards.
        """
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        """Returns the number of states and actions in the environment.

        Returns:
            A tuple of:
                - n_states: Number of discrete states.
                - n_actions: Number of discrete actions.
        """
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Builds the transition probability and reward matrices for the frozen lake environment.

        Returns:
            A tuple of:
                - transition_matrix: State transition probability matrix.
                - reward_matrix: Expected reward matrix.
        """
        return utils.frozen_lake_transition_matrices(self.env)


class PolicyIterationFrozenLake(algs.PolicyIteration):
    """Policy Iteration agent for the frozen lake environment."""

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initializes PolicyIterationFrozenLake.

        Args:
            env: The frozen lake Gymnasium environment.
            gamma: Discount factor for future rewards.
        """
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        """Returns the number of states and actions in the environment.

        Returns:
            A tuple of:
                - n_states: Number of discrete states.
                - n_actions: Number of discrete actions.
        """
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Builds the transition probability and reward matrices for the frozen lake environment.

        Returns:
            A tuple of:
                - transition_matrix: State transition probability matrix.
                - reward_matrix: Expected reward matrix.
        """
        return utils.frozen_lake_transition_matrices(self.env)


class ValueIterationCliffWalking(algs.ValueIteration):
    """Value Iteration agent for the cliff walking environment."""

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initializes ValueIterationCliffWalking.

        Args:
            env: The cliff walking Gymnasium environment.
            gamma: Discount factor for future rewards.
        """
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        """Returns the number of states and actions in the environment.

        Returns:
            A tuple of:
                - n_states: Number of discrete states.
                - n_actions: Number of discrete actions.
        """
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Builds the transition probability and reward matrices for the cliff walking environment.

        Returns:
            A tuple of:
                - transition_matrix: State transition probability matrix.
                - reward_matrix: Expected reward matrix.
        """
        return utils.cliff_walking_transition_matrices(self.env)


class QIterationCliffWalking(algs.QIteration):
    """Q-Iteration agent for the cliff walking environment."""

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initializes QIterationCliffWalking.

        Args:
            env: The cliff walking Gymnasium environment.
            gamma: Discount factor for future rewards.
        """
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        """Returns the number of states and actions in the environment.

        Returns:
            A tuple of:
                - n_states: Number of discrete states.
                - n_actions: Number of discrete actions.
        """
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Builds the transition probability and reward matrices for the cliff walking environment.

        Returns:
            A tuple of:
                - transition_matrix: State transition probability matrix.
                - reward_matrix: Expected reward matrix.
        """
        return utils.cliff_walking_transition_matrices(self.env)


class PolicyIterationCliffWalking(algs.PolicyIteration):
    """Policy Iteration agent for the cliff walking environment."""

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initializes PolicyIterationCliffWalking.

        Args:
            env: The cliff walking Gymnasium environment.
            gamma: Discount factor for future rewards.
        """
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        """Returns the number of states and actions in the environment.

        Returns:
            A tuple of:
                - n_states: Number of discrete states.
                - n_actions: Number of discrete actions.
        """
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Builds the transition probability and reward matrices for the cliff walking environment.

        Returns:
            A tuple of:
                - transition_matrix: State transition probability matrix.
                - reward_matrix: Expected reward matrix.
        """
        return utils.cliff_walking_transition_matrices(self.env)


class ValueIterationTaxi(algs.ValueIteration):
    """Value Iteration agent for the Taxi environment."""

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initializes ValueIterationTaxi.

        Args:
            env: The Taxi Gymnasium environment.
            gamma: Discount factor for future rewards.
        """
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        """Returns the number of states and actions in the Taxi environment.

        The number of states is incremented by one to account for indexing.

        Returns:
            A tuple of:
                - n_states: Number of discrete states plus one.
                - n_actions: Number of discrete actions.
        """
        n_states, n_actions = self.env.observation_space.n + 1, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Builds the transition probability and reward matrices for the Taxi environment.

        Returns:
            A tuple of:
                - transition_matrix: State transition probability matrix.
                - reward_matrix: Expected reward matrix.
        """
        return utils.taxi_transition_matrices(self.env)


class QIterationTaxi(algs.QIteration):
    """Q-Iteration agent for the Taxi environment."""

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initializes QIterationTaxi.

        Args:
            env: The Taxi Gymnasium environment.
            gamma: Discount factor for future rewards.
        """
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        """Returns the number of states and actions in the Taxi environment.

        The number of states is incremented by one to account for indexing.

        Returns:
            A tuple of:
                - n_states: Number of discrete states plus one.
                - n_actions: Number of discrete actions.
        """
        n_states, n_actions = self.env.observation_space.n + 1, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Builds the transition probability and reward matrices for the Taxi environment.

        Returns:
            A tuple of:
                - transition_matrix: State transition probability matrix.
                - reward_matrix: Expected reward matrix.
        """
        return utils.taxi_transition_matrices(self.env)


class PolicyIterationTaxi(algs.PolicyIteration):
    """Policy Iteration agent for the Taxi environment."""

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """Initializes PolicyIterationTaxi.

        Args:
            env: The Taxi Gymnasium environment.
            gamma: Discount factor for future rewards.
        """
        super().__init__(env, gamma)

    def set_n_states_actions(self) -> None:
        """Returns the number of states and actions in the Taxi environment.

        The number of states is incremented by one to account for indexing.

        Returns:
            A tuple of:
                - n_states: Number of discrete states plus one.
                - n_actions: Number of discrete actions.
        """
        n_states, n_actions = self.env.observation_space.n + 1, self.env.action_space.n
        return n_states, n_actions

    def make_transition_matrices(self) -> tuple[np.array, np.array]:
        """Builds the transition probability and reward matrices for the Taxi environment.

        Returns:
            A tuple of:
                - transition_matrix: State transition probability matrix.
                - reward_matrix: Expected reward matrix.
        """
        return utils.taxi_transition_matrices(self.env)
