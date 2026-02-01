import numpy as np
import reinforcement_learning_course.heuristics.algorithms as algs
from gymnasium.core import Env


class AstarCliffWalking(algs.Astar):
    """
    Astar agent for the cliff walking environment.
    """

    def __init__(self, env: Env[int, int], gamma: float = 1.0):
        """
        Description
        ------------------------------
        Constructor.

        Parameters & Attributes
        ------------------------------
        env   : frozen lake gymnasium environment.
        gamma : Float, discount factor.
        shape : tuple, grid shape.
        goals : Set of goal states.
        """

        super().__init__(env, gamma)
        self.shape = (4, 12)
        self.goals = set([47])

    def heuristic(self, state: int) -> float:
        """
        Description
        --------------
        Return an upper bound on the optimal value at state.

        Arguments
        --------------
        state : Int, a state.

        Returns
        --------------
        Float, the heuristic value of state.
        """

        return 0.
    
    def split(self, state: int, action: int) -> tuple[float, int]:
        """
        Description
        --------------
        Return the reward and next state induced by taking an action in a state.

        Arguments
        --------------
        state  : Int, a state.
        action : Int, an action.

        Returns
        --------------
        Float, the induced reward.
        Int, the corresponding next state.
        """

        state_index = np.unravel_index(state, self.shape)
        if action == 0: # Go Up.
            next_state_index = (max(0, state_index[0] - 1), state_index[1])

        elif action == 1: # Go right.
            next_state_index = (state_index[0], min(11, state_index[1] + 1))

        elif action == 2: # Go down.
            next_state_index = (min(3, state_index[0] + 1), state_index[1])

        elif action == 3: # Go left.
            next_state_index = (state_index[0], max(0, state_index[1] - 1))
            
        # Going over the cliff sends us back immediately to the initial state and incurs a reward of -100.
        if next_state_index[0] == 3 and next_state_index[1] in set(range(1, 11)):
            reward, next_state = -100, 36

        # In all other situations we incur a reward of -1.
        else:
            reward, next_state = -1, next_state_index[0]*self.shape[1] + next_state_index[1]

        return reward, next_state
    

class AstarFrozenLake(algs.Astar):
    """
    Astar agent for the frozen lake environment.
    """

    def __init__(self, env: Env[int, int], gamma: float = 1.):
        """
        Description
        ------------------------------
        Constructor.

        Parameters & Attributes
        ------------------------------
        env   : frozen lake gymnasium environment.
        gamma : Float, discount factor.
        map   : np.array, the grid map of the environment.
        shape : tuple, grid shape.
        goals : Set of goal states.
        holes : Set of holes, these are bad absorbing states.

        Returns
        ------------------------------
        """

        super().__init__(env, gamma)
        self.map = env.unwrapped.desc.astype(str)
        self.shape = self.map.shape
        self.goals = set(np.arange(self.map.shape[0]*self.map.shape[1]).reshape((self.map.shape[0], self.map.shape[1]))[self.map == 'G'])
        self.holes = set(np.arange(self.map.shape[0]*self.map.shape[1]).reshape((self.map.shape[0], self.map.shape[1]))[self.map == 'H'])

    def heuristic(self, state: int) -> float:
        """
        Description
        --------------
        Return an upper bound on the optimal value at state.

        Arguments
        --------------
        state : Int, a state.

        Returns
        --------------
        Float, the heuristic value of state.
        """

        if state in self.holes: return 0
        return 1
        

    def split(self, state: int, action: int) -> tuple[int, float]:
        """
        Description
        ------------------------------
        Return the reward and next state induced by taking an action in a state.

        Arguments
        ------------------------------
        state  : Int, a state.
        action : Int, an action.

        Returns
        ------------------------------
        Float, the induced reward.
        Int, the corresponding next state.
        """

        # A hole is an absorbing state.
        if state in self.holes:
            return 0, state

        state_index = np.unravel_index(state, self.shape)
        if action == 0: # Go left.
            next_state_index = np.array(state_index)
            next_state_index[1] = max(0, next_state_index[1] - 1)
            next_state = next_state_index[0]*self.shape[0] + next_state_index[1]

        elif action == 2: # Go right.
            next_state_index = np.array(state_index)
            next_state_index[1] = min(self.shape[1] - 1, next_state_index[1] + 1)
            next_state = next_state_index[0]*self.shape[0] + next_state_index[1]

        elif action == 1: # Go down.
            next_state_index = np.array(state_index)
            next_state_index[0] = min(self.shape[0] - 1, next_state_index[0] + 1)
            next_state = next_state_index[0]*self.shape[0] + next_state_index[1]

        elif action == 3: # Go up.
            next_state_index = np.array(state_index)
            next_state_index[0] = max(0, next_state_index[0] - 1)
            next_state = next_state_index[0]*self.shape[0] + next_state_index[1]

        reward = 1 if next_state in self.goals else 0
        return reward, next_state

