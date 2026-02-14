import numpy as np
import heapq
from reinforcement_learning_course.core import Agent
from gymnasium import Env
from typing import List


class Astar(Agent[int, int]):
    """A* search algorithm for finding optimal policies.
    
    Implements the A* algorithm to solve planning problems by finding the optimal
    path from an initial state to a goal state.
    """

    def __init__(self, env: Env[int, int], gamma: float = 1.0) -> None:
        """Initialize the A* agent.
        
        Args:
            env: Gymnasium environment wrapper.
            gamma: Discount factor in [0, 1], typically close to 1. Defaults to 1.0.
        
        Attributes:
            env: The gymnasium environment.
            n_states: Number of states in the environment.
            n_actions: Number of actions available in the environment.
            policy: Array of shape (n_states,) storing the best action for each state.
            goals: Set of goal states to reach.
        """

        super().__init__(env, gamma)
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.policy = np.zeros(self.n_states, dtype=int)
        self.goals = None

    def reset(self) -> None:
        """Reinitialize the policy with zeros.
        
        Resets the policy array so all states map to action 0.
        """

        self.policy = np.zeros(self.n_states, dtype=int)

    def set_n_states_actions(self) -> tuple[int, int]:
        """Define the number of states and actions.
        
        Must be implemented by subclasses to specify the problem dimensions.
        
        Returns:
            A tuple containing:
                - n_states: Number of states in the environment.
                - n_actions: Number of actions available.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError

    def heuristic(self, state: int) -> float:
        """Compute an upper bound on the optimal value from a state.
        
        Must be implemented by subclasses. Returns a heuristic estimate that guides
        the A* search by overestimating the value of reaching a goal from the state.
        
        Args:
            state: A state in the environment.
        
        Returns:
            Float, the heuristic value of the state (upper bound on optimal return).
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError

    def split(self, state: int, action: int) -> tuple[float, int]:
        """Get the reward and next state from taking an action in a state.
        
        Must be implemented by subclasses to define the environment's transition dynamics.
        
        Args:
            state: The current state.
            action: The action to take.
        
        Returns:
            A tuple containing:
                - reward: The immediate reward from taking the action.
                - next_state: The resulting state after the action.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError

    def expand(self, info_state: List) -> List:
        """Expand a state to all its successor states via available actions.
        
        Generates all children states reachable by taking each possible action
        from the given state.
        
        Args:
            info_state: A list with four elements:
                - value: Float, heuristic value of the state (cumulative reward + heuristic).
                - reward_neg: Float, negated cumulative reward from the root.
                - actions: List of actions leading to this state from the root.
                - state: Int, the state to expand.
        
        Returns:
            A list of child state information lists, each containing:
                - value_neg: Float, negated heuristic value of the child state.
                - reward_child_neg: Float, negated cumulative reward to the child.
                - actions_child: List of actions leading to the child from the root.
                - child: Int, the child state.
        """

        _, reward_neg, actions, state = info_state
        reward = -reward_neg
        info_children = []
        for action in range(self.n_actions):
            r, child = self.split(state, action)
            reward_child = reward + r
            value_child = reward_child + self.heuristic(child)
            info_child = [-value_child, -reward_child, actions + [action], child]
            info_children.append(info_child)

        return info_children
    
    def train(self) -> None:
        """Run the A* algorithm to find the optimal policy.
        
        Executes the A* search algorithm to find the optimal sequence of actions
        from the initial state to a goal state, maximizing the cumulative reward.
        Updates the policy with the discovered optimal actions. Prints results and
        goal state verification data.
        """

        memo = {}
        queue = []
        root, _ = self.env.reset()
        info_root = [0, 0, [], root]
        memo[root] = info_root
        heapq.heappush(queue, info_root)
        iters = 0
        while queue:
            iters += 1
            info_state = heapq.heappop(queue)
            _, _, actions, state = info_state
            # The search is over when we select a goal state.
            if state in self.goals:
                print('An optimal policy has been found after %d iterations.' %iters)
                self.infer(root, actions)
                return
            
            info_children = self.expand(info_state)
            to_heapify = False
            for value_child_neg, reward_child_neg, actions_child, child in info_children:
                value_child, reward_child = -value_child_neg, -reward_child_neg
                try:
                    info_child = memo[child]
                    value_child_prev = -info_child[0]
                    if value_child > value_child_prev:
                        info_child[0] = -value_child
                        info_child[1] = -reward_child
                        info_child[2] = actions_child
                        to_heapify = True

                except KeyError:
                    info_child = [-value_child, -reward_child, actions_child, child]
                    memo[child] = info_child
                    heapq.heappush(queue, info_child)

            if to_heapify:
                heapq.heapify(queue)

        print('The problem is unsolvable.')

    def infer(self, root: int, actions: List) -> None:
        """Compile the policy from a root state and a sequence of actions.
        
        Given a sequence of actions from the initial state to a goal, sets the
        policy to follow these actions from each encountered state.
        
        Args:
            root: The initial state.
            actions: List of actions to take from the root to reach a goal.
        """

        state = root
        i = 0
        while state not in self.goals:
            action = actions[i]
            self.policy[state] = action
            _, next_state = self.split(state, action)
            state = next_state
            i += 1

    def action(self, state: int) -> int:
        """Get the best action for a state according to the learned policy.
        
        Args:
            state: The current state.
        
        Returns:
            Int, the action prescribed by the policy for this state.
        """
        
        return self.policy[state]
    
