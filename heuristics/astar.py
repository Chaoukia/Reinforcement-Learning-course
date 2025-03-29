import numpy as np
import heapq
from PIL import Image

class Astar:
    """
    Description
    --------------
    Class describing the A* algorithm.
    """

    def __init__(self, env):
        """
        Description
        --------------
        Constructor of class FrozenLakeAgent.
        
        Arguments
        --------------
        env          : gymnasium environment.
        n_states     : Int, the number of states.
        n_actions    : Int, the number of actions.
        goals        : Set of goal states.
        policy       : np.array of shape (n_states,) or None, policy.
        """

        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.goals = None
        self.reset()

    def reset(self):
        """
        Description
        --------------
        Reinitialize the policy.

        Arguments
        --------------

        Returns
        --------------
        """

        self.policy = np.zeros(self.n_states).astype(int)

    def heuristic(self, state):
        """
        Description
        --------------
        Return an upper bound on the optimal value at state.

        Arguments
        --------------
        state : Int, a state

        Returns
        --------------
        Float, the heuristic value of state.
        """

        raise NotImplementedError

    def split(self, state, action):
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

        raise NotImplementedError

    def expand(self, info_state):
        """
        Description
        --------------
        Expand a state by returning the states that stem from it by taking each possible action.

        Arguments
        --------------
        info_state : List of three elements.
                        - value   : Float, negative heuristic value of state. Equal to the sum of rewards following actions from the root plus an upper bound on the return of the optimal policy from state.
                        - reward  : Float, negative sum of rewards following actions from the root.
                        - actions : List of actions that lead to state from the initial state (root).
                        - state   : Int, a state.

        Returns
        --------------
        List of lists of the form [value, actions, next_state] where:
            - value      : Float, negative heuristic value of next_state. Equal to the sum of rewards following actions from the root plus an upper bound on the return of the optimal policy from next_state.
            - reward     : Float, negative sum of rewards following actions from the root.
            - actions    : List of actions that lead to next_state from the initial state (root).
            - next_state : Int, a next (child) state that stems from taking an action in state.
        """

        raise NotImplementedError
    
    def train(self, root):
        """
        Description
        --------------
        Run the A* algorithm to find the optimal policy inducing the path of maximum reward from an initial state (root) to a goal state.

        Arguments
        --------------
        root : Int, Initial state.

        Returns
        --------------
        List of optimal actions to follow starting from the root, when the problem is solvable.
        None when the problem is unsolvable.
        """

        memo = {}
        queue = []
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

    def infer(self, root, actions):
        """
        Description
        --------------
        Infer the optimal policy from a root and a list of corresponding actions.

        Arguments
        --------------
        root    : Int, an initial state.
        actions : List of actions.

        Returns
        --------------
        """

        state = root
        i = 0
        while state not in self.goals:
            action = actions[i]
            self.policy[state] = action
            _, next_state = self.split(state, action)
            state = next_state
            i += 1

    def action(self, state):
        """
        Description
        --------------
        Take an action according to the estimated optimal policy.
        
        Arguments
        --------------
        state : np.array, state.
        
        Returns
        --------------
        Int, action.
        """
        
        return self.policy[state]
    
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
            done = False
            R = 0
            n_steps = 0
            while not done:
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
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
            
    def save_gif(self, env, file_name):
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
        state, _ = env.reset()
        done = False
        R = 0
        n_steps = 0
        while not done:
            frames.append(Image.fromarray(env.render(), mode='RGB'))
            action = self.action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = (terminated or truncated)
            state = next_state
            R += reward
            n_steps += 1
        
        frames.append(Image.fromarray(env.render(), mode='RGB'))
        frames[0].save(file_name, save_all=True, append_images=frames[1:], optimize=False, duration=150, loop=0)