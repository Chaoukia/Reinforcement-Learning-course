import numpy as np
from typing import List
from reinforcement_learning_course.core import Agent
from gymnasium import Env


class MonteCarlo(Agent[int, int]):
    """
    Monte-Carlo agent.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99) -> None:
        """
        Description
        --------------------------------------------
        Constructor.

        Parameters & Attributes
        --------------------------------------------
        env       : gymnasium environment Wrapper.
        gamma     : Float in [0, 1] generally close to 1, discount factor.
        n_states  : Int, number of states.
        n_actions : Int, number of n_actions.
        q_values  : Dict mapping states to their action values.
        self.visits    : Dict mapping states to their number of self.visits.

        Returns
        --------------------------------------------
        """

        super().__init__(env, gamma)
        self.n_states, self.n_actions = self.set_n_states_actions()
        self.q_values = {}
        self.visits = {}

    def reset(self) -> None:
        """
        Description
        --------------------------------------------
        Reinitialize the state-action values function.

        Parameters
        --------------------------------------------

        Returns
        --------------------------------------------
        """

        self.q_values = {}
        self.visits = {}

    def set_n_states_actions(self) -> None:
        """
        Description
        --------------------------------------------
        Set the number of states and actions.

        Parameters
        --------------------------------------------

        Returns
        --------------------------------------------
        """

        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        return n_states, n_actions
    
    def action_explore(self, state: int, epsilon: float) -> int:
        """
        Description
        --------------------------------------------
        Take an action according to an epsilon-greedy policy.
        
        Parameters
        --------------------------------------------
        state   : np.array, state.
        epsilon : Float in ]0, 1[, probability of taking a suboptimal action.
        
        Returns
        --------------------------------------------
        Int, action to perform.
        """

        action_max = self.action(state)
        bern = np.random.binomial(1, 1 - epsilon)
        if bern == 1:
            return action_max
        
        return np.random.choice(self.n_actions)
        
    def action(self, state: int) -> int:
        """
        Description
        --------------------------------------------
        Take an action according to the estimated optimal policy.
        
        Parameters
        --------------------------------------------
        state : Int, state.
        
        Returns
        --------------------------------------------
        Int, estimated optimal action.
        """

        if state in self.q_values:
            return self.q_values[state].argmax()
        
        return self.env.action_space.sample()
    
    def unroll(self, epsilon: float) -> tuple[List, List, List]:
        """
        Description
        --------------------------------------------
        Unroll the current epsilon-greedy policy from state.
        
        Parameters
        --------------------------------------------
        state : Int, an initial state.
        
        Returns
        --------------------------------------------
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
              n_train: int = 1000, first_visit: bool = True, print_iter: int = 10):
        """
        Description
        --------------------------------------------
        Train an on-policy first-visit MC algorithm.
        
        Parameters
        --------------------------------------------
        epsilon_start : Float in ]0, 1], initial value of epsilon.
        epsilon_stop  : Float in ]0, 1], final value of epsilon.
        decay_rate    : Float, decay rate of epsilon.
        n_train       : Int, number of training episodes.
        first_visit   : Boolean, whether to apply first visit MC or every visit MC.
        print_iter    : Int, number of episodes episodes between two consecutive prints.
        
        Returns
        --------------------------------------------
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
                
