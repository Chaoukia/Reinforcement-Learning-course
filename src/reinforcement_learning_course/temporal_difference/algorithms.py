import numpy as np
from reinforcement_learning_course.core import Agent
from gymnasium import Env


class TemporalDifference(Agent[int, int]):
    """
    Temporal Difference agent.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99) -> None:
        super().__init__(env, gamma)
        self.n_actions = self.env.action_space.n
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

    def action_explore(self, state: int, epsilon: float) -> int:
        """
        Description
        ----------------------------
        Take an action according to an epsilon-greedy policy.
        
        Arguments
        ----------------------------
        state   : np.array, state.
        epsilon : Float in ]0, 1[, probability of taking a suboptimal action.
        
        Returns
        ----------------------------
        Int, action to perform.
        """

        action_max = self.action(state)
        bern = np.random.binomial(1, 1 - epsilon)
        if bern == 1:
            return action_max
        
        return self.env.action_space.sample()
        
    def action(self, state: int) -> int:
        """
        Description
        ----------------------------
        Take an action according to the estimated optimal policy.
        
        Arguments
        ----------------------------
        state : Int, state.
        
        Returns
        ----------------------------
        Int, estimated optimal action.
        """

        if state in self.q_values:
            return self.q_values[state].argmax()
        
        return self.env.action_space.sample()
    
    def unroll(self, alpha: float, epsilon: float) -> None:
        """
        Description
        ----------------------------
        Unroll an episode with the current exploration policy and updated the estimated q-values with temporal difference.

        Parameters
        ----------------------------
        alpha   : Float in ]0, 1[, learning rate.
        epsilon : Float in ]0, 1[, exploration parameter.

        Returns
        ----------------------------
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
        """
        Description
        --------------
        Train an on-policy first-visit MC algorithm.
        
        Arguments
        --------------
        alpha         : Float in ]0, 1[, learning rate.
        epsilon_start : Float in ]0, 1[, initial value of epsilon.
        epsilon_stopt : Float in ]0, 1[, final value of epsilon.
        decay_rate    : Float, decay rate of epsilon from epsilon_start to epsilon_stop.
        n_train       : Int, total number of iterations.
        print_iter    : Int, number of iterations between two successive prints.
        
        Returns
        --------------
        """

        for i in range(n_train):
            epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*i)
            self.unroll(alpha, epsilon)
            if i%print_iter == 0:
                print('Iteration : %d' %i)
                print('Epsilon   : %.5f' %epsilon)
                print('\n')


class SARSA(TemporalDifference):
    """
    SARSA agent.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """
        Description
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
        """
        Description
        ----------------------------
        Perform a sarsa update of experience (state, action, reward, next_state).
        
        Parameters
        ----------------------------
        state      : Int, current state.
        action     : Int, performed action.
        reward     : Float, perceived reward.
        next_state : Int, next state.
        alpha      : Float, learning rate.
        epsilon    : Float, exploration parameter.
        
        Returns
        ----------------------------
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
        """
        Description
        --------------
        Unroll the current epsilon-greedy policy from state and update the q-values at each step.
        
        Arguments
        --------------
        alpha   : Float in ]0, 1[, learning rate.
        epsilon : Float in ]0, 1[, parameter of the epsilon-greedy policy.
        
        Returns
        --------------
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
    """
    SARSA agent.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """
        Description
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
        """
        Description
        ----------------------------
        Perform a sarsa update of experience (state, action, reward, next_state).
        
        Parameters
        ----------------------------
        state      : Int, current state.
        action     : Int, performed action.
        reward     : Float, perceived reward.
        next_state : Int, next state.
        alpha      : Float, learning rate.
        epsilon    : Float, exploration parameter.
        
        Returns
        ----------------------------
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
        """
        Description
        --------------
        Unroll the current epsilon-greedy policy from state and update the q-values at each step.
        
        Arguments
        --------------
        alpha   : Float in ]0, 1[, learning rate.
        epsilon : Float in ]0, 1[, parameter of the epsilon-greedy policy.
        
        Returns
        --------------
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
    """
    SARSA agent.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """
        Description
        """
        super().__init__(env, gamma)

    def update_q_value(self, 
                       state: int, 
                       action: int, 
                       reward: float, 
                       next_state: int, 
                       alpha: float, 
                       ) -> None:
        """
        Description
        ----------------------------
        Perform a sarsa update of experience (state, action, reward, next_state).
        
        Parameters
        ----------------------------
        state      : Int, current state.
        action     : Int, performed action.
        reward     : Float, perceived reward.
        next_state : Int, next state.
        alpha      : Float, learning rate.
        
        Returns
        ----------------------------
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
        """
        Description
        --------------
        Unroll the current epsilon-greedy policy from state and update the q-values at each step.
        
        Arguments
        --------------
        alpha   : Float in ]0, 1[, learning rate.
        epsilon : Float in ]0, 1[, parameter of the epsilon-greedy policy.
        
        Returns
        --------------
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
    """
    SARSA agent.
    """

    def __init__(self, env: Env[int, int], gamma: float = 0.99):
        """
        Description
        """

        super().__init__(env, gamma)
        self.q_values_1, self.q_values_2, self.visits = {}, {}, {}

    def reset(self) -> None:
        """
        Description
        --------------
        Reinitialize the action values estimates and the dictionary of visits.

        Arguments
        --------------

        Returns
        --------------
        """

        self.q_values_1, self.q_values_2, self.visits = {}, {}, {}

    def action(self, state: int):
        """
        Description
        --------------
        Take an action according to the estimated optimal policy.
        
        Arguments
        --------------
        state : Int, state.
        
        Returns
        --------------
        Int, estimated optimal action.
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
        """
        Description
        ----------------------------
        Perform a sarsa update of experience (state, action, reward, next_state).
        
        Parameters
        ----------------------------
        state      : Int, current state.
        action     : Int, performed action.
        reward     : Float, perceived reward.
        next_state : Int, next state.
        alpha      : Float, learning rate.
        epsilon    : Float, exploration parameter.
        
        Returns
        ----------------------------
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
        """
        Description
        --------------
        Unroll the current epsilon-greedy policy from state and update the q-values at each step.
        
        Arguments
        --------------
        alpha   : Float in ]0, 1[, learning rate.
        epsilon : Float in ]0, 1[, parameter of the epsilon-greedy policy.
        
        Returns
        --------------
        """

        state, _ = self.env.reset()
        done = False
        while not done:
            action = self.action_explore(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated or truncated)
            self.update_q_value(state, action, reward, next_state, alpha)
            state = next_state
    









