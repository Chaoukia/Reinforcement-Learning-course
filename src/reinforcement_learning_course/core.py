import numpy as np
from PIL import Image
from typing import Generic
from gymnasium import Env
from gymnasium.core import ObsType, ActType

class Agent(Generic[ObsType, ActType]):
    """
    Abstract class of an RL agent.
    """

    def __init__(self, env: Env[ObsType, ActType], gamma: float = 1.0) -> None:
        """
        Description
        ----------------------
        Construtor.

        Parameters & Attributes
        ----------------------
        env   : gymnasium environment where the agent acts.
        gamma : Float, discount factor.

        Returns
        ----------------------
        """

        self.env = env
        self.gamma = gamma

    def set_env(self, env: Env[ObsType, ActType]) -> None:
        """
        Description
        ----------------------
        Set the agent's environment.

        Parameters
        ----------------------
        env : gymnasium environment.

        Returns
        ----------------------
        """

        self.env = env

    def action(self, state: ObsType) -> ActType:
        """
        Description
        ----------------------
        Choose the action to perform at a state.

        Parameters
        ----------------------
        state : StateType, The state where the agent needs to choose an action.

        Returns
        ----------------------
        """

        raise NotImplementedError
    
    def train(self) -> None:
        """
        Description
        ----------------------
        Train the agent.

        Parameters
        ----------------------
        
        Returns
        ----------------------
        """

        raise NotImplementedError
    
    def test(self, n_episodes: int = 1000, verbose: bool = False) -> tuple[float, float]:
        """
        Description
        --------------
        Test the agent.
        
        Parameters
        --------------
        n_episodes : Int, number of test episodes.
        verbose    : Boolean, if True, print the episode index and its corresponding length and return.
        
        Returns
        --------------
        return_mean : Float, mean of the episode return.
        return_std  : Float, standard deviation of the episode return.
        """
        
        returns = np.empty(n_episodes)
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            R = 0
            n_steps = 0
            while not done:
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = (terminated or truncated)
                state = next_state
                # R += reward*self.gamma**n_steps
                R += reward
                n_steps += 1
                
            returns[episode] = R
            if verbose:
                print('Episode : %d, length : %d, return : %.3F' %(episode, n_steps, R))

        return_mean, return_std = returns.mean(), returns.std()
        print('mean : %.3f, std : %.3f' %(return_mean, return_std))
        return return_mean, return_std
            
    def save_gif(self, path: str, n_episodes: int = 1, duration: int = 150) -> None:
        """
        Description
        --------------
        Test the agent and save a gif.
        
        Arguments
        --------------
        file_name   : String, path to the saved gif.
        n_episoodes : Int, number of episodes.
        duration    : Int.
        
        Returns
        --------------
        """
        
        frames = []
        for i in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            R = 0
            n_steps = 0
            while not done:
                frames.append(Image.fromarray(self.env.render(), mode='RGB'))
                action = self.action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = (terminated or truncated)
                state = next_state
                R += reward*self.gamma**n_steps
                n_steps += 1

            frames.append(Image.fromarray(self.env.render(), mode='RGB'))
            
        frames[0].save(path, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)
    
    def save(self, path: str) -> None:
        """
        Description
        ----------------------
        Save the agent.

        Parameters
        ----------------------
        path: String, path where to save the agent.

        Returns
        ----------------------
        """

        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """
        Description
        ----------------------
        Load the agent.

        Parameters
        ----------------------
        path: String, path where to load the agent.

        Returns
        ----------------------
        """

        raise NotImplementedError
    

    