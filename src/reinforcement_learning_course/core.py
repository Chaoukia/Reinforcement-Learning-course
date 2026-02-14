import numpy as np
from PIL import Image
from typing import Generic
from gymnasium import Env
from gymnasium.core import ObsType, ActType

class Agent(Generic[ObsType, ActType]):
    """Abstract base class for RL agents.
    
    This class provides the common interface and utilities for reinforcement learning
    agents interacting with gymnasium environments.
    """

    def __init__(self, env: Env[ObsType, ActType], gamma: float = 1.0) -> None:
        """Initialize the RL agent.
        
        Args:
            env: Gymnasium environment where the agent acts.
            gamma: Discount factor, typically in [0, 1].
        
        Attributes:
            env: The gymnasium environment.
            gamma: The discount factor.
        """

        self.env = env
        self.gamma = gamma

    def set_env(self, env: Env[ObsType, ActType]) -> None:
        """Set the agent's environment.
        
        Args:
            env: Gymnasium environment to set.
        """

        self.env = env

    def action(self, state: ObsType) -> ActType:
        """Choose an action for the given state.
        
        This is an abstract method that must be implemented by subclasses.
        
        Args:
            state: The current state from the environment.
        
        Returns:
            The action to perform.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError
    
    def train(self) -> None:
        """Train the agent.
        
        This is an abstract method that must be implemented by subclasses.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError
    
    def test(self, n_episodes: int = 1000, verbose: bool = False) -> tuple[float, float]:
        """Test the agent's performance over multiple episodes.
        
        Evaluates the agent's policy without learning by running episodes and collecting
        return statistics.
        
        Args:
            n_episodes: Number of test episodes to run. Defaults to 1000.
            verbose: If True, print episode statistics. Defaults to False.
        
        Returns:
            A tuple containing:
                - return_mean: Mean episode return across all test episodes.
                - return_std: Standard deviation of episode returns.
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
        """Test the agent and save the episodes as an animated GIF.
        
        Runs episodes using the current policy and saves rendered frames as an
        animated GIF file.
        
        Args:
            path: File path where the GIF will be saved.
            n_episodes: Number of episodes to record. Defaults to 1.
            duration: Duration of each frame in milliseconds. Defaults to 150.
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
        """Save the agent to a file.
        
        This is an abstract method that must be implemented by subclasses.
        
        Args:
            path: File path where the agent will be saved.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """Load the agent from a file.
        
        This is an abstract method that must be implemented by subclasses.
        
        Args:
            path: File path from where the agent will be loaded.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError
    

    