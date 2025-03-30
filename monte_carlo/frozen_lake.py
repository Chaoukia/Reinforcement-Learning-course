import argparse
import gymnasium as gym
from agents import FrozenLakeMC
from time import time

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')
    
    argparser.add_argument('--gamma', type=float, default=0.99, help="Discount factor of the MDP.")
    argparser.add_argument('--map_name', type=str, default='4x4', help="Map grid, either 8x8 or 4x4.")
    argparser.add_argument('--is_slippery', type=int, default=1, help="Whether the map is slippery or not.")
    argparser.add_argument('--first_visit', type=int, default=0, help="Whether to apply first visit MC or every visit MC.")
    argparser.add_argument('--decay_rate', type=float, default=1e-3, help="Decay rate of epsilon.")
    argparser.add_argument('--n_train', type=int, default=100000, help="Number of training episodes.")
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=int, default=1, help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--save_gif', type=int, default=0, help="If 1, save gif of the tested agent, 0 otherwise.")
    
    args = argparser.parse_args()

    # Train.
    env = gym.make('FrozenLake-v1', is_slippery=args.is_slippery, map_name=args.map_name)
    agent = FrozenLakeMC(env, gamma=args.gamma)
    start_time = time()
    agent.train(n_train=args.n_train, print_iter=1000, first_visit=args.first_visit, decay_rate=args.decay_rate)
    print('Execution time :', time() - start_time)
    
    # Test.
    env = gym.make('FrozenLake-v1', is_slippery=args.is_slippery, map_name=args.map_name, render_mode='human')
    agent.test(env, n_episodes=args.n_test, verbose=args.verbose)
    env.close()
    
    # Save gif.
    if args.save_gif:
        env = gym.make("FrozenLake-v1", is_slippery=args.is_slippery, map_name=args.map_name, render_mode='rgb_array')
        agent.save_gif(env, file_name='../gifs/frozen-lake.gif')
        env.close()