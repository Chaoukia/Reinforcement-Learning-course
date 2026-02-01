import argparse
import gymnasium as gym
from reinforcement_learning_course.monte_carlo import algorithms as algs
from time import time

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')

    argparser.add_argument('--gamma', type=float, default=0.99, help="Discount factor of the MDP.")
    argparser.add_argument('--map_name', type=str, choices=['4x4', '8x8'], default='4x4', help="Map grid, either 8x8 or 4x4.")
    argparser.add_argument('--is_slippery', type=str, choices=['no', 'yes'], default='no', help="Whether the map is slippery or not.")
    argparser.add_argument('--first_visit', type=str, choices=['no', 'yes'], default='yes', help="Whether to apply first visit MC or every visit MC.")
    argparser.add_argument('--epsilon_start', type=float, default=1., help="Initial value of epsilon.")
    argparser.add_argument('--epsilon_stop', type=float, default=0.1, help="Final value of epsilon.")
    argparser.add_argument('--decay_rate', type=float, default=1e-3, help="Decay rate of epsilon.")
    argparser.add_argument('--n_train', type=int, default=10000, help="Number of training episodes.")
    argparser.add_argument('--print_iter', type=int, default=100, help="Number of training episodes.")
    argparser.add_argument('--n_test', type=int, default=5, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=str, choices=['no', 'yes'], default='yes', help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--path_gif', type=str, help="Path where to save the test gif. If not provided, no gif save will be performed.")
    
    args = argparser.parse_args()

    is_slippery = args.is_slippery == 'yes'
    first_visit = args.first_visit == 'yes'
    verbose = args.verbose == 'yes'

    # Train
    env = gym.make('FrozenLake-v1', is_slippery=is_slippery, map_name=args.map_name)
    start_time = time()
    agent = algs.MonteCarlo(env, args.gamma)
    agent.train(args.epsilon_start, args.epsilon_stop, args.decay_rate, args.n_train, first_visit, args.print_iter)
    print('Execution time :', time() - start_time)
    
    # Test.
    env = gym.make('FrozenLake-v1', is_slippery=is_slippery, map_name=args.map_name, render_mode='human')
    agent.set_env(env)
    agent.test(args.n_test, verbose=args.verbose)
    env.close()
    
    # Save gif.
    if args.path_gif is not None:
        env = gym.make('FrozenLake-v1', is_slippery=is_slippery, map_name=args.map_name, render_mode='rgb_array')
        agent.set_env(env)
        agent.save_gif(args.path_gif, n_episodes=1, duration=150)
        env.close()