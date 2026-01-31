import argparse
import gymnasium as gym
from reinforcement_learning_course.dynamic_programming import algorithms as algs
from reinforcement_learning_course.dynamic_programming.utils import frozen_lake_transition_matrices
from time import time

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')
    
    argparser.add_argument('--map_name', type=str, choices=['4x4', '8x8'], default='4x4', help="Map grid, either 8x8 or 4x4.")
    argparser.add_argument('--gamma', type=float, default=0.9, help="Discount factor of the MDP.")
    argparser.add_argument('--algorithm', type=str, choices=['value_iteration', 'q_iteration', 'policy_iteration'], default='value_iteration', help="Dynamic Programming algorithm to use {value_iteration, q_iteration, policy_iteration}.")
    argparser.add_argument('--epsilon', type=float, default=1e-12, help="Discount factor of the MDP.")
    argparser.add_argument('--n_train', type=int, default=1000, help="Number of training episodes.")
    argparser.add_argument('--n_test', type=int, default=5, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=int, default=1, help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--path_gif', type=str, help="Path where to save the test gif. If not provided, no gif save will be performed.")
    
    args = argparser.parse_args()

    # Train
    env = gym.make('FrozenLake-v1', is_slippery=True, map_name=args.map_name)
    start_time = time()
    if args.algorithm == 'value_iteration':
        agent = algs.ValueIteration(env, frozen_lake_transition_matrices, args.gamma)
        
    elif args.algorithm == 'q_iteration':
        agent = algs.QIteration(env, frozen_lake_transition_matrices, args.gamma)
        
    elif args.algorithm == 'policy_iteration':
        agent = algs.PolicyIteration(env, frozen_lake_transition_matrices, args.gamma)

    agent.train(args.n_train, args.epsilon)
    print('Execution time :', time() - start_time)
    
    # Test.
    env = gym.make('FrozenLake-v1', is_slippery=True, map_name=args.map_name, render_mode='human')
    agent.set_env(env)
    agent.test(args.n_test, verbose=args.verbose)
    env.close()
    
    # Save gif.
    if args.path_gif is not None:
        env = gym.make("FrozenLake-v1", is_slippery=True, map_name=args.map_name, render_mode='rgb_array')
        agent.set_env(env)
        agent.save_gif(args.path_gif, n_episodes=1, duration=150)
        env.close()