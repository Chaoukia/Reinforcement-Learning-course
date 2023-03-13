from frozen_lake_agent import *
from time import time
import gym
import argparse


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')
    
    argparser.add_argument('--map_name', type=str, default='8x8', help="Map grid, either 8x8 or 4x4.")
    argparser.add_argument('--gamma', type=float, default=0.9, help="Discount factor of the MDP.")
    argparser.add_argument('--algorithm', type=str, default='value_iteration', help="Dynamic Programming algorithm to use {value_iteration, q_iteration, policy_iteration}.")
    argparser.add_argument('--epsilon', type=float, default=1e-12, help="Discount factor of the MDP.")
    argparser.add_argument('--n_train', type=int, default=1000, help="Number of training episodes.")
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    
    args = argparser.parse_args()
    
    # Training.
    env = gym.make('FrozenLake-v1', is_slippery=True, map_name=args.map_name)
    agent = FrozenLakeAgent(env)
    start_time = time()
    if args.algorithm == 'value_iteration':
        agent.value_iteration(epsilon=args.epsilon, n=args.n_train)
        print('Execution time :', time() - start_time)
        
    elif args.algorithm == 'q_iteration':
        agent.q_iteration(epsilon=args.epsilon, n=args.n_train)
        print('Execution time :', time() - start_time)
        
    elif args.algorithm == 'policy_iteration':
        agent.policy_iteration(epsilon=args.epsilon, n=args.n_train)
        print('Execution time :', time() - start_time)
    
    # Test.
    env = gym.make('FrozenLake-v1', is_slippery=True, map_name=args.map_name, render_mode='human')
    agent.test(env, n_episodes=args.n_test)
    env.close()
    