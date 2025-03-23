import argparse
import gymnasium as gym
from agents import CliffWalking
from time import time

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')
    
    argparser.add_argument('--max_steps', type=float, default=100, help="Maximum number of steps per episode.")
    argparser.add_argument('--gamma', type=float, default=0.9, help="Discount factor of the MDP.")
    argparser.add_argument('--algorithm', type=str, default='value_iteration', help="Dynamic Programming algorithm to use {value_iteration, q_iteration, policy_iteration}.")
    argparser.add_argument('--epsilon', type=float, default=1e-12, help="Discount factor of the MDP.")
    argparser.add_argument('--n_train', type=int, default=1000, help="Number of training episodes.")
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=int, default=1, help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--save_gif', type=int, default=0, help="If 1, save gif of the tested agent, 0 otherwise.")
    
    args = argparser.parse_args()

    # Train.
    env = gym.make('CliffWalking-v0')
    agent = CliffWalking(env, gamma=args.gamma)
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
    env = gym.make('CliffWalking-v0', render_mode='human')
    agent.test(env, n_episodes=args.n_test, verbose=args.verbose)
    env.close()
    
    # Save gif.
    if args.save_gif:
        env = gym.make("CliffWalking-v0", render_mode='rgb_array')
        agent.save_gif(env, file_name='../gifs/cliff-walking.gif')
        env.close()