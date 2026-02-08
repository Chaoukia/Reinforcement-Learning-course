import argparse
import gymnasium as gym
from reinforcement_learning_course.heuristics.examples import agents
from time import time

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')

    argparser.add_argument('--n_test', type=int, default=1, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=int, default=1, help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--path_gif', type=str, help="Path where to save the test gif. If not provided, no gif save will be performed.")
    
    args = argparser.parse_args()

    # Train
    env = gym.make('CliffWalking-v1')
    start_time = time()
    agent = agents.AstarCliffWalking(env)
    agent.train()
    print('Execution time :', time() - start_time)
    
    # Test.
    env = gym.make('CliffWalking-v1', render_mode='human')
    agent.set_env(env)
    agent.test(args.n_test, verbose=args.verbose)
    env.close()
    
    # Save gif.
    if args.path_gif is not None:
        env = gym.make("CliffWalking-v1", render_mode='rgb_array')
        agent.set_env(env)
        agent.save_gif(args.path_gif, n_episodes=1, duration=150)
        env.close()