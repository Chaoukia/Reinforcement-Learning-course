import argparse
import gymnasium as gym
from agents import CliffWalkingAstar
from time import time

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')
    
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=int, default=1, help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--save_gif', type=int, default=0, help="If 1, save gif of the tested agent, 0 otherwise.")
    
    args = argparser.parse_args()

    # Train.
    env = gym.make('CliffWalking-v0')
    agent = CliffWalkingAstar(env)
    root, _ = env.reset()
    start_time = time()
    agent.train(root)
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