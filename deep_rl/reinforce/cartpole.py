import argparse
import gymnasium as gym
from agents import CartPoleReinforce
from time import time

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')
    
    argparser.add_argument('--gamma', type=float, default=0.99, help="Discount factor of the MDP.")
    argparser.add_argument('--n_train', type=int, default=10000, help="Number of training episodes.")
    argparser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    argparser.add_argument('--alpha_entropy', type=float, default=0.0, help="Penalty on the entropy loss term.")
    argparser.add_argument('--thresh', type=float, default=400, help="Minimum mean reward to stop training.")
    argparser.add_argument('--file_save', type=str, default='weights/cartpole_Reinforce.pth', help="Path where to save the network weights.")
    argparser.add_argument('--log_dir', type=str, default='runs/', help="Path where to log the tensorboard runs.")
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=int, default=1, help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--print_iter', type=int, default=100, help="If 1, save gif of the tested agent, 0 otherwise.")
    argparser.add_argument('--save_gif', type=int, default=0, help="If 1, save gif of the tested agent, 0 otherwise.")
    
    args = argparser.parse_args()
    
    # Train.
    env = gym.make("CartPole-v1")
    agent = CartPoleReinforce(args.gamma)
    start_time = time()
    agent.train(env, n_episodes=args.n_train, lr=args.lr, alpha_entropy=args.alpha_entropy, thresh=args.thresh, file_save=args.file_save, log_dir=args.log_dir, print_iter=args.print_iter)
    print('Execution time :', time() - start_time)
    
    # Test.
    env = gym.make("CartPole-v1", render_mode='human')
    agent.test(env, n_episodes=args.n_test, verbose=args.verbose)
    env.close()
    
    # Save gif.
    if args.save_gif:
        env = gym.make("CartPole-v1", render_mode='rgb_array')
        agent.save_gif(env, file_name='../../gifs/cartpole.gif', n_episodes=3, duration=30)
        env.close()