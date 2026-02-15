import argparse
import gymnasium as gym
from reinforcement_learning_course.deep_rl.reinforce_baseline.examples import agents
from time import time

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')

    argparser.add_argument('--gamma', type=float, default=0.99, help="Discount factor of the MDP.")
    argparser.add_argument('--n_train', type=int, default=10000, help="Number of training episodes.")
    argparser.add_argument('--lr_policy', type=float, default=1e-4, help="Learning rate for the policy network.")
    argparser.add_argument('--lr_value', type=float, default=1e-4, help="Learning rate for the value network.")
    argparser.add_argument('--alpha_entropy', type=float, default=0.0, help="Penalty on the entropy loss term.")
    argparser.add_argument('--thresh', type=float, default=400, help="Minimum mean reward to stop training.")
    argparser.add_argument('--log_dir', type=str, default='runs', help="Path where to log the tensorboard runs.")
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=int, default=1, help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--print_iter', type=int, default=100, help="If 1, save gif of the tested agent, 0 otherwise.")
    argparser.add_argument('--path_gif', type=str, help="Path where to save the test gif. If not provided, no gif save will be performed.")
    
    args = argparser.parse_args()

    # Train
    env = gym.make("CartPole-v1")
    start_time = time()
    agent = agents.CartPoleReinforceBaseline(env, args.gamma)
    start_time = time()
    agent.train(n_episodes=args.n_train, lr_policy=args.lr_policy, lr_value=args.lr_value, alpha_entropy=args.alpha_entropy, 
                thresh=args.thresh, log_dir=args.log_dir, print_iter=args.print_iter)
    print('Execution time :', time() - start_time)
    
    # Test
    env = gym.make("CartPole-v1", render_mode='human')
    agent.set_env(env)
    agent.test(args.n_test, verbose=args.verbose)
    env.close()
    
    # Save gif
    if args.path_gif is not None:
        env = gym.make("CartPole-v1", render_mode='rgb_array')
        agent.set_env(env)
        agent.save_gif(args.path_gif, n_episodes=1, duration=150)
        env.close()