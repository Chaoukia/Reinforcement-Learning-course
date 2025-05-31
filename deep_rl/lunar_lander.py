import argparse
import gymnasium as gym
from agents import LunarLanderDQN
from time import time

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')
    
    argparser.add_argument('--gamma', type=float, default=0.99, help="Discount factor of the MDP.")
    argparser.add_argument('--max_size', type=int, default=100000, help="Maximum size of the replay buffer.")
    argparser.add_argument('--n_pretrain', type=int, default=64, help="Number of episodes in the pretraining phase.")
    argparser.add_argument('--epsilon_start', type=float, default=1, help="Initial value of epsilon.")
    argparser.add_argument('--epsilon_stop', type=float, default=0.1, help="Final value of epsilon.")
    argparser.add_argument('--decay_rate', type=float, default=5e-6, help="Decay rate of epsilon.")
    argparser.add_argument('--n_train', type=int, default=10000, help="Number of training episodes.")
    argparser.add_argument('--n_learn', type=int, default=10, help="Number of iterations between two consecutive network updates.")
    argparser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    argparser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    argparser.add_argument('--thresh', type=float, default=250, help="Minimum mean reward to stop training.")
    argparser.add_argument('--file_save', type=str, default='lunar_lander_dqn.pth', help="Path where to save the network weights.")
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    argparser.add_argument('--verbose', type=int, default=1, help="Whether to print each episode evaluation during the test phase.")
    argparser.add_argument('--save_gif', type=int, default=0, help="If 1, save gif of the tested agent, 0 otherwise.")
    
    args = argparser.parse_args()

    # Train.
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=0.0, turbulence_power=0.0)
    agent = LunarLanderDQN(env, args.gamma, args.max_size)
    start_time = time()
    agent.train(n_episodes=args.n_train, n_pretrain=args.n_pretrain, epsilon_start=args.epsilon_start, epsilon_stop=args.epsilon_stop, decay_rate=args.decay_rate, 
                n_learn=args.n_learn, batch_size=args.batch_size, lr=args.lr, thresh=args.thresh, file_save=args.file_save, print_iter=100)
    print('Execution time :', time() - start_time)
    
    # Test.
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=0.0, turbulence_power=0.0, render_mode='human')
    agent.test(env, n_episodes=args.n_test, verbose=args.verbose)
    env.close()
    
    # Save gif.
    if args.save_gif:
        env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,enable_wind=False, wind_power=0.0, turbulence_power=0.0, render_mode='rgb_array')
        agent.save_gif(env, file_name='../gifs/lunar-lander.gif', n_episodes=3, duration=30)
        env.close()