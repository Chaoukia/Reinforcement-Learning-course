from dqn import *
from time import time
import gym
import argparse

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')
    
    argparser.add_argument('--train', type=int, default=1, help="If 0, test an already trained agent, in which case we need to specify the path to a .pth file.")
    argparser.add_argument('--file_load', type=str, default='lunar_lander', help="File name of the saved weights to load.")
    argparser.add_argument('--gamma', type=float, default=0.99, help="Discount factor of the MDP.")
    argparser.add_argument('--fc_1_dim', type=int, default=512, help="Dimension of the first network layer.")
    argparser.add_argument('--fc_2_dim', type=int, default=256, help="Dimension of the second network layer.")
    argparser.add_argument('--max_size', type=int, default=500000, help="Maximum size of the replay buffer.")
    argparser.add_argument('--n_pretrain', type=int, default=64, help="Number of pretraining episodes.")
    argparser.add_argument('--n_train', type=int, default=1000, help="Number of training episodes.")
    argparser.add_argument('--epsilon_start', type=float, default=1, help="Initial value of epsilon.")
    argparser.add_argument('--epsilon_stop', type=float, default=0.01, help="Final value of epsilon.")
    argparser.add_argument('--decay_rate', type=float, default=2e-5, help="Decay rate of epsilon.")
    argparser.add_argument('--n_learn', type=int, default=5, help="Number of iterations between two consecutive updates of the weights of the current network.")
    argparser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    argparser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    argparser.add_argument('--max_tau', type=int, default=25, help="Number of iterations between two consecutive updates of the weights of the target network.")
    argparser.add_argument('--log_dir', type=str, default='runs_lunar_lander', help="Directory in which to store the tensorboard events.")
    argparser.add_argument('--thresh', type=float, default=250, help="Average return over which we terminate training.")
    argparser.add_argument('--file_save', type=str, default='lunar_lander.pth', help="File name of the saved weights.")
    argparser.add_argument('--n_test', type=int, default=10, help="Number of test episodes.")
    argparser.add_argument('--save_gif', type=int, default=0, help="If 1, save gif of the tested agent, 0 otherwise.")
    
    args = argparser.parse_args()
    
    if args.train:
        env = gym.make("LunarLander-v2")
        agent = DQN(gamma=args.gamma, fc_1_dim=args.fc_1_dim, fc_2_dim=args.fc_2_dim, max_size=args.max_size)

        # Training.
        start_time = time()
        agent.train(env, n_episodes=args.n_train, n_pretrain=args.n_pretrain, epsilon_start=args.epsilon_start, epsilon_stop=args.epsilon_stop, decay_rate=args.decay_rate, 
                   n_learn=args.n_learn, batch_size=args.batch_size, lr=args.lr, max_tau=args.max_tau, log_dir=args.log_dir, thresh=args.thresh, file_save=args.file_save)
        print('Training time :', time() - start_time)
        
    else:
        agent = DQN(gamma=args.gamma, fc_1_dim=args.fc_1_dim, fc_2_dim=args.fc_2_dim, max_size=args.max_size)
        agent.load_weights(args.file_load)
    
    # Test.
    env = gym.make("LunarLander-v2", render_mode='human')
    agent.test(env, n_episodes=args.n_test)
    env.close()
    
    # Save gif.
    if args.save_gif:
        env = gym.make("LunarLander-v2", render_mode='rgb_array')
        agent.save_gif(env, file_name='../gifs/lunar-lander.gif')
        env.close()
    
    