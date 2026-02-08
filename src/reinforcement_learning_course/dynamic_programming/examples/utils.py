import numpy as np
from gymnasium.core import Env


def frozen_lake_transition_matrices(env: Env[int, int]) -> tuple[np.array, np.array]:
    """
    Description
    ------------------------------
    Generate the probability and reward transition matrices for the frozen lake environment.

    Parameters
    ------------------------------
    env : Env, a frozen lake gymnasium environment

    Returns
    ------------------------------
    p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
    r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
    n_states     : Int, number of states.
    n_actions    : Int, number of actions.
    """

    n_states, n_actions = env.observation_space.n, env.action_space.n
    map = env.unwrapped.desc.astype(str)
    shape = map.shape

    p_transition, r_transition = np.zeros((n_states, n_actions, n_states)), np.zeros((n_states, n_actions, n_states))
    for state in range(n_states):
        state_index = np.unravel_index(state, shape)
        # Holes are absorbing states.
        if map[state_index] == 'H':
            p_transition[state, :, state] = 1
            
        else:
            for action in range(n_actions):
                if action == 0: # Go left.
                    # Going left.
                    next_state_index = np.array(state_index)
                    next_state_index[1] = max(0, next_state_index[1] - 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3

                    # Slipping up.
                    next_state_index = np.array(state_index)
                    next_state_index[0] = max(0, next_state_index[0] - 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3

                    # Slipping down.
                    next_state_index = np.array(state_index)
                    next_state_index[0] = min(shape[0] - 1, next_state_index[0] + 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3

                elif action == 2: # Go right.
                    # Going right.
                    next_state_index = np.array(state_index)
                    next_state_index[1] = min(shape[1] - 1, next_state_index[1] + 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3

                    # Slipping up.
                    next_state_index = np.array(state_index)
                    next_state_index[0] = max(0, next_state_index[0] - 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3

                    # Slipping down.
                    next_state_index = np.array(state_index)
                    next_state_index[0] = min(shape[0] - 1, next_state_index[0] + 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3

                elif action == 1: # Go down.
                    # Slipping left.
                    next_state_index = np.array(state_index)
                    next_state_index[1] = max(0, next_state_index[1] - 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3

                    # Going down.
                    next_state_index = np.array(state_index)
                    next_state_index[0] = min(shape[0] - 1, next_state_index[0] + 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3

                    # Slipping right.
                    next_state_index = np.array(state_index)
                    next_state_index[1] = min(shape[1] - 1, next_state_index[1] + 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3

                elif action == 3: # Go up.
                    # Slipping left.
                    next_state_index = np.array(state_index)
                    next_state_index[1] = max(0, next_state_index[1] - 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3

                    # Going up.
                    next_state_index = np.array(state_index)
                    next_state_index[0] = max(0, next_state_index[0] - 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3

                    # Slipping right.
                    next_state_index = np.array(state_index)
                    next_state_index[1] = min(shape[1] - 1, next_state_index[1] + 1)
                    next_state = next_state_index[0]*shape[0] + next_state_index[1]
                    p_transition[state, action, next_state] += 1/3
            
                for next_state in range(n_states):
                    next_state_index = np.unravel_index(next_state, shape)
                    if map[next_state_index] == 'G':
                        r_transition[state, action, next_state] = 1

    return p_transition, r_transition


def cliff_walking_transition_matrices(env: Env[int, int]) -> tuple[np.array, np.array]:
    """
    Description
    ------------------------------
    Generate the probability and reward transition matrices for the cliff walking environment

    Parameters
    ------------------------------
    env : Env, a cliff walking gymnasium environment

    Returns
    ------------------------------
    p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
    r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
    n_states     : Int, number of states.
    n_actions    : Int, number of actions.
    """
    
    n_states, n_actions = env.observation_space.n, env.action_space.n
    shape = (4, 12)

    p_transition, r_transition = np.zeros((n_states, n_actions, n_states)), np.full((n_states, n_actions, n_states), -1)
    for state in range(n_states):
        state_index = np.unravel_index(state, shape)
        # Going over the cliff sends us back immediately to the initial state.
        if state_index[0] == 3 and state_index[1] in set(range(1, 11)):
            p_transition[state, :, 36] = 1
            
        # The goal state is an absorbing state.
        elif state_index == (3, 11):
            p_transition[state, :, state] = 1
            r_transition[state, :, :] = 0

        else:
            for action in range(n_actions):
                if action == 0: # Go Up.
                    next_state_index = (max(0, state_index[0] - 1), state_index[1])
                    next_state = next_state_index[0]*shape[1] + next_state_index[1]
                    p_transition[state, action, next_state] = 1

                elif action == 1: # Go right.
                    next_state_index = (state_index[0], min(11, state_index[1] + 1))
                    next_state = next_state_index[0]*shape[1] + next_state_index[1]
                    p_transition[state, action, next_state] = 1

                elif action == 2: # Go down.
                    next_state_index = (min(3, state_index[0] + 1), state_index[1])
                    next_state = next_state_index[0]*shape[1] + next_state_index[1]
                    p_transition[state, action, next_state] = 1

                elif action == 3: # Go left.
                    next_state_index = (state_index[0], max(0, state_index[1] - 1))
                    next_state = next_state_index[0]*shape[1] + next_state_index[1]
                    p_transition[state, action, next_state] = 1
            
                for next_state in range(n_states):
                    next_state_index = np.unravel_index(next_state, shape)
                    # Going over the cliff incurs -100 reward.
                    if next_state_index[0] == 3 and next_state_index[1] in set(range(1, 11)):
                        r_transition[state, action, next_state] = -100
                    
    return p_transition, r_transition


def taxi_transition_matrices(env: Env[int, int]) -> tuple[np.array, np.array]:
    """
    Description
    ------------------------------
    Generate the probability and reward transition matrices for the taxi enviromnent.

    Parameters
    ------------------------------
    env : Env, a taxi gymnasium environment

    Returns
    ------------------------------
    p_transition : np.array of shape (n_state, n_actions, n_states), the transition probabilities matrix.
    r_transition : np.array of shape (n_state, n_actions, n_states), the transition rewards matrix.
    n_states     : Int, number of states.
    n_actions    : Int, number of actions.
    """

    n_states, n_actions = env.observation_space.n + 1, env.action_space.n

    p_transition, r_transition = np.zeros((n_states, n_actions, n_states)), np.zeros((n_states, n_actions, n_states))
    for state in range(n_states):
        # n_states-1 is the absorbing state.
        if state == n_states - 1:
            p_transition[state, :, state] = 1
            
        else:
            taxi_row, taxi_col, passenger_loc, destination = env.unwrapped.decode(state)
            action_mask = env.unwrapped.action_mask(state)
            actions_allowed = np.arange(n_actions)[action_mask == 1]
            for action in actions_allowed:
                if action == 0: # Go south.
                    taxi_row_new, taxi_col_new, passenger_loc_new = taxi_row + 1, taxi_col, passenger_loc
                    next_state = env.unwrapped.encode(taxi_row_new, taxi_col_new, passenger_loc_new, destination)

                    p_transition[state, action, next_state] = 1
                    r_transition[state, action, next_state] = -1

                elif action == 1: # Go north.
                    taxi_row_new, taxi_col_new, passenger_loc_new = taxi_row - 1, taxi_col, passenger_loc
                    next_state = env.unwrapped.encode(taxi_row_new, taxi_col_new, passenger_loc_new, destination)
                    p_transition[state, action, next_state] = 1
                    r_transition[state, action, next_state] = -1

                elif action == 2: # Go east.
                    taxi_row_new, taxi_col_new, passenger_loc_new = taxi_row, taxi_col + 1, passenger_loc
                    next_state = env.unwrapped.encode(taxi_row_new, taxi_col_new, passenger_loc_new, destination)
                    p_transition[state, action, next_state] = 1
                    r_transition[state, action, next_state] = -1

                elif action == 3: # Go west.
                    taxi_row_new, taxi_col_new, passenger_loc_new = taxi_row, taxi_col - 1, passenger_loc
                    next_state = env.unwrapped.encode(taxi_row_new, taxi_col_new, passenger_loc_new, destination)
                    p_transition[state, action, next_state] = 1
                    r_transition[state, action, next_state] = -1

                elif action == 4: # Pickup passenger.
                    taxi_row_new, taxi_col_new, passenger_loc_new = taxi_row, taxi_col, 4
                    next_state = env.unwrapped.encode(taxi_row_new, taxi_col_new, passenger_loc_new, destination)
                    p_transition[state, action, next_state] = 1
                    r_transition[state, action, next_state] = -1

                elif action == 5: # Drop off passenger and transition to the absorbing state.
                    p_transition[state, action, n_states - 1] = 1
                    # If the passenger is dropped at the correct destination, incur a reward of 20. Otherwise incur a reward of -10.
                    if (taxi_row, taxi_col) == env.unwrapped.locs[destination]:
                        r_transition[state, action, n_states - 1] = 20

                    else:
                        r_transition[state, action, n_states - 1] = -10
                    
    return p_transition, r_transition