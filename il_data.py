import numpy as np
import random
import math
import pickle
from tqdm import tqdm

from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
from od_mstar3 import cpp_mstar

from environment import Environment
import config


def astar(world, start, goal):
    try:
        # Use ODrM* to find the path
        path = cpp_mstar.find_path(world, start, goal, 2, 5)
    except Exception as e:
        path = None

    return path


def generate_expert_trajs(env):
    '''
    actions:
        list of indices
            0 up
            1 down
            2 left
            3 right
            4 stay
    '''
    direction_dic = {(-1,0):0, (1,0):1, (0,-1):2, (0,1):3, (0,0):4}
    path = astar(env.map, env.agents_pos, env.goals_pos)
    
    action_list = []
    if path is not None:
        if len(path)>256:
            print(len(path))
        for timestep in range(min(len(path) - 1, config.max_episode_length)):
            direction = [(x1 - x2, y1 - y2) for (x1, y1), (x2, y2) in zip(path[timestep+1], path[timestep])]
            actions = [direction_dic[d] for d in direction]
            action_list.append(actions)
    else:
        return None
    
    return np.stack(action_list)
    

def generate_dataset(num_iterations = 10000):
    # datasets = {}
    env = Environment()
    with tqdm(range(config.init_env_settings[0], config.max_num_agents+1)) as pbar:
        for num_agents in pbar:
            for map_len in range(config.init_env_settings[1], config.max_map_length+1, 5):
                pbar.set_postfix({"num_agents":num_agents, "map_len":map_len})
                dataset = []
                for _ in range(num_iterations):
                    try:
                        env.reset(num_agents, map_len)
                    except:
                        continue
                    action_list = generate_expert_trajs(env)
                    # print(num_agents, map_len)
                    if action_list is not None:
                        data = {'env':[env.map, env.agents_pos, env.goals_pos], 'action_list':action_list}
                        dataset.append(data)
                with open('data/dataset_{}numagents_{}maplen_10000.pkl'.format(num_agents, map_len), 'wb') as f:
                    pickle.dump(dataset, f)
                # datasets[(num_agents, map_len)] = dataset

# Generate datasets
generate_dataset()

# Read the saved datasets
with open('data/dataset.pkl', 'rb') as f:
    loaded_dataset = pickle.load(f)
    print("done")
    for name, data in loaded_dataset.items():
        print(name, len(data))
