'''create test set and test model'''
import os
import random
import pickle
from typing import Tuple, Union
import warnings
warnings.simplefilter("ignore", UserWarning)
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import torch
import torch.multiprocessing as mp
from environment import Environment
from model import Network
import config

warnings.simplefilter("ignore", UserWarning)
torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
os.environ['PYTHONHASHSEED'] = str(config.test_seed)
torch.cuda.manual_seed_all(config.test_seed)
torch.backends.cudnn.deterministic = True
DEVICE = torch.device('cpu')
torch.set_num_threads(1)


def create_test(test_env_settings: Tuple = config.test_env_settings, num_test_cases: int = config.num_test_cases):
    '''
    create test set
    '''

    for map_length, num_agents, density in test_env_settings:

        name = f'./test_set/{map_length}length_{num_agents}agents_{density}density.pth'
        print(f'-----{map_length}length {num_agents}agents {density}density-----')

        tests = []

        env = Environment(fix_density=density, num_agents=num_agents, map_length=map_length)

        for _ in tqdm(range(num_test_cases)):
            tests.append((np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)))
            env.reset(num_agents=num_agents, map_length=map_length)
        print()

        with open(name, 'wb') as f:
            pickle.dump(tests, f)



def test_model(epoch: int, test_set: Tuple = config.test_env_settings, path = config.gail_test_path):
    '''
    test model in 'saved_models' folder
    '''
    network = Network()
    network.eval()
    network.to(DEVICE)

    pool = mp.Pool(mp.cpu_count()//2)

    model_path = os.path.join(path, 'generator{}.pth'.format(epoch))
    state_dict = torch.load(model_path, map_location=DEVICE)
    network.load_state_dict(state_dict)
    network.eval()
    network.share_memory()

    res = {"success rate":{}, "average step":{}, "communication times":{}, "collision rate":{}, "collision with obstacles":{}, "collision with agents":{}}

    print(f'----------test model {model_path}----------')

    for case in test_set:
        print(f"test set: {case[0]} length {case[1]} agents {case[2]} density")
        with open('./test_set/{}length_{}agents_{}density.pth'.format(case[0], case[1], case[2]), 'rb') as f:
            tests = pickle.load(f)

        tests = [(test, network) for test in tests]
        ret = pool.map(test_one_case, tests)

        # ret = []
        # for test in tests:
        #     ret.append(test_one_case(test))

        success, steps, num_comm, collision, agent_collision, crowd_times, envmaps, heatmaps = zip(*ret)
        ms = sum(steps)/len(steps)

        setting = '{}_{}_{}'.format(case[0], case[1], case[2])
        res["success rate"][setting] = sum(success)/len(success)*100
        res["average step"][setting] = sum(steps)/len(steps)
        res["communication times"][setting] = sum(num_comm)/len(num_comm)
        res["collision rate"][setting] = sum(collision) / (len(collision) * ms)
        res["collision with obstacles"][setting] = (sum(collision) - sum(agent_collision)) / len(collision)
        res["collision with agents"][setting] = sum(agent_collision) / len(collision)

        print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
        print("average step: {}".format(sum(steps)/len(steps)))
        print("communication times: {}".format(sum(num_comm)/len(num_comm)))
        print("collision rate: {:.2f}".format(sum(collision) / (len(collision) * ms)))
        print("collision with obstacles: {:.2f}".format((sum(collision) - sum(agent_collision)) / len(collision)))
        print("collision with agents: {:.2f}".format(sum(agent_collision) / len(collision)))
        
        # crowd_each_case = list(zip(success, steps, crowd_times))
        # np.savetxt('crowd/{}_{}_{}.csv'.format(case[0], case[1], case[2]), crowd_each_case, delimiter=',', fmt='%d')

        # heatmap_savepath = 'heatmap/{}_{}_{}'.format(case[0], case[1], case[2])
        # if not os.path.exists(heatmap_savepath):
        #     os.makedirs(heatmap_savepath)
        # for i, envmap in enumerate(envmaps):
        #     np.savetxt('heatmap/{}_{}_{}/envmap_{}.csv'.format(case[0], case[1], case[2], i), envmap, delimiter=',', fmt='%d')

        # for i, heatmap in enumerate(heatmaps):
        #     np.savetxt('heatmap/{}_{}_{}/heatmap_{}.csv'.format(case[0], case[1], case[2], i), heatmap, delimiter=',', fmt='%d')

        print()
        
    return res

def test_one_case(args):

    env_set, network = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, urgency, crowd, last_act, pos = env.observe()
    
    done = False
    network.reset()

    step = 0
    num_comm = 0
    while not done and env.steps < config.max_episode_length:
        actions, _, _, _, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE),
                                                    torch.as_tensor(urgency.astype(np.float32)).to(DEVICE),
                                                    torch.as_tensor(crowd.astype(np.float32)).to(DEVICE),
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(np.int)))
        (obs, urgency, crowd, last_act, pos), _, done, _ = env.step(actions)
        step += 1
        num_comm += np.sum(comm_mask[0]) + np.sum(comm_mask[1])

    return np.array_equal(env.agents_pos, env.goals_pos), step, num_comm, env.collision_times, env.agent_collision_times, env.crowd_times, env.map, env.heatmap


def make_animation(model_name: int, test_set_name: tuple, test_case_idx: int, steps: int):
    '''
    visualize running results
    model_name: model number in 'models' file
    test_set_name: (length, num_agents, density)
    test_case_idx: int, the test case index in test set
    steps: how many steps to visualize in test case
    '''
    color_map = np.array([[255, 255, 255],   # white
                    [190, 190, 190],   # gray
                    [0, 191, 255],   # blue
                    [255, 165, 0],   # orange
                    [0, 250, 154]])  # green

    network = Network()
    network.eval()
    device = torch.device('cpu')
    network.to(device)
    state_dict = torch.load('saved_models/generator{}.pth'.format(model_name), map_location=device)
    network.load_state_dict(state_dict)

    test_name = f'./test_set/{test_set_name[0]}length_{test_set_name[1]}agents_{test_set_name[2]}density.pth'
    with open(test_name, 'rb') as f:
        tests = pickle.load(f)

    env = Environment()
    env.load(tests[test_case_idx][0], tests[test_case_idx][1], tests[test_case_idx][2])

    fig = plt.figure()
            
    done = False
    obs, urgency, crowd, last_act, pos = env.observe()

    imgs = []
    while not done and env.steps < steps:
        imgs.append([])
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)

        imgs[-1].append(img)

        for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(env.agents_pos, env.goals_pos)):
            text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)
            text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)

        actions, _, _, _, _ = network.step(torch.from_numpy(obs.astype(np.float32)).to(device),
                                           torch.from_numpy(urgency.astype(np.float32)).to(device),
                                           torch.from_numpy(crowd.astype(np.float32)).to(device),
                                           torch.from_numpy(last_act.astype(np.float32)).to(device),
                                           torch.from_numpy(pos.astype(np.float32)).type(torch.long).to(device))
        (obs, _, _, _, pos), _, done, _ = env.step(actions)
        # print(done)

    if done and env.steps < steps:
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)
        for _ in range(steps-env.steps):
            imgs.append([])
            imgs[-1].append(img)
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(env.agents_pos, env.goals_pos)):
                text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
                imgs[-1].append(text)
                text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
                imgs[-1].append(text)


    ani = animation.ArtistAnimation(fig, imgs, interval=600, blit=True, repeat_delay=1000)

    ani_path = 'videos/{}_{}_{}'.format(*test_set_name)
    if not os.path.exists(ani_path):
        os.makedirs(ani_path)
    ani.save('videos/{}_{}_{}/{}.gif'.format(*test_set_name, test_case_idx), writer='pillow')


if __name__ == '__main__':
    # create test environments
    # create_test(test_env_settings=config.test_env_settings, num_test_cases=config.num_test_cases)
    
    # load trained model and reproduce results in paper
    test_model(85000)

    # Visualizaiton
    # for test_env_setting in config.test_env_settings:
    #     for i in range(200):
    #         make_animation(85000, test_env_setting, i ,config.max_episode_length)
