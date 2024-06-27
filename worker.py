import time
import random
import os
import pickle
import sys

import psutil
import pynvml
from copy import deepcopy
from typing import Deque, List, Tuple, Dict
import threading
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler
import numpy as np
from model import Network, Discriminator
from environment import Environment
from buffer import SumTree, LocalBuffer, EpisodeData


from utils import save_model, get_module, get_super_module, set_module, freeze_module
import config

from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
from od_mstar3 import cpp_mstar

from torch.utils.tensorboard import SummaryWriter
from utils import get_formatted_time
from pympler.asizeof import asizeof


@ray.remote(num_cpus=1)
class GlobalBuffer:
    def __init__(self, buffer_capacity=config.buffer_capacity, init_env_settings=config.init_env_settings,
                alpha=config.prioritized_replay_alpha, beta=config.prioritized_replay_beta, chunk_capacity=config.chunk_capacity):

        self.capacity = buffer_capacity
        self.chunk_capacity = chunk_capacity
        self.num_chunks = buffer_capacity // chunk_capacity
        self.ptr = 0

        # prioritized experience replay
        self.priority_tree = SumTree(buffer_capacity)
        self.alpha = alpha
        self.beta = beta

        self.counter = 0
        self.batched_data = []
        self.stat_dict = {init_env_settings:[]}
        self.lock = threading.Lock()
        self.env_settings_set = ray.put([init_env_settings])

        self.obs_buf = [None] * self.num_chunks
        self.urgency_buf = [None] * self.num_chunks
        self.crowd_buf = [None] * self.num_chunks
        self.last_act_buf = [None] * self.num_chunks
        self.act_buf = np.zeros((buffer_capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((buffer_capacity), dtype=np.float16)
        self.pred_d_buf = np.zeros((buffer_capacity), dtype=np.float16)
        self.hid_buf = [None] * self.num_chunks
        self.size_buf = np.zeros(self.num_chunks, dtype=np.uint8)
        self.relative_pos_buf = [None] * self.num_chunks
        self.comm_mask_buf = [None] * self.num_chunks
        self.gamma_buf = np.zeros((self.capacity), dtype=np.float16)
        self.num_agents_buf = np.zeros((self.num_chunks), dtype=np.uint8)
        
        self.writer = SummaryWriter('./tb_log/log-{}'.format(get_formatted_time()))
        self.writer_cnt = 0

    def __len__(self):
        return np.sum(self.size_buf)

    def run(self):
        self.background_thread = threading.Thread(target=self._prepare_data, daemon=True)
        self.background_thread.start()

    def _prepare_data(self):
        while True:
            if len(self.batched_data) <= 4:
                data = self.sample_batch(config.batch_size)
                data_id = ray.put(data)
                self.batched_data.append(data_id)
                
            else:
                time.sleep(0.1)
        
    def get_batched_data(self):
        '''
        get one batch of data, called by learner.
        '''
        if len(self.batched_data) == 0:
            # print('no prepared data')
            data = self._sample_batch(config.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)

    def add(self, data: EpisodeData):
        '''
        Add one episode data into replay buffer, called by actor if actor finished one episode.

        data: ('actor_id', 'num_agents', 'map_len', 'obs', 'urgency', 'crowd', 'last_act', 'actions', 'rewards',
                'hiddens', 'relative_pos', 'comm_mask', 'gammas', 'td_errors', 'sizes', 'done')
        '''
        if data.actor_id >= 9: #eps-greedy < 0.01
            stat_key = (data.num_agents, data.map_len)
            if stat_key in self.stat_dict:
                self.stat_dict[stat_key].append(data.done)
                if len(self.stat_dict[stat_key]) == config.cl_history_size+1:
                    self.stat_dict[stat_key].pop(0)

        with self.lock:

            for i, size in enumerate(data.sizes):
                idxes = np.arange(self.ptr*self.chunk_capacity, (self.ptr+1)*self.chunk_capacity)
                start_idx = self.ptr*self.chunk_capacity
                # update buffer size
                self.counter += size

                self.priority_tree.batch_update(idxes, data.td_errors[i*self.chunk_capacity:(i+1)*self.chunk_capacity]**self.alpha)

                self.obs_buf[self.ptr] = np.copy(data.obs[i*self.chunk_capacity:(i+1)*self.chunk_capacity+config.burn_in_steps+config.forward_steps])
                self.urgency_buf[self.ptr] = np.copy(data.urgency[i*self.chunk_capacity:(i+1)*self.chunk_capacity+config.burn_in_steps+config.forward_steps])
                self.crowd_buf[self.ptr] = np.copy(data.crowd[i*self.chunk_capacity:(i+1)*self.chunk_capacity+config.burn_in_steps+config.forward_steps])
                self.last_act_buf[self.ptr] = np.copy(data.last_act[i*self.chunk_capacity:(i+1)*self.chunk_capacity+config.burn_in_steps+config.forward_steps])
                self.act_buf[start_idx:start_idx+size] = data.actions[i*self.chunk_capacity:i*self.chunk_capacity+size]
                self.rew_buf[start_idx:start_idx+size] = data.rewards[i*self.chunk_capacity:i*self.chunk_capacity+size]
                self.pred_d_buf[start_idx:start_idx+size] = data.pred_d[i*self.chunk_capacity:i*self.chunk_capacity+size]
                self.hid_buf[self.ptr] = np.copy(data.hiddens[i*self.chunk_capacity:i*self.chunk_capacity+size+config.forward_steps])
                self.size_buf[self.ptr] = size
                self.relative_pos_buf[self.ptr] = np.copy(data.relative_pos[i*self.chunk_capacity:(i+1)*self.chunk_capacity+config.burn_in_steps+config.forward_steps])
                self.comm_mask_buf[self.ptr] = np.copy(data.comm_mask[i*self.chunk_capacity:(i+1)*self.chunk_capacity+config.burn_in_steps+config.forward_steps])
                self.gamma_buf[start_idx:start_idx+size] = data.gammas[i*self.chunk_capacity:i*self.chunk_capacity+size]
                self.num_agents_buf[self.ptr] = data.num_agents

                self.ptr = (self.ptr+1) % self.num_chunks
            
            del data
            

    def _sample_batch(self, batch_size: int) -> Tuple:

        b_obs, b_urgency, b_crowd, b_last_act, b_steps, b_relative_pos, b_comm_mask = [], [], [], [], [], [], []
        b_hidden = []
        idxes, priorities = [], []

        with self.lock:

            idxes, priorities = self.priority_tree.batch_sample(batch_size)
            global_idxes = idxes // self.chunk_capacity
            local_idxes = idxes % self.chunk_capacity
            max_num_agents = np.max(self.num_agents_buf[global_idxes])

            for global_idx, local_idx in zip(global_idxes.tolist(), local_idxes.tolist()):
                
                assert local_idx < self.size_buf[global_idx], 'index is {} but size is {}'.format(local_idx, self.size_buf[global_idx])

                steps = min(config.forward_steps, self.size_buf[global_idx].item()-local_idx)

                relative_pos = self.relative_pos_buf[global_idx][local_idx:local_idx+config.burn_in_steps+steps+1]
                comm_mask = self.comm_mask_buf[global_idx][local_idx:local_idx+config.burn_in_steps+steps+1]
                obs = self.obs_buf[global_idx][local_idx:local_idx+config.burn_in_steps+steps+1]
                urgency = self.urgency_buf[global_idx][local_idx:local_idx+config.burn_in_steps+steps+1]
                crowd = self.crowd_buf[global_idx][local_idx:local_idx+config.burn_in_steps+steps+1]
                last_act = self.last_act_buf[global_idx][local_idx:local_idx+config.burn_in_steps+steps+1]
                hidden = self.hid_buf[global_idx][local_idx]

                if steps < config.forward_steps:
                    pad_len = config.forward_steps - steps
                    obs = np.pad(obs, ((0, pad_len), (0, 0), (0, 0), (0, 0), (0, 0)))
                    urgency = np.pad(urgency, ((0, pad_len), (0, 0), (0, 0)))
                    crowd = np.pad(crowd, ((0, pad_len), (0, 0), (0, 0)))
                    last_act = np.pad(last_act, ((0, pad_len), (0, 0), (0, 0)))
                    relative_pos = np.pad(relative_pos, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
                    comm_mask = np.pad(comm_mask, ((0, pad_len), (0, 0), (0, 0), (0, 0)))

                if self.num_agents_buf[global_idx] < max_num_agents:
                    pad_len = max_num_agents - self.num_agents_buf[global_idx].item()
                    obs = np.pad(obs, ((0, 0), (0, pad_len), (0, 0), (0, 0), (0, 0)))
                    urgency = np.pad(urgency, ((0, 0), (0, pad_len), (0, 0)))
                    crowd = np.pad(crowd, ((0, 0), (0, pad_len), (0, 0)))
                    last_act = np.pad(last_act, ((0, 0), (0, pad_len), (0, 0)))
                    relative_pos = np.pad(relative_pos, ((0, 0), (0, pad_len), (0, pad_len), (0, 0)))
                    comm_mask = np.pad(comm_mask, ((0, 0), (0, 0), (0, pad_len), (0, pad_len)))
                    hidden = np.pad(hidden, ((0, pad_len), (0, 0)))

                b_obs.append(obs)
                b_urgency.append(urgency)
                b_crowd.append(crowd)
                b_last_act.append(last_act)
                b_steps.append(steps)
                b_relative_pos.append(relative_pos)
                b_comm_mask.append(comm_mask)
                b_hidden.append(hidden)

            # importance sampling weight
            min_p = np.min(priorities)
            weights = np.power(priorities/min_p, -self.beta)

            b_action = self.act_buf[idxes]
            b_reward = self.rew_buf[idxes]
            b_pred_d = self.pred_d_buf[idxes]
            b_gamma = self.gamma_buf[idxes]

            data = (
                torch.from_numpy(np.stack(b_obs)).transpose(1,0).contiguous(),
                torch.from_numpy(np.stack(b_urgency)).transpose(1,0).contiguous(),
                torch.from_numpy(np.stack(b_crowd)).transpose(1,0).contiguous(),
                torch.from_numpy(np.stack(b_last_act)).transpose(1,0).contiguous(),
                torch.from_numpy(b_action).unsqueeze(1),
                torch.from_numpy(b_reward).unsqueeze(1),
                torch.from_numpy(b_pred_d).unsqueeze(1),
                torch.from_numpy(b_gamma).unsqueeze(1),
                torch.ByteTensor(b_steps),

                torch.from_numpy(np.concatenate(b_hidden, axis=0)),
                torch.from_numpy(np.stack(b_relative_pos)),
                torch.from_numpy(np.stack(b_comm_mask)),

                idxes,
                torch.from_numpy(weights.astype(np.float16)).unsqueeze(1),
                self.ptr
            )

            return data

    def update_priorities(self, idxes: np.ndarray, priorities: np.ndarray, old_ptr: int):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the indices that already been discarded in replay buffer during training
            if self.ptr > old_ptr:
                # range from [old_ptr, self.ptr)
                mask = (idxes < old_ptr*self.chunk_capacity) | (idxes >= self.ptr*self.chunk_capacity)
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.ptr < old_ptr:
                # range from [0, self.ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr*self.chunk_capacity) & (idxes >= self.ptr*self.chunk_capacity)
                idxes = idxes[mask]
                priorities = priorities[mask]

            self.priority_tree.batch_update(np.copy(idxes), np.copy(priorities)**self.alpha)

    def stats(self, interval: int):
        '''
        Print log
        '''
        print('buffer update speed: {}/s'.format(self.counter/interval))
        print('buffer size: {}'.format(np.sum(self.size_buf)))
        print('buffer writer_cnt: {}'.format(np.sum(self.writer_cnt)))
        print()
        
        print('  ', end='')
        for i in range(config.init_env_settings[1], config.max_map_length+1, 5):
            print('   {:2d}   '.format(i), end='')
        print()

        for num_agents in range(config.init_env_settings[0], config.max_num_agents+1):
            print('{:2d}'.format(num_agents), end='')
            for map_len in range(config.init_env_settings[1], config.max_map_length+1, 5):
                if (num_agents, map_len) in self.stat_dict:
                    print('{:4d}/{:<3d}'.format(sum(self.stat_dict[(num_agents, map_len)]), len(self.stat_dict[(num_agents, map_len)])), end='')
                else:
                    print('   N/A  ', end='')
            print()

        for key, val in self.stat_dict.copy().items():
            # print('{}: {}/{}'.format(key, sum(val), len(val)))
            if len(val) == config.cl_history_size and sum(val) >= config.cl_history_size*config.pass_rate:
                # add number of agents
                add_agent_key = (key[0]+1, key[1]) 
                if add_agent_key[0] <= config.max_num_agents and add_agent_key not in self.stat_dict:
                    self.stat_dict[add_agent_key] = []
                
                if key[1] < config.max_map_length:
                    add_map_key = (key[0], key[1]+5) 
                    if add_map_key not in self.stat_dict:
                        self.stat_dict[add_map_key] = []
                
        self.env_settings_set = ray.put(list(self.stat_dict.keys()))

        self.counter = 0

        self.add_running_perf_status(self.writer_cnt)
        self.writer_cnt += 1

    def ready(self):
        if len(self) >= config.learning_starts:
            return True
        else:
            return False
    
    def get_env_settings(self):
        return self.env_settings_set

    def get_process_running_status(self):
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            pynvml.nvmlInit()
        current_process_util = psutil.Process(os.getpid())
        cpu_info = {
            'RAM (MB)': current_process_util.memory_info().rss / 1024**2,
            'CPU': {
                '# cores': current_process_util.cpu_num(),
                'usage (%)': current_process_util.cpu_percent()
            }
        }
        res = dict(cpu=cpu_info)
        
        if has_gpu:
            used_vram = 0.
            num_competing_processes = 0
            
            for di in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(di)
                running_processes_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for pi in running_processes_info:
                    pid = getattr(pi, 'pid')
                    if os.getpid() == pid:
                        used_vram += getattr(pi, 'usedGpuMemory')
                        num_competing_processes += len(running_processes_info) - 1
                        break
            
            used_vram = used_vram / 1024**2
            gpu_info = {
                'VRAM (MB)': used_vram,
                '# competing processes': num_competing_processes
            }
            res['gpu'] = gpu_info
        
        return res    

    def add_running_perf_status(self, global_step: int):
        sys_status = self.get_process_running_status()
        self.writer.add_scalar('running_perf/RAM', sys_status['cpu']['RAM (MB)'], global_step)
        self.writer.add_scalar('running_perf/CPU_core', sys_status['cpu']['CPU']['# cores'], global_step)
        self.writer.add_scalar('running_perf/CPU_usage', sys_status['cpu']['CPU']['usage (%)'], global_step)
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            self.writer.add_scalar('running_perf/VRAM', sys_status['gpu']['VRAM (MB)'], global_step)
            self.writer.add_scalar('running_perf/num_competing_processes', sys_status['gpu']['# competing processes'], global_step)
        
        self.writer.add_scalar('buffer_varf/priority_tree', asizeof(self.priority_tree), global_step)
        self.writer.add_scalar('buffer_varf/stat_dict', asizeof(self.stat_dict), global_step)
        self.writer.add_scalar('buffer_varf/batched_data', asizeof(self.batched_data), global_step)
        self.writer.add_scalar('buffer_varf/obs_buf', asizeof(self.obs_buf), global_step)
        self.writer.add_scalar('buffer_varf/urgency_buf', asizeof(self.urgency_buf), global_step)
        self.writer.add_scalar('buffer_varf/crowd_buf', asizeof(self.crowd_buf), global_step)
        self.writer.add_scalar('buffer_varf/last_act_buf', asizeof(self.last_act_buf), global_step)
        self.writer.add_scalar('buffer_varf/act_buf', asizeof(self.act_buf), global_step)
        self.writer.add_scalar('buffer_varf/rew_buf', asizeof(self.rew_buf), global_step)
        self.writer.add_scalar('buffer_varf/hid_buf', asizeof(self.hid_buf), global_step)
        self.writer.add_scalar('buffer_varf/size_buf', asizeof(self.size_buf), global_step)
        self.writer.add_scalar('buffer_varf/relative_pos_buf', asizeof(self.relative_pos_buf), global_step)
        self.writer.add_scalar('buffer_varf/comm_mask_buf', asizeof(self.comm_mask_buf), global_step)
        self.writer.add_scalar('buffer_varf/gamma_buf', asizeof(self.gamma_buf), global_step)
        self.writer.add_scalar('buffer_varf/num_agents_buf', asizeof(self.num_agents_buf), global_step)
        


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer: GlobalBuffer, expert_buffer: GlobalBuffer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelg = Network()
        self.modeld = Discriminator()
        self.modelg.to(self.device)
        self.modeld.to(self.device)
        self.tar_modelg = deepcopy(self.modelg)
        self.optimizer_g = Adam(self.modelg.parameters(), lr=config.gail_lr_g)
        self.optimizer_d = Adam(self.modeld.parameters(), lr=config.gail_lr_d)
        self.scheduler = MultiStepLR(self.optimizer_g, milestones=[40000, 80000], gamma=0.5)
        self.scheduler_d = MultiStepLR(self.optimizer_g, milestones=[40000, 80000], gamma=0.5)
        self.buffer = buffer
        self.expert_buffer = expert_buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.loss = 0
        self.lossd = 0

        self.data_list = []

        self.store_weights()
        self.save_path = config.save_path
        # self.save_path = config.save_path + '_{}'.format(get_formatted_time())
        # os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # gail frozen layers
        # self.preload_layers = ['obs_encoder', 'gru', 'comm', 'val', 'state']
        # self.frozen_layers = ['obs_encoder', 'gru', 'comm']
        # if config.gail_resume != '':
        #     print("Preparing to continue training Net-G and Net-D.")
        #     config.gail_start_iter = torch.load(os.path.join(config.gail_resume, 'netG.pth'))['epoch']
        #     self.modelg.load_state_dict(torch.load(os.path.join(config.gail_resume, 'netG.pth'))['state_dict'])
        #     self.modeld.load_state_dict(torch.load(os.path.join(config.gail_resume, 'netD.pth'))['state_dict'])
        # else:
        #     print("\nLoading pre-trained Encoder-Comm.")
        #     checkpoint = torch.load(config.checkpoint, map_location=self.device)
        #     state_dict = {key: value for key, value in checkpoint.items() if key.split('.')[0] in self.preload_layers}
        #     self.modelg.load_state_dict(state_dict=state_dict, strict=False)

        # for name, layer in self.modelg.named_modules():
        #     if name != '':
        #         print(f"Layer name: {name}")
        
        # freeze_module(self.modelg, self.frozen_layers)

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.modelg.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        self.learning_thread = threading.Thread(target=self._train, daemon=True)
        self.learning_thread.start()

    def _train(self):
        scaler = GradScaler()
        scaler_d = GradScaler()
        b_seq_len = torch.LongTensor(config.batch_size)
        b_seq_len[:] = config.burn_in_steps+1

        for i in range(1, config.training_steps+1):
            """
            data = (
                torch.from_numpy(np.stack(b_obs)).transpose(1,0).contiguous(),
                torch.from_numpy(np.stack(b_urgency)).transpose(1,0).contiguous(),
                torch.from_numpy(np.stack(b_crowd)).transpose(1,0).contiguous(),
                torch.from_numpy(np.stack(b_last_act)).transpose(1,0).contiguous(),
                torch.from_numpy(b_action).unsqueeze(1),
                torch.from_numpy(b_reward).unsqueeze(1),
                torch.from_numpy(b_gamma).unsqueeze(1),
                torch.ByteTensor(b_steps),

                torch.from_numpy(np.concatenate(b_hidden, axis=0)),
                torch.from_numpy(np.stack(b_relative_pos)),
                torch.from_numpy(np.stack(b_comm_mask)),

                idxes,
                torch.from_numpy(weights.astype(np.float16)).unsqueeze(1),
                self.ptr
            )
            """
            data_id = ray.get(self.buffer.get_batched_data.remote())
            data = ray.get(data_id)

            b_obs, b_urgency, b_crowd, b_last_act, b_action, b_reward, b_pred_d, b_gamma, b_steps, b_hidden, b_relative_pos, b_comm_mask, idxes, weights, old_ptr = data
            b_obs, b_urgency, b_crowd, b_last_act, b_action, b_reward, b_pred_d = b_obs.to(self.device), b_urgency.to(self.device), b_crowd.to(self.device), b_last_act.to(self.device), b_action.to(self.device), b_reward.to(self.device), b_pred_d.to(self.device)
            b_gamma, weights = b_gamma.to(self.device), weights.to(self.device)
            b_hidden = b_hidden.to(self.device)
            b_relative_pos, b_comm_mask = b_relative_pos.to(self.device), b_comm_mask.to(self.device)

            b_action = b_action.long()

            b_obs, b_urgency, b_crowd, b_last_act = b_obs.half(), b_urgency.half(), b_crowd.half(), b_last_act.half()

            b_next_seq_len = b_seq_len + b_steps
            
            with torch.no_grad():
                b_q_ = self.tar_modelg(b_obs, b_urgency, b_crowd, b_last_act, b_next_seq_len, b_hidden, b_relative_pos, b_comm_mask).max(1, keepdim=True)[0]

            target_q = b_reward + b_gamma * b_q_

            b_q = self.modelg(b_obs[:-config.forward_steps], b_urgency[:-config.forward_steps], b_crowd[:-config.forward_steps], b_last_act[:-config.forward_steps], b_seq_len, b_hidden, b_relative_pos[:, :-config.forward_steps], b_comm_mask[:, :-config.forward_steps]).gather(1, b_action)

            td_error = target_q - b_q

            priorities = td_error.detach().clone().squeeze().abs().clamp(1e-6).cpu().numpy()

            loss = F.mse_loss(b_q, target_q)
            self.loss += loss.item()

            self.optimizer_g.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer_g)
            nn.utils.clip_grad_norm_(self.modelg.parameters(), config.grad_norm_dqn)
            scaler.step(self.optimizer_g)
            scaler.update()
            
            self.scheduler.step()

            self.buffer.update_priorities.remote(idxes, priorities, old_ptr)

            # discriminator update
            bce_loss = nn.BCEWithLogitsLoss()
            action_one_hot = torch.nn.functional.one_hot(b_action.squeeze(), num_classes=config.action_dim)
            num_agents = int(b_hidden.shape[0] / 128)
            _b_hidden = b_hidden.view(config.batch_size, num_agents, config.hidden_dim)
            pred_fake = self.modeld(torch.cat([_b_hidden[:,0].clone().detach(), action_one_hot.clone().detach()], dim=1))
            # print('pred_fake', pred_fake)
            # print('b_pred_d', b_pred_d)
            loss_d_fake = bce_loss(pred_fake, b_pred_d)

            expert_data_id = ray.get(self.expert_buffer.get_batched_data.remote())
            expert_data = ray.get(expert_data_id)
            _, _, _, _, expert_action, _, expert_pred_d, _, _, expert_hidden, _, _, _, _, _ = expert_data
            expert_action, expert_pred_d, expert_hidden = expert_action.to(self.device), expert_pred_d.to(self.device), expert_hidden.to(self.device)
            expert_action = expert_action.long()

            action_one_hot = torch.nn.functional.one_hot(expert_action.squeeze(), num_classes=config.action_dim)
            num_agents = int(expert_hidden.shape[0] / 128)
            _expert_hidden = expert_hidden.view(config.batch_size, num_agents, config.hidden_dim)
            pred_real = self.modeld(torch.cat([_expert_hidden[:,0].clone().detach(), action_one_hot.clone().detach()], dim=1))
            
            loss_d_real = bce_loss(pred_real, expert_pred_d)
            # print('pred_real', pred_real)
            # print('expert_pred_d', expert_pred_d)

            loss_d = 0.5 * loss_d_real + 0.5 * loss_d_fake
            self.lossd += loss_d.item()
            self.optimizer_d.zero_grad()
            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(self.optimizer_d)
            nn.utils.clip_grad_norm_(self.modeld.parameters(), config.grad_norm_dqn)
            scaler_d.step(self.optimizer_d)
            scaler_d.update()
            self.scheduler_d.step()

            # store new weights in shared memory
            if i % 2  == 0:
                self.store_weights()

            self.counter += 1

            # update target net, save model
            if i % config.target_network_update_freq == 0:
                self.tar_modelg.load_state_dict(self.modelg.state_dict())
            if i % config.save_interval == 0:
                torch.save(self.modelg.state_dict(), os.path.join(config.gail_save_path, f'generator{self.counter}.pth'))
                torch.save(self.modeld.state_dict(), os.path.join(config.gail_save_path, f'discriminator{self.counter}.pth'))

        self.done = True

    def stats(self, interval: int):
        '''
        print log
        '''
        print('number of updates: {}'.format(self.counter))
        print('update speed: {}/s'.format((self.counter-self.last_counter)/interval))
        if self.counter != self.last_counter:
            print('loss_g: {:.4f}'.format(self.loss/(self.counter-self.last_counter)))
            print('loss_d: {:.4f}'.format(self.lossd/(self.counter-self.last_counter)))
        
        self.last_counter = self.counter
        self.loss = 0
        self.lossd = 0
        return self.done


@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, worker_id: int, epsilon: float, learner: Learner, buffer: GlobalBuffer, expert_buffer: GlobalBuffer):
        self.id = worker_id
        self.modelg = Network()
        self.modeld = Discriminator()
        self.modelg.eval()
        self.modeld.eval()
        self.env = Environment(curriculum=True)
        self.expert_env = Environment(curriculum=True)
        self.epsilon = epsilon
        self.learner = learner
        self.global_buffer = buffer
        self.global_expert_buffer = expert_buffer
        self.max_episode_length = config.max_episode_length
        self.counter = 0

        self.loaded_dataset = None

    def run(self):
        done = False
        obs, urgency, crowd, last_act, pos, pick_expert_data, local_buffer, expert_local_buffer = self._reset()

        while True:

            # sample action
            actions, q_val, hidden, relative_pos, comm_mask = self.modelg.step(torch.from_numpy(obs.astype(np.float32)), 
                                                                            torch.from_numpy(urgency.astype(np.float32)),
                                                                            torch.from_numpy(crowd.astype(np.float32)),
                                                                            torch.from_numpy(last_act.astype(np.float32)), 
                                                                            torch.from_numpy(pos.astype(np.int)))
            
            # epsilon-greedy
            if random.random() < self.epsilon:
                # Note: only one agent do random action in order to keep the environment stable
                actions[0] = np.random.randint(0, config.action_dim)

            # take action in env
            (next_obs, next_urgency, next_crowd, last_act, next_pos), rewards, done, _ = self.env.step(actions)

            # return data and update observation
            with torch.no_grad():
                actions_one_hot = np.zeros((self.env.num_agents, 5), dtype=np.bool)
                actions_one_hot[np.arange(self.env.num_agents), np.array(actions)] = 1
                # print(torch.tensor(hidden, dtype=torch.float32).shape)
                # # print(torch.tensor(actions_one_hot[0], dtype=torch.float32).shape)
                # print(torch.tensor(actions_one_hot, dtype=torch.float32).shape)
                hidden_t = torch.tensor(hidden, dtype=torch.float32)
                actions_one_hot_t = torch.tensor(actions_one_hot, dtype=torch.float32)
                if hidden_t.shape[0] != actions_one_hot_t.shape[0]:
                    pred_d = self.modeld(torch.cat([hidden_t, actions_one_hot_t[0]]))
                elif hidden_t.shape[0] == actions_one_hot_t.shape[0]:
                    pred_d = self.modeld(torch.cat([hidden_t[0], actions_one_hot_t[0]]))
                # pred_d = self.modeld(torch.cat([torch.tensor(hidden, dtype=torch.float32), torch.tensor(actions_one_hot[0], dtype=torch.float32)]))
                # raise NotImplementedError("hidden: {}\nactions_one_hot: {}".format(torch.tensor(hidden, dtype=torch.float32).shape, torch.tensor(actions_one_hot, dtype=torch.float32).shape))
                reward_d = pred_d * 0.15 - 0.075    # 映射成奖励值
            
            pred_d_fake = torch.tensor([0.0])
            local_buffer.add(q_val[0], actions[0], last_act, rewards[0] + reward_d, pred_d_fake, next_obs, next_urgency, next_crowd, hidden, relative_pos, comm_mask)

            if done == False and self.env.steps < self.max_episode_length:
                obs, pos, urgency, crowd = next_obs, next_pos, next_urgency, next_crowd
            else:
                # finish and send buffer
                if done:
                    data = local_buffer.finish()
                else:
                    _, q_val, _, relative_pos, comm_mask = self.modelg.step(torch.from_numpy(next_obs.astype(np.float32)), 
                                                                            torch.from_numpy(urgency.astype(np.float32)), 
                                                                            torch.from_numpy(crowd.astype(np.float32)), 
                                                                            torch.from_numpy(last_act.astype(np.float32)), 
                                                                            torch.from_numpy(next_pos.astype(np.int)))
                    data = local_buffer.finish(q_val[0], relative_pos, comm_mask)

                # generate expert tau
                # expert act
                expert_obs, expert_urgency, exper_crowd, expert_last_act, expert_pos = self.expert_env.observe()
                expert_actions = pick_expert_data['action_list']
                for expert_action in expert_actions:
                    _, expert_q_val, expert_hidden, expert_relative_pos, expert_comm_mask = self.modelg.step(torch.from_numpy(expert_obs.astype(np.float32)), 
                                                                                                               torch.from_numpy(expert_urgency.astype(np.float32)), 
                                                                            torch.from_numpy(exper_crowd.astype(np.float32)), 
                                                                            torch.from_numpy(expert_last_act.astype(np.float32)), 
                                                                            torch.from_numpy(expert_pos.astype(np.int)))
                    (expert_next_obs, expert_next_urgency, exper_next_crowd, expert_last_act, expert_next_pos), expert_rewards, done, _ = self.expert_env.step(expert_action)
                    pred_d_real = torch.tensor([1.0])

                    expert_local_buffer.add(expert_q_val[0], expert_action[0], expert_last_act, expert_rewards[0], pred_d_real, expert_next_obs, expert_next_urgency, exper_next_crowd, expert_hidden, expert_relative_pos, expert_comm_mask)
                    expert_obs, expert_pos, expert_urgency, exper_crowd = expert_next_obs, expert_next_pos, expert_next_urgency, exper_next_crowd
                expert_data = expert_local_buffer.finish()

                self.global_buffer.add.remote(data)
                self.global_expert_buffer.add.remote(expert_data)
                
                done = False
                obs, urgency, crowd, last_act, pos, pick_expert_data, local_buffer, expert_local_buffer = self._reset()

            self.counter += 1
            if self.counter == config.actor_update_steps:
                self._update_weights()
                self.counter = 0

    def _update_weights(self):
        '''load weights from learner'''
        # update network parameters
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.modelg.load_state_dict(weights)

        # update environment settings set (number of agents and map size)
        new_env_settings_set = ray.get(self.global_buffer.get_env_settings.remote())
        self.env.update_env_settings_set(ray.get(new_env_settings_set))
    
    def _reset(self):
        self.modelg.reset()

        self.env.reset()

        # with open('data/dataset_{}numagents_{}maplen_10000.pkl'.format(self.env.num_agents, self.env.map_size[0]), 'rb') as f:
        with open('data/dataset_{}numagents_{}maplen.pkl'.format(self.env.num_agents, self.env.map_size[0]), 'rb') as f:
            self.loaded_dataset = pickle.load(f)
        
        pick_expert_data = self.loaded_dataset[random.randint(0,len(self.loaded_dataset)-1)]
        self.expert_env.load(pick_expert_data['env'][0], pick_expert_data['env'][1], pick_expert_data['env'][2])

        init_obs, init_urgency, init_crowd, last_act, pos = self.env.observe()
        expert_init_obs, expert_urgency, exper_crowd, _, _ = self.expert_env.observe()

        local_buffer = LocalBuffer(self.id, self.env.num_agents, self.env.map_size[0], init_obs, init_urgency, init_crowd)
        expert_local_buffer = LocalBuffer(self.id, self.expert_env.num_agents, self.expert_env.map_size[0], expert_init_obs, expert_urgency, exper_crowd)
        return init_obs, init_urgency, init_crowd, last_act, pos, pick_expert_data, local_buffer, expert_local_buffer
    