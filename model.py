from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import config

# Attention + GRU
class MessageAggregation(nn.Module):
    def __init__(self, input_dim=config.hidden_dim, message_dim=32, pos_embed_dim=16, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.message_dim = message_dim
        self.pos_embed_dim = pos_embed_dim
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(input_dim) # 归一化

        self.position_embeddings = nn.Linear((2*config.obs_radius+1)**2, pos_embed_dim) # 9*9->16

        # multi-head attention
        self.message_key = nn.Linear(input_dim+pos_embed_dim, message_dim * num_heads)  # 256+16->32*4
        self.message_value = nn.Linear(input_dim+pos_embed_dim, message_dim * num_heads)    # 256+16->32*4
        self.hidden_query = nn.Linear(input_dim, message_dim * num_heads)   # 256->32*4

        self.head_agg = nn.Linear(message_dim * num_heads, message_dim * num_heads) # 32*4->32*4

        self.update = nn.GRUCell(num_heads * message_dim, input_dim)  # input_size:32*4,hidden_size:input_dim->hidden_size:input_dim
    
    def forward(self, hidden, relative_pos, comm_mask):
        batch_size, num_agents, hidden_dim = hidden.size()
        attn_mask = (comm_mask==False).unsqueeze(3).unsqueeze(4)
        relative_pos = relative_pos.clone()

        batch_size, num_agents, _, _ = relative_pos.size()
        # mask out out of FOV agent
        relative_pos[(relative_pos.abs() > config.obs_radius).any(3)] = 0

        one_hot_position = torch.zeros((batch_size*num_agents*num_agents, 9*9), dtype=hidden.dtype, device=hidden.device)
        relative_pos += config.obs_radius
        relative_pos = relative_pos.reshape(batch_size*num_agents*num_agents, 2)
        relative_pos_idx = relative_pos[:, 0] + relative_pos[:, 1]*9
        one_hot_position[torch.arange(batch_size*num_agents*num_agents), relative_pos_idx.long()] = 1

        input = hidden

        hidden = self.norm(hidden)
        position_embedding = self.position_embeddings(one_hot_position)

        message_input = hidden.repeat_interleave(num_agents, dim=1).view(batch_size*num_agents*num_agents, hidden_dim)
        message_input = torch.cat((message_input, position_embedding), dim=1)
        message_input = message_input.view(batch_size, num_agents, num_agents, self.input_dim+self.pos_embed_dim)

        hidden_q = self.hidden_query(hidden).view(batch_size, 1, num_agents, self.num_heads, self.message_dim) # batch_size x num_agents x message_dim*num_heads
        message_k = self.message_key(message_input).view(batch_size, num_agents, num_agents, self.num_heads, self.message_dim)
        message_v = self.message_value(message_input).view(batch_size, num_agents, num_agents, self.num_heads, self.message_dim)

        # attention
        attn_score = (hidden_q * message_k).sum(4, keepdim=True) / self.message_dim**0.5 # batch_size x num_agents x num_agents x self.num_heads x 1
        attn_score.masked_fill_(attn_mask, torch.finfo(attn_score.dtype).min)
        attn_weights = F.softmax(attn_score, dim=1)

        # agg
        agg_message = (message_v * attn_weights).sum(1).view(batch_size, num_agents, self.num_heads*self.message_dim)
        agg_message = self.head_agg(agg_message)

        # update hidden with request message 
        input = input.view(-1, hidden_dim)
        agg_message = agg_message.view(batch_size*num_agents, self.num_heads*self.message_dim)

        updated_hidden = self.update(agg_message, input)

        # some agents may not receive message, keep it as original
        update_mask = comm_mask.any(1).view(-1, 1)
        hidden = torch.where(update_mask, updated_hidden, input)
        hidden = hidden.view(batch_size, num_agents, hidden_dim)

        return hidden

class CommBlock(nn.Module):
    def __init__(self, hidden_dim=config.hidden_dim, message_dim=128, pos_embed_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.pos_embed_dim = pos_embed_dim

        self.request_comm = MessageAggregation()
        self.reply_comm = MessageAggregation()

    def forward(self, latent, relative_pos, comm_mask):
        '''
        latent shape: batch_size x num_agents x latent_dim
        relative_pos shape: batch_size x num_agents x num_agents x 2
        comm_mask shape: batch_size x num_agents x num_agents
        '''
        batch_size, num_agents, latent_dim = latent.size()

        assert relative_pos.size() == (batch_size, num_agents, num_agents, 2), relative_pos.size()
        assert comm_mask.size() == (batch_size, num_agents, num_agents), comm_mask.size()

        if torch.sum(comm_mask).item() == 0:
            return latent

        hidden = self.request_comm(latent, relative_pos, comm_mask)

        comm_mask = torch.transpose(comm_mask, 1, 2)

        hidden = self.reply_comm(hidden, relative_pos, comm_mask)

        return hidden

class Network(nn.Module):
    def __init__(self, input_shape=config.obs_shape, comm=config.comm, long_range_comm = config.long_range_comm):
        super().__init__()
        
        self.obs_shape = input_shape    # (6,9,9)
        self.obs_embedding_dim = config.obs_embedding_dim   # 252
        self.urgency_embedding_dim = config.urgency_embedding_dim   # 6
        self.hidden_dim = config.hidden_dim # 256
        # self.embedding_dim = self.obs_embedding_dim + 2 + self.urgency_embedding_dim    # 256
        self.embedding_dim = self.obs_embedding_dim + 2 + 2    # 256
        self.latent_dim = self.embedding_dim + 5   # 256+5
        
        self.comm = comm
        self.long_range_comm = long_range_comm

        self.hidden = None
        
        # encode
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 192, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(192, 252, 3, 1),  # (252,1,1)
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
        )
        # self.urgency_encoder = nn.Sequential(
        #     nn.Linear(2, self.urgency_embedding_dim),
        #     # nn.Linear(12, self.urgency_embedding_dim),
        #     nn.Tanh(),
        # )

        # gru
        self.gru = nn.GRUCell(self.latent_dim, self.hidden_dim)

        # communication
        self.comm = CommBlock(self.embedding_dim)

        # dueling q network
        self.adv = nn.Linear(self.hidden_dim, 5)
        self.state = nn.Linear(self.hidden_dim, 1)
    
    @autocast()
    def forward(self, obs, urgency, crowd, last_act, steps, hidden, relative_pos, comm_mask):
        # obs shape: seq_len, batch_size, num_agents, obs_shape
        # relative_pos shape: batch_size, seq_len, num_agents, num_agents, 2
        # comm_mask shape: batch_size, seq_len, 2, num_agents, num_agents
        # prepare data
        seq_len, batch_size, num_agents, *_ = obs.size()
        obs = obs.view(seq_len*batch_size*num_agents, *self.obs_shape)
        last_act = last_act.view(seq_len*batch_size*num_agents, config.action_dim)

        # encode
        embedding = self.obs_encoder(obs)
        # urgency_embedding = self.urgency_encoder(urgency)
        urgency_embedding = urgency
        # urgency_embedding = urgency_embedding.view(seq_len*batch_size*num_agents, config.urgency_embedding_dim)
        urgency_embedding = urgency_embedding.view(seq_len*batch_size*num_agents, 2)
        embedding = torch.cat((embedding, urgency_embedding), dim=1)
        crowd = crowd.view(seq_len*batch_size*num_agents, 2)
        embedding = torch.cat((embedding, crowd), dim=1)

        # gru
        latent = torch.cat((embedding, last_act), dim=1)    # 256+5
        latent = latent.view(seq_len, batch_size * num_agents, self.latent_dim)

        # comm_mask = comm_mask.view(2, batch_size, seq_len, num_agents, num_agents)
        hidden_buffer = []
        for i in range(seq_len):
            # embedding size: batch_size * num_agents x self.embedding_dim
            embedding = self.gru(latent[i], hidden)
            embedding = embedding.view(batch_size, num_agents, self.hidden_dim)

            # communicate with comm_mask agents
            _comm_mask = comm_mask[:, i]
            comm_mask_or = torch.bitwise_or(_comm_mask[:, 0], _comm_mask[:, 1])
            hidden = self.comm(embedding, relative_pos[:, i], comm_mask_or)

            # only hidden from agent 0
            hidden_buffer.append(hidden[:, 0])
            hidden = hidden.view(batch_size*num_agents, self.hidden_dim)

        # hidden buffer size: batch_size x seq_len x self.hidden_dim
        hidden_buffer = torch.stack(hidden_buffer).transpose(0, 1)

        # hidden size: batch_size x self.hidden_dim
        hidden = hidden_buffer[torch.arange(config.batch_size), steps-1]

        # dueling q network
        adv_val = self.adv(hidden)
        state_val = self.state(hidden)
        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val

    # inference
    @torch.no_grad()
    def step(self, obs, urgency, crowd, last_act, pos):
        num_agents = obs.size(0)
        agent_indexing = torch.arange(num_agents)
        relative_pos = pos.unsqueeze(0)-pos.unsqueeze(1)    # (num,num,2)，relative position
        
        # 观测范围内 mask
        in_obs_mask = (relative_pos.abs() <= config.obs_radius).all(2)
        in_obs_mask[agent_indexing, agent_indexing] = 0     # (num,num)，neighbor relationship mask

        if self.long_range_comm:    # long range communication
            """ first: selective one-hop communication  """
            test_mask = in_obs_mask.clone()
            test_mask[agent_indexing, agent_indexing] = 1
            num_in_obs_agents = test_mask.sum(1)    # eg:[1,1,1,1,1,3,3,3],nums=8
            
            # control communicaiton unit
            test_obs = torch.repeat_interleave(obs, num_agents, dim=0).view(num_agents, num_agents, *config.obs_shape)[test_mask]   # 需要交流的智能体的观测值
            test_relative_pos = relative_pos[test_mask]
            test_relative_pos += config.obs_radius
            test_obs[torch.arange(num_in_obs_agents.sum()), 0, test_relative_pos[:, 0], test_relative_pos[:, 1]] = 0    # o i,-j
            test_last_act = torch.repeat_interleave(last_act, num_in_obs_agents, dim=0)
            test_urgency = torch.repeat_interleave(urgency, num_in_obs_agents, dim=0)
            test_crowd = torch.repeat_interleave(crowd, num_in_obs_agents, dim=0)

            if self.hidden is None:
                test_hidden = torch.zeros((num_in_obs_agents.sum(), self.embedding_dim))
            else:
                test_hidden = torch.repeat_interleave(self.hidden, num_in_obs_agents, dim=0)

            test_latent = self.obs_encoder(test_obs)
            # urgency_embedding = self.urgency_encoder(test_urgency)
            urgency_embedding = test_urgency
            test_latent = torch.cat((test_latent, urgency_embedding), dim=1)
            test_latent = torch.cat((test_latent, test_crowd), dim=1)
            test_latent = torch.cat((test_latent, test_last_act), dim=1)
            test_hidden = self.gru(test_latent, test_hidden)

            origin_agent_idx = torch.zeros(num_agents, dtype=torch.long)
            for i in range(num_agents-1):
                origin_agent_idx[i+1] = test_mask[i, i:].sum() + test_mask[i+1, :i+1].sum() + origin_agent_idx[i]
            self.hidden = test_hidden[origin_agent_idx]
            
            # compare
            adv_val = self.adv(test_hidden)
            state_val = self.state(test_hidden)
            test_q_val = (state_val + adv_val - adv_val.mean(1, keepdim=True))
            test_actions = torch.argmax(test_q_val, 1)
            actions_mat = torch.ones((num_agents, num_agents), dtype=test_actions.dtype) * -1
            actions_mat[test_mask] = test_actions
            diff_action_mask = actions_mat != actions_mat[agent_indexing, agent_indexing].unsqueeze(1)

            assert (in_obs_mask[agent_indexing, agent_indexing] == 0).all()
            
            one_comm_mask = torch.bitwise_and(in_obs_mask, diff_action_mask)
            assert (one_comm_mask[agent_indexing, agent_indexing] == 0).all()
            self.hidden = self.comm(self.hidden.unsqueeze(0), relative_pos.unsqueeze(0), one_comm_mask.unsqueeze(0))
            
            """ secondly: selective two-hop communication  """
            two_hop_obs_mask = torch.matmul(in_obs_mask.int(), in_obs_mask.int())
            two_hop_obs_mask[agent_indexing, agent_indexing] = 0    # exclude itself
            two_hop_obs_mask[in_obs_mask == 1] = 0  # exclude one-hop neighbors
            two_hop_obs_mask = two_hop_obs_mask.bool()
            hidden2 = self.comm(self.hidden, relative_pos.unsqueeze(0), two_hop_obs_mask.unsqueeze(0))

            self.hidden = self.hidden.squeeze(0)
            hidden2 = hidden2.squeeze(0)

            onehop_adv_val = self.adv(self.hidden)
            onehop_state_val = self.state(self.hidden)
            onehop_test_q_val = (onehop_state_val + onehop_adv_val - onehop_adv_val.mean(1, keepdim=True))
            onehop_test_actions = torch.argmax(onehop_test_q_val, 1)

            twohop_adv_val = self.adv(hidden2)
            twohop_state_val = self.state(hidden2)
            twohop_test_q_val = (twohop_state_val + twohop_adv_val - twohop_adv_val.mean(1, keepdim=True))
            twohop_test_actions = torch.argmax(twohop_test_q_val, 1)

            # compare
            diff_action_agent = (onehop_test_actions != twohop_test_actions).int()
            diff_action_mask2 = torch.transpose(torch.repeat_interleave(diff_action_agent.unsqueeze(0), num_agents, dim=0), 0, 1)

            two_comm_mask = torch.bitwise_and(two_hop_obs_mask, diff_action_mask2)
            
            assert (two_comm_mask[agent_indexing, agent_indexing] == 0).all()

            # if not (diff_action_agent == 0).all():
            #     self.hidden = hidden2
            #     print(self.hidden)

        else:
            embedding = self.obs_encoder(obs)   # num_agents, config.obs_shape
            # urgency_embedding = self.urgency_encoder(urgency)
            urgency_embedding = urgency
            embedding = torch.cat((embedding, urgency_embedding), dim=1)
            embedding = torch.cat((embedding, crowd), dim=1)
            latent = torch.cat((embedding, last_act), dim=1)
            
            if self.comm:   # short range communication
                # mask out agents that are far away
                dist_mat = (relative_pos[:, :, 0]**2 + relative_pos[:, :, 1]**2)    # Relative distance x2+y2
                # at most communicate with (max_comm_agents-1) agents
                _, ranking = dist_mat.topk(min(config.max_comm_agents, num_agents), dim=1, largest=False)
                dist_mask = torch.zeros((num_agents, num_agents), dtype=torch.bool)
                dist_mask.scatter_(1, ranking, True)
                
                # communication mask
                comm_mask = torch.bitwise_and(in_obs_mask, dist_mask)
            else:
                comm_mask[torch.arange(num_agents), torch.arange(num_agents)] = 0
            
            if self.hidden is None:
                self.hidden = self.gru(latent)
            else:
                self.hidden = self.gru(latent, self.hidden)
            
            one_comm_mask, two_comm_mask = comm_mask, comm_mask
            assert (one_comm_mask[agent_indexing, agent_indexing] == 0).all()
            assert (two_comm_mask[agent_indexing, agent_indexing] == 0).all()
        
        # dueling q network
        adv_val = self.adv(self.hidden)
        state_val = self.state(self.hidden)
        q_val = (state_val + adv_val - adv_val.mean(1, keepdim=True))

        actions = torch.argmax(q_val, 1).tolist()
        
        comm_mask_or = torch.bitwise_or(one_comm_mask, two_comm_mask)
        self.hidden = self.comm(self.hidden.unsqueeze(0), relative_pos.unsqueeze(0), comm_mask_or.unsqueeze(0))
        self.hidden = self.hidden.squeeze(0)
        
        return actions, q_val.numpy(), self.hidden.squeeze(0).numpy(), relative_pos.numpy(), np.array((one_comm_mask.numpy(), two_comm_mask.numpy()))
    
    def reset(self):
        self.hidden = None


class Discriminator(nn.Module):
    def __init__(self, num_inputs=256+5, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.logic = nn.Linear(last_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    @autocast()
    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        prob = torch.sigmoid(self.logic(x))
        return prob
