from utils import get_formatted_time
formatted_time = get_formatted_time()
############################################################
####################    environment     ####################
############################################################

obs_radius = 4
crowd_per_obs_radius = 2
reward_fn = dict(move=-0.075,
                stay_on_goal=0,
                stay_off_goal=-0.075,
                collision=-0.55,
                finish=3)

obs_shape = (6, 2*obs_radius+1, 2*obs_radius+1)
action_dim = 5

############################################################
####################         DQN        ####################
############################################################

# basic training setting
num_actors = 16
log_interval = 10
training_steps = 100000
save_interval = 5000
gamma = 0.99
batch_size = 128
learning_starts = 50000
target_network_update_freq = 1750
save_path='./saved_models'
max_episode_length = 256

# from_saved_model = False
# saved_model = '1'

buffer_capacity = 262144
chunk_capacity = 64
burn_in_steps = 20

actor_update_steps = 200

# gradient norm clipping
grad_norm_dqn=40

# n-step forward
forward_steps = 2

# prioritized replay
prioritized_replay_alpha=0.6
prioritized_replay_beta=0.4

# curriculum learning
init_env_settings = (1, 10)
max_num_agents = 20
max_map_length = 30
pass_rate = 0.80

# dqn network setting
cnn_channel = 128
obs_embedding_dim = 252
urgency_embedding_dim = 6
hidden_dim = 256

# communication settings
comm = True
long_range_comm = True
# only works when short range communicate
max_comm_agents = 3

# crowd threshold
crowd_threshold = 0.7

# curriculum learning
cl_history_size = 100

test_seed = 0
num_test_cases = 200
test_env_settings = (
                    (20, 8, 0.0), (20, 8, 0.1), (20, 8, 0.2), (20, 8, 0.3),
                    (20, 16, 0.0), (20, 16, 0.1), (20, 16, 0.2), (20, 16, 0.3),
                    (20, 32, 0.0), (20, 32, 0.1), (20, 32, 0.2), (20, 32, 0.3),
                    (20, 64, 0.0), (20, 64, 0.1), (20, 64, 0.2), (20, 64, 0.3)
                    ) # map length, number of agents, density

############################################################
####################         GAIL        ###################
############################################################

# gail_resume = ''
gail_save_path = './saved_models'
gail_test_path = './saved_models'
gail_start_iter = 0
gail_train_n_iter = 10000
gail_lr_d = 2e-4
gail_lr_g = 2e-4
print_freq = 250
save_freq = 1000
