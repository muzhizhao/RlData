# IMPORTS
import numpy.random
from NeuralNetwork import Agent
from ReplayMemory import *
import gym
import numpy as np
import time
import sys
import math
import os
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
import dogfight_client as df
from PPO import *

max_ep_len = 10000
update_timestep = max_ep_len * 4  # update policy every n timesteps
K_epochs = 80  # update policy for K epochs in one PPO update
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network

random_seed = 0  # set random seed if required (0 = no random seed)
has_continuous_action_space = True  # continuous action space; else discrete

max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)  # save model frequency (in num timesteps)

action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.005  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
state_dim = 11
action_dim =3
envName = 'Harfang_GYM'
name = "Harfang_GYM"

from HarfangEnv_GYM import *

##################################################################################
########################   网络模式设置
###################################################################################
checkpoint_path = 'G:\\RL\\Gym_Env_Tekli\\NNs\\model.pth'
df.connect("192.168.2.12", 50888)
planes = df.get_planes_list()
print(planes)
Plane_id_ally = planes[0]
Plane_id_oppo = planes[2]
df.reset_machine(Plane_id_ally)
df.reset_machine(Plane_id_oppo)
df.set_plane_thrust(Plane_id_ally, 1)
df.set_plane_thrust(Plane_id_oppo, 1)
df.set_client_update_mode(True)

# STARTING TIME
start = time.time()

df.disable_log()

##################################################################################
########################   训练次数设置
###################################################################################
trainingEpisodes = 10000
validationEpisodes = 1  #1
explorationEpisodes = 0
maxStep = 1000

Test = False
df.set_renderless_mode(False)  # 打开为True
df.disable_log()

bufferSize = (10 ** 6)
gamma = 0.99
criticLR = 1e-3
actorLR = 1e-3
tau = 0.005
checkpointRate = 100
SEED = 1996
highScore_ally = -math.inf
highScore_oppo = -math.inf
batchSize = 128  #

hiddenLayer1 = 128
hiddenLayer2 = 256
stateDim = 11
actionDim = 3
useLayerNorm = True
usePER = True


checkpointRate = 100
SEED = 1996
hiddenLayer1 = 128
hiddenLayer2 = 256
stateDim = 11
actionDim = 3
useLayerNorm = True
usePER = True
envName = 'Harfang_GYM'
name = "Harfang_GYM"

# INITIALIZATION
env = HarfangEnv()
env.seed(SEED)
hyperparams = {
    "lrvalue": 0.0005,
    "lrpolicy": 0.0001,
    "gamma": 0.9,
    "episodes": 150000,
    "buffersize": 500000,
    "tau": 0.001,
    "batchsize": 128,
    "alpha": 0.2,
    "maxlength": 10000,
    "hidden": 256,
}
HyperParams = namedtuple("HyperParams", hyperparams.keys())
hyprm = HyperParams(**hyperparams)

agent_ally = PPO(stateDim, actionDim,lr_actor, lr_critic,gamma,K_epochs,eps_clip,has_continuous_action_space,action_std)



writer = SummaryWriter(log_dir="runs/Harfang_GYM/" + name)
arttir = 0

if not Test:
    # RANDOM EXPLORATION
    print("Exploration Started")
    agent_ally.load(checkpoint_path)
    for episode in range(explorationEpisodes):
        df.activate_IA(Plane_id_ally)
        df.deactivate_IA(Plane_id_oppo)
        state_ally, state_oppo = env.reset()
        done = False
        for step in range(maxStep):
            if not done:
                #action_ally = env.action_space.sample()
                action_ally = env.get_action(Plane_id_ally)
                action_oppo = np.array([0,0,0])
                n_state_ally, reward_ally, done, info = env.step(action_ally)
                if step is maxStep - 1:
                    done = True
                agent_ally.update()
                state_ally = n_state_ally
        #sys.stdout.write("\rExploration Completed: %.2f%%\n" % ((episode + 1) / explorationEpisodes * 100))
    #sys.stdout.write("\n")
    agent_ally.save(checkpoint_path)
    agent_ally.load(checkpoint_path)
    # TRAINING  训练
    print("Training Started")
    scores = []
    # env.rst()
    scores_ally = []# 记录所有回合的奖励
    scores_oppo = []
    ma_rewards = [] # 记录所有回合的滑动平均奖励

    for episode in range(trainingEpisodes):  # 1000回合
        state_ally, state_oppo = env.reset()
        df.deactivate_IA(Plane_id_ally)
        df.deactivate_IA(Plane_id_oppo)
        # env.render()
        totalReward_ally = 0
        totalReward_oppo = 0
        done = False
        for step in range(maxStep):
            if not done:
                action_ally = agent_ally.select_action(state_ally)
                #action_ally = env.get_action(Plane_id_ally)
                action_oppo = np.array([0, 0, 0])
                #action_oppo = action_ally
                n_state_ally, reward_ally, done, info = env.step(action_ally)
                if step is maxStep - 1:
                    done = True
                agent_ally.update()
                state_ally = n_state_ally
                totalReward_ally += reward_ally
        scores_ally.append(totalReward_ally)

        now = time.time()
        seconds = int((now - start) % 60)
        minutes = int(((now - start) // 60) % 60)
        hours = int((now - start) // 3600)
        print('Episode: ', episode + 1, ' Completed: %r' % done, \
              "Steps：", step,
              ' FinalReward_ally: %.2f' % totalReward_ally, \
              # ' FinalReward_oppo: %.2f' % totalReward_oppo, \
              ' Last100AverageReward_ally: %.2f' % np.mean(scores_ally[-100:]), \
              # ' Last100AverageReward_oppo: %.2f' % np.mean(scores_oppo[-100:]), \
              'RunTime: ', hours, ':', minutes, ':', seconds)

        # VALIDATION

        if (((episode + 1) % checkpointRate) == 0):
            valScores_ally = []
            valScores_oppo = []
            for e in range(validationEpisodes):
                state_ally, state_oppo = env.reset()
                totalReward_ally = 0
                totalReward_oppo = 0
                done = False
                for step in range(maxStep):
                    while not done:
                        action_ally = agent_ally.select_action(state_ally)
                        # action_oppo = [0,0,0]
                        n_state_ally, reward_ally, done, info = env.step(action_ally)
                        if step is maxStep - 1:
                            done = True
                        state_ally = n_state_ally
                        # state_oppo = n_state_oppo
                        totalReward_ally += reward_ally
                        # totalReward_oppo += reward_oppo
                valScores_ally.append(totalReward_ally)
                # valScores_oppo.append(totalReward_oppo)

            highScore_ally = mean(valScores_ally)
            arttir += 1
            print('Validation Episode: ', (episode // checkpointRate) + 1, ' Average Reward_ally:',
                  highScore_ally )
            writer.add_scalar('Validation Reward', highScore_ally, episode)
            # writer.add_scalar('Validation Reward', mean(valScores_oppo), episode)

