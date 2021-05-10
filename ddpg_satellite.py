import random
import time
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn

MAX_EPISODES = 300
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
N_HIDDEN_UNIT = 30
RENDER = False
# -----------   declear environment   ----------------------

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
# -----------   infomation of env   --------------------
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_max = env.action_space.high
a_min = env.action_space.low

# ---------   declear ddpg   -----------
class DDPG(object):
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        # actor net
        self.actor_eval = self._build_actor()
        self.actor_target = self._build_actor()
        self.actor_eval_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_A)
        # eval net
        self.critic_eval = self._build_critic()
        self.critic_target = self._build_critic()
        self.critic_eval_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_C)
        # copy target net parameter to eval net
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())

    def _build_critic(self):
        net = nn.Sequential(
            nn.Linear(s_dim + a_dim, N_HIDDEN_UNIT),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
        return net

    def _build_actor(self):
        net = nn.Sequential(
            nn.Linear(s_dim, N_HIDDEN_UNIT),
            nn.ReLU(),
            nn.Linear(N_HIDDEN_UNIT, a_dim),
            nn.Tanh()
        )
        return net

    def choose_action(self, s):
        s = (torch.FloatTensor(s))
        action = self.actor_eval(s).data.numpy() * a_max
        return np.clip(action, a_min, a_max)

    def learn(self):
        sample_index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        b_memory = np.array([self.memory[x] for x in sample_index])
        b_s = torch.FloatTensor(b_memory[:, :s_dim])
        b_a = torch.FloatTensor(b_memory[:, s_dim:s_dim + a_dim])
        b_r = torch.FloatTensor(b_memory[:, -s_dim - 1: -s_dim])
        b_s_ = torch.FloatTensor(b_memory[:, -s_dim:])

        self.actor_learn(b_s)
        self.critic_learn(b_s, b_a, b_r, b_s_)
        self.soft_replace()

    def actor_learn(self,b_s):
        a = self.actor_eval.forward(b_s)  # 进行actor_eval的更新
        ce_s = torch.cat([b_s, a], 1)
        q = self.critic_eval.forward(ce_s)
        a_loss = torch.mean(-q)

        self.actor_eval_optim.zero_grad()
        a_loss.backward()
        self.actor_eval_optim.step()

    def critic_learn(self,b_s,b_a,b_r,b_s_):
        ce_s = torch.cat([b_s,b_a], 1)
        q = self.critic_eval.forward(ce_s)  # 进行critic_eval的更新
        a_ = self.actor_target.forward(b_s_).detach()
        ct_s = torch.cat([b_s_,a_],1)
        q_ = self.critic_target.forward(ct_s).detach()
        q_target = b_r + GAMMA * q_
        loss_func = nn.MSELoss()
        td_error = loss_func(q_target, q)

        self.critic_eval_optim.zero_grad()
        td_error.backward()
        self.critic_eval_optim.step()

    def soft_replace(self):  # 缓慢替换target网络
        for eval_param, target_param in zip(self.actor_eval.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + eval_param.data * TAU)
        for eval_param, target_param in zip(self.critic_eval.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + eval_param.data * TAU)

    def store_transition(self, s, a, r, s_):  # 记忆库存储函数
        record = np.hstack((s, a, r, s_))
        self.memory.append(record)


def main():
    ddpg = DDPG()
    RENDER = False
    var = 3  # control exploration
    for i in range(MAX_EPISODES):
        s = env.reset()  # 重置状态
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            a = ddpg.choose_action(s)  # 选择动作
            a = np.clip(np.random.normal(a, var), -2, 2)    # 添加噪音
            s_, r, done, info = env.step(a)  # 仿真
            ddpg.store_transition(s, a, r, s_)  # 存储记忆库

            if len(ddpg.memory) > MEMORY_CAPACITY - 1:  #学习并减少噪音
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_  # 更新状态
            ep_reward += r

            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -300 : RENDER = True
                break


# class Environment(object):
#     # INTRODUCING AND INITIALIZING ALL THE PARAMETERS AND VARIABLES OF THE ENVIRONMENT
#     def __init__(self):  # Satellite to BeiJing,Time span: 24 Nov 2020 04:51:27 to 24 Nov 2020 05:06:16
#         # count
#         self.counts = 0
#         # use a list to represent the time window
#         self.time_windows = []
#         # initial the total used time
#         self.total_time = 0.0
#         # the time span of the satellite
#         self.max_time = 888
#         # initial the free time
#         self.free_time = 888
#         # the total storage at the present:
#         self.total_storage = 0.0
#         # the maximum storage
#         self.max_storage = 175
#         # storage consumption Stori
#         self.each_storage = 5
#         # the total task
#         self.total_task = 50
#         # transfer time
#         self.transfer_time = 5.0
#         # initial profit
#         self.total_profit = 0
#         self.et = 0
#         self.atasks = 0
#         # inital tasks [[‘arrival time','shortest execution time','storage','profit','start time','end time'],[‘arrival time','shortest execution time','storage','profit','start time','end time'].....]

#         self.task=[]
#         self.arrival=[]
#         for i in range(self.total_task):
#             a= random.randint(0,888)
#             self.arrival.append(a)
#         self.arrival.sort()
#         for i in range(self.total_task):
#             self.task.append([])
#         self.each_storage = 5
#         for i in range(self.total_task):
#             self.task[i].append(self.arrival[i])
#             self.et= random.randint(10, 30)
#             self.task[i].append(self.et)
#             self.task[i].append(self.each_storage)
#             self.profit = random.randint(1,10)
#             self.task[i].append(self.profit)
#             self.task[i].append(0)
#             self.task[i].append(0)
#         # initial the state
#         # the state should include: [each task's arrival time,profit of each task,reamin storage,reamin free time]
#         self.state = None
#
#     # MAKING A METHOD THAT UPDATES THE ENVIRONMENT RIGHT AFTER THE AI PLAYS AN ACTION
#     def update_env(self, action):  # action['task_number']
#         self.atasks = len(self.time_windows)
#         # print("!!!!!!",self.time_windows)
#         # Getting game over
#         done = bool(
#             self.total_storage > self.max_storage
#             or self.counts >= self.total_task
#         )
#         self.done = bool(done)
#         # if done==False:
#         # print("!!!!",self.total_storage)
#         # print('***',self.counts)
#         # 未接受当前任务
#         if (action == 1):
#             self.counts = self.counts + 1
#             # normalize the state to [0,1]
#             normalized_arrival_time = self.task[self.counts][0] / self.max_time
#             normalized_profit = self.task[self.counts][3] / 10
#             normalized_storage = self.total_storage / self.max_storage
#             normalized_time = (self.max_time - self.total_time) / self.max_time
#             next_state = (normalized_arrival_time, normalized_profit, normalized_storage, normalized_time)
#             return np.array(next_state), self.total_profit, self.done
#         else:
#             # 分配起止时间
#             if (self.atasks == 0):
#                 self.task[self.counts][4] = self.task[self.counts][0]
#                 self.task[self.counts][5] = self.task[self.counts][4] + self.task[self.counts][1]
#
#             else:  # 接受任务且有合适的时间间隔来分配当前任务
#                 # 查看是否还有足够的时间来分配当前任务
#                 # 关于空闲时间的处理
#                 count2 = self.time_windows[self.atasks - 1]
#                 t = self.max_time - self.task[count2][5]
#                 q = 2 * self.transfer_time + self.task[self.counts][1]
#                 if (t > q):  # 比较最后一个任务的结束时间和当前任务的到达时间是否冲突
#                     if (self.task[count2][5] + 2 * self.transfer_time <= self.task[self.counts][0]):
#                         self.task[self.counts][4] = self.task[count2][5] + 2 * self.transfer_time
#                         self.task[self.counts][5] = self.task[self.counts][4] + self.task[self.counts][1]
#                     else:  # 接受了有时间但是会产生时间冲突
#                         self.counts = self.counts + 1
#                         # normalize the state to [0,1]
#                         normalized_arrival_time = self.task[self.counts][0] / self.max_time
#                         normalized_profit = self.task[self.counts][3] / 10
#                         normalized_storage = self.total_storage / self.max_storage
#                         normalized_time = (self.max_time - self.total_time) / self.max_time
#                         next_state = (0, 0, 0, 0)
#                         return np.array(next_state), self.total_profit, self.done
#
#                 else:  # 接受了但是没有合适的时间间隔来处理
#                     self.counts = self.counts + 1
#                     # normalize the state to [0,1]
#                     normalized_arrival_time = self.task[self.counts][0] / self.max_time
#                     normalized_profit = self.task[self.counts][3] / 10
#                     normalized_storage = self.total_storage / self.max_storage
#                     normalized_time = (self.max_time - self.total_time) / self.max_time
#                     next_state = (normalized_arrival_time, normalized_profit, normalized_storage, normalized_time)
#                     return np.array(next_state), self.total_profit, self.done
#
#         # Updating the total used storage
#         self.total_storage += self.task[self.counts][2]
#         # Updating the total taken time
#         self.total_time += self.task[self.counts][1]
#
#         # Update the reward
#
#         self.time_windows = self.time_windows + [self.counts]
#         self.total_profit = self.total_profit + self.task[self.counts][3]
#         self.counts = self.counts + 1
#         # normalize the state to [0,1]
#         normalized_arrival_time = self.task[self.counts][0] / self.max_time
#         normalized_profit = self.task[self.counts][3] / 10
#         normalized_storage = self.total_storage / self.max_storage
#         normalized_time = (self.max_time - self.total_time) / self.max_time
#         next_state = (normalized_arrival_time, normalized_profit, normalized_storage, normalized_time)
#         return np.array(next_state), self.total_profit, self.done
#
#     # MAKING A METHOD THAT RESETS THE ENVIRONMENT
#     def reset(self):
#         self.time_windows.clear()
#         # self.arrival.clear()
#         self.counts = 0
#         self.atasks = 0
#         # self.et = 0
#         self.total_storage = 0
#         self.total_profit = 0
#         self.done = False
#         self.total_time = 0.0
#         #        for i in range(50):
#         #            a= random.randint(0,888)
#         #            self.arrival.append(a)
#         #        self.arrival.sort()
#         #        for i in range(self.total_task):
#         #            self.task.append([])
#         #        self.each_storage = 5
#         #        for i in range(self.total_task):
#         #            self.task[i].append( self.arrival[i])
#         #            self.et= random.randint(10, 30)
#         #            self.task[i].append(self.et)
#         #            self.task[i].append(self.each_storage)
#         #            self.profit = random.randint(1,10)
#         #            self.task[i].append(self.profit)
#         #            self.task[i].append(0)
#         #            self.task[i].append(0)
#         self.state = None
#
#     def observe(self):
#         normalized_arrival_time = self.task[0][0] / self.max_time
#         normalized_profit = self.task[0][3] / 10
#         normalized_storage = self.total_storage / self.max_storage
#         normalized_time = (self.max_time - self.total_time) / self.max_time
#         current_state = (normalized_arrival_time, normalized_profit, normalized_storage, normalized_time)
#         return np.array(current_state)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print(time_end - time_start, 's')