# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:25:45 2020

@author: 93585
"""


# -*- coding: utf-8 -*-

# 包导入
import time
import numpy as np
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from DLR_environment import Environment
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot




# 实现一个namedtuple用例
from collections import namedtuple

Tr = namedtuple('tr',('name_a','value_b'))
Tr_object=Tr('名称为A',100)
print(Tr_object)
print(Tr_object.value_b)
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

# 常量的设定
GAMMA = 0.99 # 时间折扣率
MAX_TASKS = 50 # 1次实验的step数
NUM_EPISODES = 100  # 最大尝试次数
eval_profit_list = []
avg_profit_list=[]
eval_tasks_list=[]
avg_list = []
def plot_profit(profit,avg):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total reward")
 
    plt.title('50 tasks in 888s')
  

    
    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit")
    p2, = host.plot(range(len(avg)), avg, label="Average profit")
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0,200])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.show()
    
    
def plot_tasks(tasks):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total accepted tasks")
 
    plt.title('')
    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Total Accepetd Tasks")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0,100])
    host.set_ylim([0,50])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.show()


def plot_pprofit(tasks):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total Profit")

    plt.title('')
    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Total Profit")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0, 250])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()

def plot_avgprofit(tasks):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Average Profit")

    plt.title('')
    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Average Profit")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0, 10])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()
 #定义于存储经验的内存类


class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # 下面memory的最大长度
        self.memory = []  # 存储过往的经验
        self.index = 0  # 表示要保存的索引

    def push(self, state, action, state_next, reward):
        '''将transition = (state, action, state_next, reward)保存在存储器中'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 内存未满时添加

        # 使用namedtuple对象Transition将值和字段名称保存为一对
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 将保存的index移动一位

    def sample(self, batch_size):
        '''随机检索Batch_size大小的样本并返回'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''返回当前memory长度'''
        return len(self.memory)


# BATCH_SIZE = 100
BATCH_SIZE = 25
CAPACITY = 1000


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 获取CartPloe 的2个动作（向左或向右）

        # 创建存储经验的对象
        self.memory = ReplayMemory(CAPACITY)

        # 构建一个神经网络
        self.model = nn.Sequential()
        self.model.add_module('fc1',nn.Linear(num_states,32))
        self.model.add_module('relu1',nn.ReLU())
        self.model.add_module('fc2',nn.Linear(32,32))
        self.model.add_module('relu2',nn.ReLU())
        self.model.add_module('fc3',nn.Linear(32, num_actions))

        print(self.model)  # 输出网络的形状

        # 最优方法的设定
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.003)
        #0.005  0.00625    
        
    def replay(self):
        '''通过Experience Replay学习网络的连接参数'''

        # 1.检查经验池的大小
        #经验池大小小于小批量数据时不执行任何操作
        if len(self.memory) < BATCH_SIZE:
            return

        # 2.创建小批量数据
        #2.1从经验池获取小批量数据
        transitions = self.memory.sample(BATCH_SIZE)
        
        #2.2将每个变量转换为小批量数据对应的形式
        #的带的transitions存储了一个BATCH_SIZE的（state，action,state_next，reward)
        #即（state，action,state_next，reward)*BATCH_SIZE
        #想把它变成小批量数据换句话说
        #设为（statexBATCH_SIZE，actionxBATCH_SIZE,state_nextxBATCH_SIZE，rewardxBATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        #2.3将每个变量的元素转换为与小批量数据对应的形式
        #例如，对于state，形状为[torch.FloatTensor of size 1x4]
        #将其转换为 torch.FloatTensor of  BATCH_SIZEx 4
        #cat 是指 Concatenates(连接)
        
        state_batch = torch.cat(batch.state)
        #print(state_batch)
        action_batch = torch.cat(batch.action)
        #print(action_batch)
        #print('****',batch.reward)
        reward_batch = torch.cat(batch.reward)
        #print(reward_batch)
        #print('!!!',batch.next_state)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
    
        #3求取Q（s_t,a_t)值作为监督信号
        #3.1将网络切换到推理模式
        self.model.eval()
        #3.2求取网络输出的Q（s_t,a_t)
        #self.model(state_batch)输出左右两个q值
        #成为[torch.FloatTensor of size BATCH_SIZEx 2]
        #为了求得于此处执行顶多a_t对应的Q值，求取由action_batch执行的动作a_t是向右还是向左index
        #用gather获得相应的Q值
       # print("******************")
       # print(self.model(state_batch))
        #print(self.model(state_batch).gather(1, action_batch))
        state_action_values = self.model(state_batch).gather(1, action_batch)
        #3.3求取max{Q(s_t+1,a)}值，请注意以下状态
        #创建索引掩码一检查cartpole是否未完成且具有next_state
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,batch.next_state)))
        #首先全部设置为0
        next_state_values = torch.zeros(BATCH_SIZE)
        #求取具有下一次状态的index的最大Q值
        #访问输出并通过max()求列方向最大值的[]value,index
        #并输出其Q值（index=0）
        #用detach取出该值
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        #3.4从Q公式中求取Q（s_t,a_t)值作为监督信息
        expected_state_action_values = reward_batch + GAMMA * next_state_values
        
        
        #4更新连接参数
        #4.1 切换到训练模式
        self.model.train()

        #4.2 计算损失函数（smooth_l1_loss 是Huber loss)
        #expected_state_action_values 的size是[minbatch],通过
        #unsqueeze得到[minibatchx1]
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))
        
        #更新连接参数
        self.optimizer.zero_grad()  #重置改变
        loss.backward()  # 计算反向传播
        self.optimizer.step()  #更新连接参数
        
    def decide_action(self,state,episode):
        '''根据当前状态确定动作'''
        #采用e -贪婪法逐步采用最佳动作
        epslion = 0.5*(1/(episode+1))
        #print('*******',self.model(state))
        if epslion<= np.random.uniform(0,1):
            self.model.eval()#将网络切换到推理模式
            with torch.no_grad():
                action=self.model(state).max(1)[1].view(1,1)
            #获取网络最大值的索引 index = max(1)[1]
            #.view(1,1)将[torch.LongTwnsor of size 1]转换为size 1x1 大小
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])#随即返回0，1的动作
            #action 的形式为[torch.LongTwnsor of size 1x1]
        return action

#这是一个在CartPole上运行的智能体类，他是一个带有杆的小车
class Agent:
    def __init__(self, num_states, num_actions):
        '''设置任务状态和动作数量'''
        self.brain = Brain(num_states, num_actions)  # 为智能体生成大脑来决定他们的动作

    def update_q_function(self):
        '''更新Q函数'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''确定动作'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''将state, action, state_next, reward的内容保存在经验池中'''
        self.brain.memory.push(state, action, state_next, reward)


# 这是一个执行CartPole的环境类

class Environment_d:

    def __init__(self):
        self.env = Environment()  # 设定要执行的任务
        self.num_states = 4  # 设定任务状态和动作数量
        self.num_actions = 2 # 动作
        # 创建Agent在环境中执行的动作
        self.agent = Agent(self.num_states,self.num_actions)

    def run(self):
        print(self.env.task)
        summ = 0
        stask = 0
        atask = 0
        av = 0
        t1 = time.time()
        for episode in range(NUM_EPISODES):  # 重复试验次数
            total_reward = 0
            accepted_tasks=0
            self.env.reset()
            observation = self.env.observe()  # 环境初始化

            state = observation  # 直接使用观测作为状态state使用
            state = torch.from_numpy(state).type(
                torch.FloatTensor)  #将Numpy变量转换为 Pytorch Tensor
            #FloatTensor of  size  4转换为size 1x4
            state = torch.unsqueeze(state, 0)  # 
            for step in range(MAX_TASKS-1):  # 1 episode 循环
                action = self.agent.get_action(state, episode)  # 求取动作
                
                # 通过执行动作a_t求s_{t+1}和done标志
                # 从action中指定.item()并获取内容
                observation_next, reward, done = self.env.update_env(
                    action.item())  # 使用'_'是因为在后边的流程中不适用reward 和 info

                # 给予奖励，对epsoide是否结束以及是否有下一个状态进行判断
                if done: 
                    state_next = None
                    break
                else: 
                    reward_t = torch.FloatTensor([reward]) 
                    
                    state_next = observation_next  # 保持观察不变
                    #print('???',state_next)
                    state_next = torch.from_numpy(state_next).type(
                        torch.FloatTensor)  # 将Numpy变量转换为 Pytorch Tensor
                    state_next = torch.unsqueeze(state_next, 0)  # FloatTensor of  size  4转换为size 1x4
                    #print('***',state_next)
                # 向经验池中添加经验
                self.agent.memorize(state, action, state_next, reward_t)

                # 经验回放中更新Q函数
                self.agent.update_q_function()

                # 更新观测值
                state = state_next
            
            state_next = None
            total_reward = self.env.total_profit
            accepted_tasks= self.env.atasks
            eval_profit_list.append(total_reward)
            avg_1 = sum(eval_profit_list)/len(eval_profit_list)
            avgg = total_reward/accepted_tasks
            avg_list.append(avgg)
            avg_profit_list.append(avg_1)        
            eval_tasks_list.append(accepted_tasks)
            print('%d Episode: Accepted tasks numbers：%d Total reward: %d'%(episode,accepted_tasks,total_reward))
            summ += total_reward
            stask += accepted_tasks
            atask = stask/100
            av = summ/100
            #print(self.env.time_windows)
        print('%d' % av)
        print(atask)

        t2 = (time.time()-t1)/100
        print("The execution time is %f" % t2)
        # plot_profit(eval_profit_list,avg_profit_list)
        plot_pprofit(eval_profit_list)
        plot_tasks(eval_tasks_list)
        plot_avgprofit(avg_list)
                    
# main 类
cartpole_env = Environment_d()
cartpole_env.run()