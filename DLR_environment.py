"""
Description:
Online scheduling of image satellites based on neural networks and deep reinforcement learning:

Environment:
Given a finite time period and an image satellite with limited storage,tasks arrive dynamically,the time windows of the satellite for tasks can be obtained through per-processing,and the tasks could only
be observed inside the time windows.If task is accepted successfully,the profit associated with the accepted task
is received.The objective is to maximize the total profit from accepted tasks at the end of given time period.

Observation:

Type: list
Using a list called to show the observation
for example [0,1] means the satellite has accepted task 1 and task 2 at the present

Actions:
Type:
Num: for each task arrive dynamiclly, the satellites will accept or not
Using a number to show the action,for example 0 means accept this task, 1 means doesn't accept this task.

Reward:
The initial reward is zero, then if a task is accepted,the value of the task’s profit will be added into the total reward.

Starting State:
the start state is a satellite which does't accept any task:  []

Episode Termination:
1.When this total weight is larger than the  maximum  storage of the satellites
2.Time
3.Task number

The input of DQN network:
The input of the network is a sequence of scheduling information (denoted as si), including the arrival time of the arrival task (Ai), the profit of the arrival task
(xi), the remaining satellite storage ratio (Stor Ai ð Þ=StorMax), and the remaining free time ratio of the satellite (Pmk¼1ðetfk  stfkÞ=T).

The output of DQN network:
Q value
"""
import random

import numpy as np

task_number = 0
reward = 0


class Environment(object):
    # INTRODUCING AND INITIALIZING ALL THE PARAMETERS AND VARIABLES OF THE ENVIRONMENT
    def __init__(self):  # Satellite to BeiJing,Time span: 24 Nov 2020 04:51:27 to 24 Nov 2020 05:06:16
        # count
        self.counts = 0
        # use a list to represent the time window
        self.time_windows = []
        # initial the total used time
        self.total_time = 0.0
        # the time span of the satellite
        self.max_time = 3000
        # initial the free time
        self.free_time = 3000
        # the total storage at the present:
        self.total_storage = 0.0
        # the maximum storage
        self.max_storage = 175
        # storage consumption Stori
        self.each_storage = 5
        # the total task
        self.total_task = 50
        # transfer time
        self.transfer_time = 5.0
        # initial profit
        self.total_profit = 0
        self.et = 0
        self.atasks = 0
        # inital tasks [[‘arrival time','shortest execution time','storage','profit','start time','end time'],[‘arrival time','shortest execution time','storage','profit','start time','end time'].....]
        # self.task=[[5, 12, 5, 9, 0, 0], [36, 23, 5, 4, 0, 0], [40, 20, 5, 7, 0, 0], [43, 12, 5, 5, 0, 0], [60, 27, 5, 3, 0, 0], [67, 22, 5, 7, 0, 0], [75, 11, 5, 9, 0, 0], [138, 25, 5, 3, 0, 0], [138, 23, 5, 1, 0, 0], [140, 28, 5, 10, 0, 0], [147, 23, 5, 10, 0, 0], [157, 30, 5, 7, 0, 0], [158, 21, 5, 3, 0, 0], [197, 10, 5, 7, 0, 0], [197, 10, 5, 5, 0, 0], [211, 19, 5, 5, 0, 0], [233, 24, 5, 7, 0, 0], [250, 12, 5, 5, 0, 0], [277, 20, 5, 4, 0, 0], [279, 22, 5, 6, 0, 0], [302, 29, 5, 3, 0, 0], [348, 18, 5, 5, 0, 0], [354, 12, 5, 2, 0, 0], [366, 26, 5, 4, 0, 0], [384, 10, 5, 4, 0, 0], [391, 22, 5, 1, 0, 0], [393, 22, 5, 3, 0, 0], [393, 30, 5, 1, 0, 0], [409, 23, 5, 7, 0, 0], [413, 15, 5, 6, 0, 0], [415, 13, 5, 3, 0, 0], [438, 18, 5, 4, 0, 0], [475, 24, 5, 4, 0, 0], [508, 11, 5, 9, 0, 0], [532, 11, 5, 9, 0, 0], [533, 27, 5, 3, 0, 0], [590, 20, 5, 8, 0, 0], [601, 17, 5, 7, 0, 0], [624, 21, 5, 4, 0, 0], [640, 21, 5, 4, 0, 0], [642, 20, 5, 1, 0, 0], [645, 22, 5, 7, 0, 0], [648, 28, 5, 10, 0, 0], [680, 28, 5, 1, 0, 0], [706, 27, 5, 9, 0, 0], [707, 12, 5, 5, 0, 0], [743, 22, 5, 5, 0, 0], [783, 28, 5, 1, 0, 0], [835, 28, 5, 8, 0, 0], [860, 23, 5, 4, 0, 0]]
        # self.task=[[58, 25, 5, 10, 0, 0], [58, 14, 5, 7, 0, 0], [67, 11, 5, 3, 0, 0], [77, 22, 5, 5, 0, 0], [94, 29, 5, 4, 0, 0], [110, 29, 5, 1, 0, 0], [130, 11, 5, 5, 0, 0], [134, 12, 5, 3, 0, 0], [169, 28, 5, 9, 0, 0], [192, 17, 5, 8, 0, 0], [207, 17, 5, 7, 0, 0], [207, 18, 5, 5, 0, 0], [212, 19, 5, 3, 0, 0], [213, 24, 5, 3, 0, 0], [240, 29, 5, 7, 0, 0], [252, 29, 5, 9, 0, 0], [284, 21, 5, 5, 0, 0], [321, 30, 5, 5, 0, 0], [354, 17, 5, 5, 0, 0], [357, 22, 5, 4, 0, 0], [358, 11, 5, 7, 0, 0], [409, 13, 5, 6, 0, 0], [418, 28, 5, 3, 0, 0], [429, 11, 5, 8, 0, 0], [436, 30, 5, 2, 0, 0], [478, 14, 5, 8, 0, 0], [482, 21, 5, 6, 0, 0], [529, 28, 5, 4, 0, 0], [542, 20, 5, 6, 0, 0], [561, 21, 5, 6, 0, 0], [570, 25, 5, 1, 0, 0], [581, 12, 5, 3, 0, 0], [620, 28, 5, 2, 0, 0], [620, 27, 5, 2, 0, 0], [634, 20, 5, 9, 0, 0], [635, 29, 5, 6, 0, 0], [642, 23, 5, 4, 0, 0], [665, 19, 5, 3, 0, 0], [686, 25, 5, 6, 0, 0], [720, 30, 5, 2, 0, 0], [733, 15, 5, 5, 0, 0], [749, 16, 5, 10, 0, 0], [759, 24, 5, 10, 0, 0], [775, 16, 5, 4, 0, 0], [795, 13, 5, 6, 0, 0], [801, 16, 5, 9, 0, 0], [814, 24, 5, 2, 0, 0], [821, 15, 5, 6, 0, 0], [851, 15, 5, 7, 0, 0], [887, 30, 5, 5, 0, 0]]
        # self.task = [[4, 20, 5, 10, 0, 0], [28, 29, 5, 3, 0, 0], [44, 13, 5, 8, 0, 0], [46, 20, 5, 9, 0, 0],
        #              [65, 24, 5, 5, 0, 0], [70, 15, 5, 4, 0, 0], [72, 26, 5, 3, 0, 0], [85, 30, 5, 3, 0, 0],
        #              [89, 21, 5, 6, 0, 0], [153, 13, 5, 1, 0, 0], [171, 20, 5, 8, 0, 0], [190, 18, 5, 8, 0, 0],
        #              [213, 30, 5, 1, 0, 0], [214, 14, 5, 3, 0, 0], [219, 14, 5, 8, 0, 0], [226, 17, 5, 7, 0, 0],
        #              [226, 23, 5, 3, 0, 0], [230, 29, 5, 10, 0, 0], [313, 24, 5, 10, 0, 0], [337, 18, 5, 2, 0, 0],
        #              [338, 24, 5, 8, 0, 0], [341, 17, 5, 5, 0, 0], [351, 11, 5, 5, 0, 0], [358, 11, 5, 9, 0, 0],
        #              [376, 12, 5, 1, 0, 0], [429, 30, 5, 1, 0, 0], [463, 11, 5, 7, 0, 0], [517, 12, 5, 5, 0, 0],
        #              [528, 14, 5, 1, 0, 0], [529, 11, 5, 5, 0, 0], [541, 28, 5, 6, 0, 0], [555, 12, 5, 3, 0, 0],
        #              [560, 10, 5, 1, 0, 0], [577, 26, 5, 5, 0, 0], [612, 28, 5, 4, 0, 0], [635, 16, 5, 9, 0, 0],
        #              [670, 19, 5, 8, 0, 0], [679, 21, 5, 6, 0, 0], [694, 14, 5, 3, 0, 0], [699, 30, 5, 7, 0, 0],
        #              [701, 11, 5, 2, 0, 0], [715, 15, 5, 1, 0, 0], [717, 10, 5, 5, 0, 0], [719, 29, 5, 2, 0, 0],
        #              [755, 25, 5, 6, 0, 0], [771, 24, 5, 3, 0, 0], [774, 29, 5, 7, 0, 0], [801, 18, 5, 7, 0, 0],
        #              [837, 21, 5, 8, 0, 0], [861, 20, 5, 5, 0, 0]]
        self.task=[]
        self.arrival=[]
        for i in range(self.total_task):
            a= random.randint(0,2000)
            self.arrival.append(a)
        self.arrival.sort()
        for i in range(self.total_task):
            self.task.append([])
        self.each_storage = 5
        for i in range(self.total_task):
            self.task[i].append(self.arrival[i])
            self.et= random.randint(10, 30)
            self.task[i].append(self.et)
            self.task[i].append(self.each_storage)
            self.profit = random.randint(1,10)
            self.task[i].append(self.profit)
            self.task[i].append(0)
            self.task[i].append(0)
        # initial the state
        # the state should include: [each task's arrival time,profit of each task,reamin storage,reamin free time]
        self.state = None

    # MAKING A METHOD THAT UPDATES THE ENVIRONMENT RIGHT AFTER THE AI PLAYS AN ACTION
    def update_env(self, action):  # action['task_number']
        self.atasks = len(self.time_windows)
        # print("!!!!!!",self.time_windows)
        # Getting game over
        done = bool(
            self.total_storage >= self.max_storage
            or self.counts >= self.total_task
        )
        self.done = bool(done)
        # if done==False:
        # print("!!!!",self.total_storage)
        # print('***',self.counts)
        # 未接受当前任务
        if (action == 1):
            self.counts = self.counts + 1
            # normalize the state to [0,1]
            normalized_arrival_time = self.task[self.counts][0] / self.max_time
            normalized_profit = self.task[self.counts][3] / 10
            normalized_storage = self.total_storage / self.max_storage
            normalized_time = (self.max_time - self.total_time) / self.max_time
            next_state = (normalized_arrival_time, normalized_profit, normalized_storage, normalized_time)
            return np.array(next_state), self.total_profit, self.done
        else:
            # 分配起止时间
            if (self.atasks == 0):
                self.task[self.counts][4] = self.task[self.counts][0]
                self.task[self.counts][5] = self.task[self.counts][4] + self.task[self.counts][1]

            else:  # 接受任务且有合适的时间间隔来分配当前任务
                # 查看是否还有足够的时间来分配当前任务
                # 关于空闲时间的处理
                count2 = self.time_windows[self.atasks - 1]
                t = self.max_time - self.task[count2][5]
                q = 2 * self.transfer_time + self.task[self.counts][1]
                if (t > q):  # 比较最后一个任务的结束时间和当前任务的到达时间是否冲突
                    if (self.task[count2][5] + 2 * self.transfer_time <= self.task[self.counts][0]):
                        self.task[self.counts][4] = self.task[count2][5] + 2 * self.transfer_time
                        self.task[self.counts][5] = self.task[self.counts][4] + self.task[self.counts][1]
                    else:  # 接受了有时间但是会产生时间冲突
                        self.counts = self.counts + 1
                        # normalize the state to [0,1]
                        normalized_arrival_time = self.task[self.counts][0] / self.max_time
                        normalized_profit = self.task[self.counts][3] / 10
                        normalized_storage = self.total_storage / self.max_storage
                        normalized_time = (self.max_time - self.total_time) / self.max_time
                        next_state = (0, 0, 0, 0)
                        return np.array(next_state), self.total_profit, self.done

                else:  # 接受了但是没有合适的时间间隔来处理
                    self.counts = self.counts + 1
                    # normalize the state to [0,1]
                    normalized_arrival_time = self.task[self.counts][0] / self.max_time
                    normalized_profit = self.task[self.counts][3] / 10
                    normalized_storage = self.total_storage / self.max_storage
                    normalized_time = (self.max_time - self.total_time) / self.max_time
                    next_state = (normalized_arrival_time, normalized_profit, normalized_storage, normalized_time)
                    return np.array(next_state), self.total_profit, self.done

        # Updating the total used storage
        self.total_storage += self.task[self.counts][2]
        # Updating the total taken time
        self.total_time += self.task[self.counts][1]

        # Update the reward

        self.time_windows = self.time_windows + [self.counts]
        self.total_profit = self.total_profit + self.task[self.counts][3]
        self.counts = self.counts + 1
        # normalize the state to [0,1]
        normalized_arrival_time = self.task[self.counts][0] / self.max_time
        normalized_profit = self.task[self.counts][3] / 10
        normalized_storage = self.total_storage / self.max_storage
        normalized_time = (self.max_time - self.total_time) / self.max_time
        next_state = (normalized_arrival_time, normalized_profit, normalized_storage, normalized_time)
        return np.array(next_state), self.total_profit, self.done

    # MAKING A METHOD THAT RESETS THE ENVIRONMENT
    def reset(self):
        self.time_windows.clear()
        # self.arrival.clear()
        self.counts = 0
        self.atasks = 0
        # self.et = 0
        self.total_storage = 0
        self.total_profit = 0
        self.done = False
        self.total_time = 0.0
        #        for i in range(50):
        #            a= random.randint(0,888)
        #            self.arrival.append(a)
        #        self.arrival.sort()
        #        for i in range(self.total_task):
        #            self.task.append([])
        #        self.each_storage = 5
        #        for i in range(self.total_task):
        #            self.task[i].append( self.arrival[i])
        #            self.et= random.randint(10, 30)
        #            self.task[i].append(self.et)
        #            self.task[i].append(self.each_storage)
        #            self.profit = random.randint(1,10)
        #            self.task[i].append(self.profit)
        #            self.task[i].append(0)
        #            self.task[i].append(0)
        self.state = None

    def observe(self):
        normalized_arrival_time = self.task[0][0] / self.max_time
        normalized_profit = self.task[0][3] / 10
        normalized_storage = self.total_storage / self.max_storage
        normalized_time = (self.max_time - self.total_time) / self.max_time
        current_state = (normalized_arrival_time, normalized_profit, normalized_storage, normalized_time)
        return np.array(current_state)





