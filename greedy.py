# In this task, 50 tasks will be scheduled
# Greedy algorithm will be used here, it means for tasks that arrive first, they are prioritized for scheduling without causing conflicts in time and space.
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from operator import itemgetter

avg_list = []
total_list = []
count_list = []
T = 3600


def plot_profit(profit, avg):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("Testing episdoes")
    host.set_ylabel("Total reward")

    plt.title('heuristic algorithm')

    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit", color = '#90EE90')
    p2, = host.plot(range(len(avg)), avg, label="Average profit", color = '#87CEEB')
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0, 225])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()


def plot_tasks(tasks):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total accepted tasks")

    plt.title('50 tasks in 888s')
    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Total Accepetd Tasks")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0, 50])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()

stor = 5  # each of the task's storage is 5
max_stor = 175  # the max storage of the satellite is 175
total_profit = 0
count = 0
num = 0
t1 = time.time()
for i in range(100):
    current_profit = 0
    current_stor = 0
    accepted_tasks = []
    tasks = [[] for j in range(50)]  # this means that the total number of tasks that should be scheduled is 50
    # each task's format:  ['arrival time', 'execution time', 'storage', 'profit', 'task_number', 'greedy']
    arr_time = []
    for k in range(50):
        temp_arr = random.randint(0, T)
        arr_time.append(temp_arr)
    arr_time.sort()

    for l in range(50):
        temp_execu = random.randint(10, 30)
        temp_profit = random.randint(1, 10)
        tasks[l].append(arr_time[l])
        tasks[l].append(temp_execu)
        tasks[l].append(stor)
        tasks[l].append(temp_profit)
        tasks[l].append(l)

    # above part is to create the tasks that need to be scheduled

    current_time = 0  # this time is used to give the specific time that will not cause any conflicts

    for m in range(50):
        # calculate the greedy number, the higher selection priority
        execu_percent = float(tasks[m][1] / T)
        stor_percent = float(tasks[m][2] / max_stor)
        a = float(execu_percent / stor_percent)
        # greedy_num = float(tasks[m][3] / tasks[m][1] / a) + float(tasks[m][3] / tasks[m][2])
        greedy_num = float(tasks[m][3]/(execu_percent*2+stor_percent))
        tasks[m].append(float(greedy_num))
        #heur

    tasks = sorted(tasks, key=(lambda x: x[5]), reverse=True)

    sche_time = []
    for m in range(T):
        sche_time.append(0)

    for m in range(50):
        conflitt = 0
        start_time = tasks[m][0]
        end_time = tasks[m][0] + tasks[m][1]
        if end_time > T-1:
            continue
        else:
            for n in range(start_time, end_time):
                if sche_time[n] == 1:
                    conflitt = 1
                    break
        if conflitt == 1:
            continue

        if current_stor + stor > max_stor:
            break

        count += 1
        current_profit += tasks[m][3]
        current_stor += stor
        num+=1
        for n in range(start_time, end_time):
            sche_time[n] = 1

    print("%d,   %d" % (count, current_profit))
    total_profit += current_profit
    avg = total_profit/(i+1)

    count_list.append(count)
    total_list.append(current_profit)
    avg_list.append(avg)
    count = 0

t2 = (time.time()-t1)/100
plot_profit(total_list, avg_list)
plot_tasks(count_list)
av = total_profit/100
print(av)
ff =  float(num/100)
print(ff)
print("执行时间 %f" % t2)

