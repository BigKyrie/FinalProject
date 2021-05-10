import random

import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.axes_grid1 import host_subplot

llll = 1
mmmm = 10
summ = 0

avg_list = []
total_list = []
count_list =[]


def plot_profit(profit, avg):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

    # set labels
    host.set_xlabel("Number of iterations")
    host.set_ylabel("Total reward")

    plt.title('Quantum genetic algorithm')

    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit", color ='#90EE90')
    p2, = host.plot(range(len(avg)), avg, label="Average profit")
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 1000])
    host.set_ylim([0, 200])
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
    host.set_xlim([0, 100])
    host.set_ylim([0, 50])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()


###初始化一个种群 popsize为种群个数，n为每个种群内的个数
def init(popsize, n):
    population = []
    for i in range(popsize):
        pop = ''
        for j in range(n):
            ###每个种群内的每个个体可以取值为0或1，0为不被选择，1为被选择
            pop = pop + str(np.random.randint(0, 2))
        population.append(pop)
    return population


def decode1(x, n, w, c, W, ttime, T):
    s = []  # 储存被选择物体的下标集合
    ttt = []
    ###初始化时间
    for j in range(T + 1):
        ttt.append(0)
    g = 0
    f = 0
    # for i in range(n):
    #     if x[i] == '1':
    #         ###当容量小于最大容量时，才可以继续被选择，如果超出最大容量则直接停止
    #         tt = np.random.randint(0, T)  ###选择一个该任务的开始时间
    #         count = 0
    #         if g + w <= W:
    #             for k in range(tt, tt + ttime):
    #                 if k > T - ttime:
    #                     break
    #                 elif ttt[k] == 0:
    #                     count += 1
    #             if count == ttime:
    #                 g = g + w  ###容量
    #                 f = f + c[i]  ###价值
    #                 s.append(i)
    #                 for k in range(tt, tt + ttime):
    #                     ttt[k] = 1
    #             else:
    #                 continue
    #         else:
    #             break
    arrival_time = produce_arrive_tasks()
    exetime = produce_exetime()
    current_weight = 0
    for i in range(n):
        if x[i] == '1':
            if current_weight + w <= W and ttt[arrival_time[i]] == 0 and arrival_time[i] + exetime[i] < T:
                current_weight += w
                f += c[i]
                s.append(i)
                for j in range(arrival_time[i], arrival_time[i] + exetime[i]):
                    ttt[j] = 1
            else:
                continue
    return f, s


def fitnessfun1(population, n, w, c, W, ttime, T):
    value = []  ###储存每个种群的价值
    ss = []  ###储存每个种群被选择的索引
    for i in range(len(population)):
        [f, s] = decode1(population[i], n, w, c, W, ttime, T)
        value.append(f)
        ss.append(s)
    return value, ss


###轮盘模型
###以每个种群的价值占总价值和的比作为轮盘的构成，价值高的则占轮盘的面积大，即该染色体生存或选择概率更大
def roulettewheel(population, value, pop_num):
    fitness_sum = []
    ###价值总和
    value_sum = sum(value)
    ###每个价值的分别占比， 总和为1
    fitness = [i / value_sum for i in value]
    ###从种群索引0开始逐渐构成一个总和为1的轮盘
    for i in range(len(population)):  ##
        if i == 0:
            fitness_sum.append(fitness[i])
        else:
            fitness_sum.append(fitness_sum[i - 1] + fitness[i])
    population_new = []
    for j in range(pop_num):  ###
        ###轮盘指针随机转
        r = np.random.uniform(0, 1)
        ###选择是哪一个种群（染色体）被选中了
        for i in range(len(fitness_sum)):  ###
            if i == 0:
                if 0 <= r <= fitness_sum[i]:
                    population_new.append(population[i])
            else:
                if fitness_sum[i - 1] <= r <= fitness_sum[i]:
                    population_new.append(population[i])
    return population_new


###交叉
def crossover(population_new, pc, ncross):
    a = int(len(population_new) / 2)
    ###选择出所有种群的双亲（所有染色体的双亲）
    parents_one = population_new[:a]
    parents_two = population_new[a:]
    ###随机每个种群（染色体的顺序）
    np.random.shuffle(parents_one)
    np.random.shuffle(parents_two)
    ###后代
    offspring = []
    for i in range(a):
        r = np.random.uniform(0, 1)
        if r <= pc:
            ###在每个种群中产生两个断点
            point1 = np.random.randint(0, (len(parents_one[i]) - 1))
            point2 = np.random.randint(point1, len(parents_one[i]))
            ###两个父代交叉产生两个后代，假如父代分别为 abc和def 则两个后代为aec和dbf
            off_one = parents_one[i][:point1] + parents_two[i][point1:point2] + parents_one[i][point2:]
            off_two = parents_two[i][:point1] + parents_one[i][point1:point2] + parents_two[i][point2:]
            ncross = ncross + 1
        else:
            off_one = parents_one[i]
            off_two = parents_two[i]
        offspring.append(off_one)
        offspring.append(off_two)
    return offspring


###变异1
###每整条染色体分别检验变异概率，如果变异，则在该染色体上产生一个需要变异的点
def mutation1(offspring, pm, nmut):
    for i in range(len(offspring)):
        r = np.random.uniform(0, 1)
        if r <= pm:
            ###随机选出一个点进行变异，如果该点是选择就变成不被选择，如果是不被选择则变成被选择
            point = np.random.randint(0, len(offspring[i]))
            if point == 0:
                if offspring[i][point] == '1':
                    offspring[i] = '0' + offspring[i][1:]
                else:
                    offspring[i] = '1' + offspring[i][1:]
            else:
                if offspring[i][point] == '1':
                    offspring[i] = offspring[i][:(point - 1)] + '0' + offspring[i][point:]
                else:
                    offspring[i] = offspring[i][:(point - 1)] + '1' + offspring[i][point:]
            nmut = nmut + 1
    return offspring


# 对每条染色体上的每个点进行变异概率检验
def mutation2(offspring, pm, nmut):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            r = np.random.uniform(0, 1)
            if r <= pm:
                if j == 0:
                    if offspring[i][j] == '1':
                        offspring[i] = '0' + offspring[i][1:]
                    else:
                        offspring[i] = '1' + offspring[i][1:]
                else:
                    if offspring[i][j] == '1':
                        offspring[i] = offspring[i][:(j - 1)] + '0' + offspring[i][j:]
                    else:
                        offspring[i] = offspring[i][:(j - 1)] + '1' + offspring[i][j:]
                nmut = nmut + 1
    return offspring


def producetasks():
    tasks = []
    for i in range(n):
        individual_profit = random.randint(llll, mmmm)
        tasks.append(individual_profit)

    return tasks


def produce_arrive_tasks():
    arr_time = []
    for k in range(n):
        temp_arr = random.randint(0, T)
        arr_time.append(temp_arr)
    arr_time.sort()
    return arr_time


def produce_exetime():
    exetime_tasks = []
    for i in range(n):
        excu = random.randint(10, 30)
        exetime_tasks.append(excu)
    return exetime_tasks


# 主程序----------------------------------------------------------------------------------------------------------------------------------
# 参数设置-----------------------------------------------------------------------
gen = 1000  # 迭代次数
pc = 0.3  # 交叉概率
pm = 0.05  # 变异概率
popsize = 30  # 种群大小
n = 50  # 任务数量,即染色体长度n
# c = [5, 7, 9, 4, 3, 5, 6, 4, 7, 1, 8, 6, 1, 7, 2, 9, 5, 3, 2, 6]  # 每个物品的价值列表
c = producetasks()
w = 5  # 每个物品所占据的重量
W = 175  # 存储空间
ttime = 6  # 每个任务花费的时间
T = 3600 # 总时间区间

fun = 1  # 1-第一种解码方式，2-第二种解码方式（惩罚项）
# 初始化-------------------------------------------------------------------------
# 初始化种群（编码）


population = init(popsize, n)
# 适应度评价（解码）
if fun == 1:
    value, s = fitnessfun1(population, n, w, c, W, ttime, T)
# 初始化交叉个数
ncross = 0
# 初始化变异个数
nmut = 0
# 储存每代种群的最优值及其对应的个体
t = []
best_ind = []
last = []  # 储存最后一代个体的适应度值
realvalue = []  # 储存最后一代解码后的值
# 循环---------------------------------------------------------------------------
t1 = time.time()
for i in range(gen):
    print("迭代次数：")
    print(i)
    # 交叉
    offspring_c = crossover(population, pc, ncross)
    # 变异
    # offspring_m=mutation1(offspring,pm,nmut)
    offspring_m = mutation2(offspring_c, pm, nmut)
    mixpopulation = population + offspring_m
    # 适应度函数计算
    if fun == 1:
        value, s = fitnessfun1(mixpopulation, n, w, c, W, ttime, T)
    # 轮盘赌选择
    population = roulettewheel(mixpopulation, value, popsize)
    # 储存当代的最优解
    result = []
    if i == gen - 1:
        if fun == 1:
            value1, s1 = fitnessfun1(population, n, w, c, W, ttime, T)
            realvalue = s1
            result = value1
            last = value1
    else:
        if fun == 1:
            value1, s1 = fitnessfun1(population, n, w, c, W, ttime, T)
            result = value1
    print(result)
    maxre = max(result)
    h = result.index(max(result))
    print(h)
    # 将每代的最优解加入结果种群
    t.append(maxre)
    total_list.append(maxre)
    summ += maxre
    avg = summ / (gen + 1)

    best_ind.append(population[h])

# 输出结果-----------------------------------------------------------------------
if fun == 1:
    best_value = max(t)
    hh = t.index(max(t))
    f2, s2 = decode1(best_ind[hh], n, w, c, W, ttime, T)
    print("此次最优组合为：")
    print(s2)
    print("此次最优解为：")
    print(max(t))
    print("此次最优解出现的代数：")
    print(hh)
    t2 = time.time() - t1
    print("执行时间 %f" % t2)
    plot_profit(total_list, avg_list)


