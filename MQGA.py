import random

import numpy as np


def init(popsize, n):
    population = []
    for i in range(popsize):
        pop = []
        for j in range(n):
            ###每个种群内的每个个体都可以获得一个角度
            pop.append(np.pi * 2 * random.random())
        population.append(pop)
    print(population)
    print(population[0][0])
    return population


def decode1(x, n, w, c, W):
    s = []  # 储存被选择物体的下标集合
    g = 0
    f = 0
    for i in range(n):
        temp = random.random()
        if temp < (np.sin(x[i])) ** 2:
            ###当容量小于最大容量时，才可以继续被选择，如果超出最大容量则直接停止
            if g + w[i] <= W:
                g = g + w[i]  ###容量
                f = f + c[i]  ###价值
                s.append(i)
            else:
                break
    return f, s


def fitnessfun1(population, n, w, c, W):
    value = []  ###储存每个种群的价值
    ss = []  ###储存每个种群被选择的索引
    for i in range(len(population)):
        [f, s] = decode1(population[i], n, w, c, W)
        value.append(f)
        ss.append(s)
    return value, ss


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


def mutation1(offspring, pm, nmut):
    for i in range(len(offspring)):
        r = np.random.uniform(0, 1)
        if r <= pm:
            ###随机选出一个点进行变异，如果该点是选择就变成不被选择，如果是不被选择则变成被选择
            point = np.random.randint(0, len(offspring[i]))
            offspring[i][point] += np.pi * 0.2 * random.random()
            nmut = nmut + 1
    return offspring

gen = 1000  # 迭代次数
pc = 0.25  # 交叉概率
pm = 0.5  # 变异概率
popsize = 10  # 种群大小
n = 20  # 物品数,即染色体长度n
w = [2, 5, 18, 3, 2, 5, 10, 4, 8, 12, 5, 10, 7, 15, 11, 2, 8, 10, 5, 9]  # 每个物品的重量列表
c = [5, 10, 12, 4, 3, 11, 13, 10, 7, 15, 8, 19, 1, 17, 12, 9, 15, 20, 2, 6]  # 每个物品的代价列表
W = 40  # 背包容量
M = 5  # 惩罚值
fun = 1  # 1-第一种解码方式，2-第二种解码方式（惩罚项）
# 初始化-------------------------------------------------------------------------
# 初始化种群（编码）


population = init(popsize, n)
# 适应度评价（解码）
if fun == 1:
    value, s = fitnessfun1(population, n, w, c, W)
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
for i in range(gen):
    print("迭代次数：")
    print(i)
    # 交叉
    offspring_c = crossover(population, pc, ncross)
    # 变异
    offspring_m=mutation1(offspring_c,pm,nmut)
    # offspring_m = mutation2(offspring_c, pm, nmut)
    mixpopulation = population + offspring_m
    # 适应度函数计算
    if fun == 1:
        value, s = fitnessfun1(mixpopulation, n, w, c, W)
    # 轮盘赌选择
    population = roulettewheel(mixpopulation, value, popsize)
    # 储存当代的最优解
    result = []
    if i == gen - 1:
        if fun == 1:
            value1, s1 = fitnessfun1(population, n, w, c, W)
            realvalue = s1
            result = value1
            last = value1
    else:
        if fun == 1:
            value1, s1 = fitnessfun1(population, n, w, c, W)
            result = value1
    maxre = max(result)
    h = result.index(max(result))
    # 将每代的最优解加入结果种群
    t.append(maxre)
    best_ind.append(population[h])

# 输出结果-----------------------------------------------------------------------
if fun == 1:
    best_value = max(t)
    hh = t.index(max(t))
    f2, s2 = decode1(best_ind[hh], n, w, c, W)
    print("最优组合为：")
    print(s2)
    print("最优解为：")
    print(f2)
    print("最优解出现的代数：")
    print(hh)
