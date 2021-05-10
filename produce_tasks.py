# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:31:03 2020

@author: 93585
"""
import numpy as np
import pandas as pd
import random
task=[]
arrival=[]
for i in range(50):
    a= random.randint(0,888)
    arrival.append(a)
arrival.sort()
for i in range(50):
    task.append([])
storage = 5
for i in range(50):
    task[i].append(arrival[i])
    et= random.randint(10,30)
    task[i].append(et)
    task[i].append(storage)
    profit = random.randint(1,10)
    task[i].append(profit)
    task[i].append(0)
    task[i].append(0)

print(task)