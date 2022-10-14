import sys
import os
import json
import copy
import subprocess
import numpy as np
import pathlib

current_path = str(pathlib.Path(__file__).parent.absolute())
openfile = current_path + "/label_propgation_gpua6000_power.log"
f = open(openfile, 'r')
alllines = f.readlines()
f.close()

samples = np.zeros((60), dtype=float) 
avg_array = np.zeros((5), dtype=float)

line = 0
for eachline in alllines:
    if eachline.__contains__("2022/07/21"):
        block_data = eachline.split(",")
        power_str = block_data[2]
        power_data = power_str.split(" ")
        print(float(power_data[1]))
        print(line)
        if line % 60 > 7:
            samples[line % 60] = float(power_data[1])
        line = line + 1
        if line % 60 == 59:
            line_idx = line // 60
            avg_array[line_idx] = np.sum(samples) / 53


print(avg_array)
#print(samples)
#avg = np.average(samples)
#print("Average power of spmv, spmm, spgemm, spmspv is: ", avg)
