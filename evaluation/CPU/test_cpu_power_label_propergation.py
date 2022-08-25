import sys
import re
import os
import json
import copy
import subprocess
import pathlib

exe_path = "/home/zz546/Sparse_Layout_Dialect/evaluation/CPU/build"
exe1 = "mkl_spgemm_lp"
exe2 = "mkl_spmm"
input_path = "/work/shared/common/datasets/versatile_sparse_xcel/lp"

prefix = "arxiv"
input0= "_adj_matrix_row_major.mtx"
datasets = ["_label_matrix_1.mtx", 
            "_label_matrix_2.mtx",
            "_label_matrix_3.mtx", 
            "_label_matrix_4.mtx"]
 
spgemm_runs = [15170, 3610, 1125, 776]
spmm_runs = 6529

datasets_num = len(datasets)
for d in range(datasets_num):
    powertest_run = {}
    numrun_str = str(spgemm_runs[d]) 
    sh_cmd = "make " + exe1 + " CXXFLAGS=-Dvar=" + numrun_str
    print(sh_cmd)
    os.system(sh_cmd)
    exe_cmd = exe_path + "/" + exe1
    arg0 = input_path + "/" + prefix + input0
    arg1 = input_path + "/" + prefix + datasets[d] 
    powertest_run[0] = subprocess.Popen([exe_cmd, arg0, arg1], env=os.environ)
    powertest_run[1] = subprocess.Popen(["powerstat", "-R", "0.5", "120"], env=os.environ) 
    powertest_run[0].wait()
    powertest_run[1].wait()
    sh_cmd = "make clean"
    print(sh_cmd)
    os.system(sh_cmd)

sh_cmd = "make " + exe2 + " CXXFLAGS=-Dvar=" + str(spmm_runs)
print(sh_cmd)
os.system(sh_cmd)
exe_cmd = exe_path + "/" + exe2
arg = input_path + "/" + prefix + input0
run_spmm = subprocess.Popen([exe_cmd, arg], env=os.environ)
run_powerstat = subprocess.Popen(["powerstat", "-R", "0.5", "120"], env=os.environ)
run_spmm.wait()
run_powerstat.wait()
sh_cmd = "make clean"
print(sh_cmd)
os.system(sh_cmd)  
