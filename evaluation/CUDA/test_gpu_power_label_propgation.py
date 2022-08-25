import sys
import re
import os
import json
import copy
import subprocess
import pathlib

exe_path = "/home/zz546/bench-sparse/cuda/build"
exe1 = "cusparse_spgemm_lp"
exe2 = "cusparse_spmm"
input_path = "/work/shared/common/datasets/versatile_sparse_xcel/lp"

prefix = "arxiv"
input0 = "_adj_matrix_csr_float32.npz"
datasets = ["_label_matrix_1_csr_float32.npz", 
            "_label_matrix_2_csr_float32.npz",
            "_label_matrix_3_csr_float32.npz", 
            "_label_matrix_4_csr_float32.npz"]

spgemm_runs = [16104, 7408, 2969, 1931]
spmm_runs = 3357

datasets_num = len(datasets)
for d in range(datasets_num):
    powertest_run = {}

    numrun_str = str(spgemm_runs[d])
    sh_cmd = "make " + exe1 + " CUDA_CFLAGS=-DVAR=" + numrun_str
    print(sh_cmd)
    os.system(sh_cmd)
    exe_cmd = exe_path + "/" + exe1
    arg0 = input_path + "/" + prefix + input0
    arg1 = input_path + "/" + prefix + datasets[d]
    powertest_run[0] = subprocess.Popen([exe_cmd, arg0, arg1], env=os.environ)
    powertest_run[1] = subprocess.Popen(["timeout", "60s", "nvidia-smi", "-i", " 0", "--query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr", "--format=csv", "-l", "1"], env=os.environ)

    powertest_run[0].wait()
    powertest_run[1].wait()
    sh_cmd = "make clean"
    print(sh_cmd)
    os.system(sh_cmd)

sh_cmd = "make " + exe2 + " CUDA_CFLAGS=-DVAR=" + str(spmm_runs)
print(sh_cmd)
os.system(sh_cmd)
exe_cmd = exe_path + "/" + exe2
arg = input_path + "/" + prefix + input0
run_spmm = subprocess.Popen([exe_cmd, arg], env=os.environ)
powerstat = subprocess.Popen(["timeout", "60s", "nvidia-smi", "-i", " 0", "--query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr", "--format=csv", "-l", "1"], env=os.environ)
run_spmm.wait()
powerstat.wait()
sh_cmd = "make clean"
print(sh_cmd)
os.system(sh_cmd)


