import sys
import re
import os
import json
import copy
import subprocess
import pathlib

exe_path = "/home/zz546/Sparse_Layout_Dialect/evaluation/CPU/build"
exe = "mkl_spmm"
input0_path = "/work/shared/common/datasets/versatile_sparse_xcel"

datasets = ["web-Google_row_major.mtx",
            "mario002_row_major.mtx",
            "amazon0312_row_major.mtx",
            "m133-b3_row_major.mtx",
            "scircuit_row_major.mtx",
            "p2p-Gnutella31_row_major.mtx",
            "offshore_row_major.mtx",
            "cage12_row_major.mtx",
            "2cubes_sphere_row_major.mtx",
            "filter3D_row_major.mtx",
            "ca-CondMat_row_major.mtx",
            "wiki-Vote_row_major.mtx",
            "poisson3Da_row_major.mtx",
            "CollegeMsg_row_major.mtx",
            "email-Eu-core_row_major.mtx"]

runtimes = [84, 395, 151, 1359, 1077, 3366, 200, 573, 726, 499, 5354, 12481, 3630, 51277, 64649]

datasets_num = len(datasets)
for d in range(datasets_num):
    powertest_run = {}

    numrun_str = str(runtimes[d]) 
    sh_cmd = "make " + exe + " CXXFLAGS=-Dvar=" + numrun_str
    print(sh_cmd)
    os.system(sh_cmd)
    exe_cmd = exe_path + "/" + exe
    arg = input0_path + "/" + datasets[d] 
    powertest_run[0] = subprocess.Popen([exe_cmd, arg], env=os.environ)
    powertest_run[1] = subprocess.Popen(["powerstat", "-R", "0.5", "120"], env=os.environ) 
    powertest_run[0].wait()
    powertest_run[1].wait()
    sh_cmd = "make clean"
    print(sh_cmd)
    os.system(sh_cmd)
  
