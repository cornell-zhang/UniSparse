import sys
import re
import os
import json
import copy
import subprocess
import pathlib

exe_path = "/home/zz546/bench-sparse/cuda/build"
exe = "cusparse_spgemm"
input0_path = "/work/shared/common/datasets/versatile_sparse_xcel"

datasets = ["web-Google_csr_float32.npz",
            "mario002_csr_float32.npz",
            "amazon0312_csr_float32.npz",
            "m133-b3_csr_float32.npz",
            "scircuit_csr_float32.npz",
            "p2p-Gnutella31_csr_float32.npz",
            "offshore_csr_float32.npz",
            "cage12_csr_float32.npz",
            "2cubes_sphere_csr_float32.npz",
            "filter3D_csr_float32.npz",
            "ca-CondMat_csr_float32.npz",
            "wiki-Vote_csr_float32.npz",
            "poisson3Da_csr_float32.npz",
            "CollegeMsg_csr_float32.npz",
            "email-Eu-core_csr_float32.npz"]

runtimes = [1923, 6054, 3344, 9320, 7525, 24508, 1493, 3010, 3628, 1513, 12070, 12945, 8681, 27832, 27000]

datasets_num = len(datasets)
for d in range(datasets_num):
    powertest_run = {}

    numrun_str = str(runtimes[d])
    sh_cmd = "make " + exe + " CUDA_CFLAGS=-DVAR=" + numrun_str
    print(sh_cmd)
    os.system(sh_cmd)
    exe_cmd = exe_path + "/" + exe
    arg = input0_path + "/" + datasets[d]
    powertest_run[0] = subprocess.Popen([exe_cmd, arg], env=os.environ)
    powertest_run[1] = subprocess.Popen(["timeout", "60s", "nvidia-smi", "-i", " 0", "--query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr", "--format=csv", "-l", "1"], env=os.environ)

    powertest_run[0].wait()
    powertest_run[1].wait()
    sh_cmd = "make clean"
    print(sh_cmd)
    os.system(sh_cmd)
