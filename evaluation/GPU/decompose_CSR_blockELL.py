import scipy.sparse as sparse
from scipy.sparse import coo_matrix
from scipy.io import mmread
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import copy

def ceildiv(a, b):
    return -(a // -b)

def sparse_plot(filename):
    A = mmread(filename)
    plt.figure(figsize=(5, 5))
    plt.spy(A, markersize=1)
    plt.savefig("density"+filename+".png")
    plt.show()
    plt.close()

def decompose_CSR_ELL(filename, block_size, block_density_thres, col_density_thres):
    A = mmread(filename)
    row_size = A.shape[0]
    col_size = A.shape[1]
    row_block = ceildiv(row_size, block_size)
    col_block = ceildiv(col_size, block_size)
    row_block_cols=[]
    
    # store block col id in row_block_cols[] if nnz>=50
    block_nnz = [0]*col_block
    cur_row_block = 0
    for j in range(A.data.size):
        row_block_id = A.row[j]//block_size
        col_block_id = A.col[j]//block_size
        if row_block_id == cur_row_block:
            block_nnz[col_block_id] += 1
        else: 
            # print(A.row[j])
            # print(len(block_nnz))
            # print(block_nnz)
            block_index = [i for i in range(len(block_nnz)) \
                if block_nnz[i] >= block_size*block_size*block_density_thres]
            # for nnz in block_nnz:
            #     if nnz >= block_size*block_size*block_density_thres:
                    # ell_nnz = ell_nnz+nnz
            row_block_cols.append(block_index)
            cur_row_block = row_block_id
            block_nnz = [0]*col_block
            block_nnz[col_block_id] += 1
    block_index = [i for i in range(len(block_nnz)) \
        if block_nnz[i] >= block_size*block_size*block_density_thres]
    row_block_cols.append(block_index)
    # for nnz in block_nnz:
    #     if nnz >= block_size*block_size*block_density_thres:
    #         ell_nnz = ell_nnz+nnz
    # print(row_block_cols)

    # prepare the ell col ids
    row_block_num = len(row_block_cols)
    print(row_block_num)
    max_blocks = max([len(row_block_cols[i]) for i in range(row_block_num)])
    col_block_num = math.ceil(col_density_thres*max_blocks)
    print(col_block_num)
    for i in range(row_block_num):
        if len(row_block_cols[i]) >= col_block_num:
            row_block_cols[i] = row_block_cols[i][0:col_block_num]
        else:
            for j in range(col_block):
                if j not in row_block_cols[i]:
                    row_block_cols[i].append(j)
                if len(row_block_cols[i]) == col_block_num:
                    break
    print(row_block_cols)

    # split A into ELL and COO
    A_COO = coo_matrix((row_size, col_size))
    ell_val = []
    row_ell_val = []
    for i in range(col_block_num):
        block_val = [0]*block_size*block_size
        row_ell_val.append(block_val)
    cur_row_block = 0
    ell_nnz = 0
    # ell_nnz_new = 0
    for i in range(A.data.size):
        row_block_id = A.row[i]//block_size
        col_block_id = A.col[i]//block_size
        row_id = A.row[i]%block_size
        col_id = A.col[i]%block_size
        if row_block_id == cur_row_block:
            if col_block_id in row_block_cols[row_block_id]:
                cur_col_block = row_block_cols[row_block_id].index(col_block_id)
                row_ell_val[cur_col_block][row_id*block_size+col_id] = A.data[i]
                id1= row_id*block_size+col_id
                # print("(",A.row[i],",",A.col[i],") ")
                # print("(",cur_col_block,",",id1,") ",A.data[i])
                ell_nnz +=1
            else: 
                A_COO.row = np.append(A_COO.row, A.row[i])
                A_COO.col = np.append(A_COO.col, A.col[i])
                A_COO.data = np.append(A_COO.data, A.data[i])
        else:
            ell_val.extend(copy.deepcopy(row_ell_val))
            # print(sum([sum(x) for x in row_ell_val]))
            print(ell_nnz)
            # print(A_COO.nnz)
            # print(len(ell_val))
            # print(sum([sum(x) for x in ell_val[len(ell_val)-col_block_num:len(ell_val)]]))
            # ell_nnz_new = 0
            cur_row_block = row_block_id
            for i in range(col_block_num):
                for j in range(block_size*block_size):
                    row_ell_val[i][j] = 0
            if col_block_id in row_block_cols[row_block_id]:
                cur_col_block = row_block_cols[row_block_id].index(col_block_id)
                row_ell_val[cur_col_block][row_id*block_size+col_id] = A.data[i]
                id1= row_id*block_size+col_id
                ell_nnz +=1
                # print("(",cur_col_block,",",id1,") ",A.data[i])
                # ell_nnz_new+=1
            else: 
                A_COO.row = np.append(A_COO.row, A.row[i])
                A_COO.col = np.append(A_COO.col, A.col[i])
                A_COO.data = np.append(A_COO.data, A.data[i])

    ell_val.extend(copy.deepcopy(row_ell_val))

    A_CSR = A_COO.tocsr().astype(float)
    sparse.save_npz(filename+"_CSR.npz", A_CSR)
    # num_rows = np.array(row_size)
    np.savez(filename+"_ELL.npz", num_rows=row_size,\
            num_cols=col_size,\
            ell_blocksize=block_size,\
            ell_cols=block_size*col_block_num,\
            num_blocks=col_block_num*row_block_num,\
            col_ind=row_block_cols,\
            values=ell_val,\
            nnz=ell_nnz)
    # np.savez(filename+"_ELL.npz", ell_val)
    print(ell_nnz)
    
    print(A_CSR.nnz)

    return col_block_num, row_block_cols, ell_val

if __name__ == "__main__":
    DATASET_PATH = "/work/shared/common/datasets/versatile_sparse_xcel/"
    col_block_num, row_block_cols, ell_val = \
        decompose_CSR_ELL(DATASET_PATH+"email-Eu-core_row_major.mtx", 16, 0.2, 0.7)
    print(sum([sum(x) for x in ell_val]))
    print(col_block_num)
    print(row_block_cols)
    # for file in os.listdir("."):
    #     filename = os.fsdecode(file)
    #     if filename.endswith(".mtx"): 
    #         sparse_plot(filename)
    #         decompose_CSR_ELL(filename, 100, 0.5, 0.7)

