import scipy
from scipy import io
import scipy.sparse as sparse
import sys

# a=sparse.load_npz('roadNet-CA_row_major.npz')
# print(a.shape)

# a = io.mmread('roadNet-PA.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("roadNet-PA_csr_double.npz",b)

# a = io.mmread('web-google.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("web-google_csr_double.npz",b)

# a = io.mmread('amazon0312.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("amazon0312_csr_double.npz",b)

# a = io.mmread('mario002.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("mario002_csr_double.npz",b)

# a = io.mmread('offshore.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("offshore_csr_double.npz",b)

# a = io.mmread('m133-b3.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("m133-b3_csr_double.npz",b)

# a = io.mmread('scircuit.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("scircuit_csr_double.npz",b)

# a = io.mmread('m133-b3.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("m133-b3_csr_double.npz",b)

# a = io.mmread('cage12.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("cage12_csr_double.npz",b)

# a = io.mmread('filter3D.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("filter3D_csr_double.npz",b)

# a = io.mmread('2cubes_sphere.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("2cubes_sphere_csr_double.npz",b)

# a = io.mmread('p2p-Gnutella31.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("p2p-Gnutella31_csr_double.npz",b)

# a = io.mmread('ca-CondMat.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("ca-CondMat_csr_double.npz",b)

# a = io.mmread('poisson3Da.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("poisson3Da_csr_double.npz",b)

# a = io.mmread('wiki-Vote.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("wiki-Vote_csr_double.npz",b)

# a = io.mmread('CollegeMsg.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("CollegeMsg_csr_double.npz",b)

# a = io.mmread('email-Eu-core.mtx')
# b=a.tocsr().astype(double)
# # io.mmwrite('c8_mat11_row_major.mtx', b)
# sparse.save_npz("email-Eu-core_csr_double.npz",b)
# a1 = io.mmread('web-Google_row_major.mtx')
# sparse.save_npz("web-Google_coo_row_major.npz",a1)

# a2 = io.mmread('mario002_row_major.mtx')
# sparse.save_npz("mario002_coo_row_major.npz",a2)

# a3 = io.mmread('amazon0312_row_major.mtx')
# sparse.save_npz("amazon0312_coo_row_major.npz",a3)

# a4 = io.mmread('m133-b3_row_major.mtx')
# sparse.save_npz("m133-b3_coo_row_major.npz",a4)

# a5 = io.mmread('scircuit_row_major.mtx')
# sparse.save_npz("scircuit_coo_row_major.npz",a5)

# a6 = io.mmread('heart1_row_major.mtx')
# sparse.save_npz("heart1_coo_row_major.npz",a6)

# a7 = io.mmread('cari_row_major.mtx')
# sparse.save_npz("cari_coo_row_major.npz",a7)

# a8 = io.mmread('c8_mat11_row_major.mtx')
# sparse.save_npz("c8_mat11_coo_row_major.npz",a8)

# a9 = io.mmread('bibd_18_9_row_major.mtx')
# sparse.save_npz("bibd_18_9_coo_row_major.npz",a9)

# a10 = io.mmread('roadNet-PA_row_major.mtx')
# sparse.save_npz("roadNet-PA_coo_row_major.npz",a10)

# a11 = io.mmread('netherlands_osm.mtx')
# b11=a11.tocsr().astype(float)
# io.mmwrite('netherlands_osm_row_major.mtx', b11)
# sparse.save_npz("netherlands_osm_row_major.npz",b11)
# c11 = io.mmread('netherlands_osm_row_major.mtx')
# sparse.save_npz("netherlands_osm_coo_row_major.npz",c11)

# a12 = io.mmread('asia_osm.mtx')
# b12=a12.tocsr().astype(float)
# io.mmwrite('asia_osm_row_major.mtx', b12)
# sparse.save_npz("asia_osm_row_major.npz",b12)
# c12 = io.mmread('asia_osm_row_major.mtx')
# sparse.save_npz("asia_osm_coo_row_major.npz",c12)

a = io.mmread(sys.argv[1]+'.mtx')
b=a.tocsr().astype(float)
io.mmwrite(sys.argv[1]+'_row_major.mtx', b)
# sparse.save_npz(sys.argv[1]+"_row_major.npz",b)
# c = io.mmread(sys.argv[1]+'_row_major.mtx')
# sparse.save_npz(sys.argv[1]+"_coo_row_major.npz",c)

