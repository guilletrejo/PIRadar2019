import numpy as np

P = np.load("/home/lac/datos_lluvia/Y_os_all.npy")

M = np.zeros(shape=P.shape)
M[P==-1]=1

np.save("/home/lac/datos_lluvia/Mask_os_all.npy", M)
