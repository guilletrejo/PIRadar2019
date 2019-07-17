import numpy as np

P = np.load("/home/awf/datos_lluvia/precipitacion.npy")

M = np.zeros(shape=P.shape)
M[P==-1]=1

np.save("/home/awf/datos_lluvia/mask_precipitacion.npy", M)