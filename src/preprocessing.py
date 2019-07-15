import numpy as np

'''
	Parametros
'''
X1_data_dir = "/home/lac/datos_modelo/z_altura3_2017-11-01.npy"
X2_data_dir = "/home/lac/datos_modelo/z_altura8_2017-11-01.npy"
X3_data_dir = "/home/lac/datos_modelo/z_altura18_2017-11-01.npy"

X11_data_dir = "/home/lac/datos_modelo/z_altura4_2017-11-01.npy"
X21_data_dir = "/home/lac/datos_modelo/z_altura9_2017-11-01.npy"
X31_data_dir = "/home/lac/datos_modelo/z_altura19_2017-11-01.npy"

X12_data_dir = "/home/lac/datos_modelo/z_altura5_2017-11-01.npy"
X22_data_dir = "/home/lac/datos_modelo/z_altura10_2017-11-01.npy"
X32_data_dir = "/home/lac/datos_modelo/z_altura20_2017-11-01.npy"

Y_data_dir = "/home/lac/datos_lluvia/precipitacion.npy"
estacion = 53

'''
	Carga de datos
'''
y = np.load(Y_data_dir).astype(int)
missing = np.where(y[:,estacion]==-1)

X1 = np.delete(np.load(X1_data_dir),missing,0)
X2 = np.delete(np.load(X2_data_dir),missing,0)
X3 = np.delete(np.load(X3_data_dir),missing,0)
X11 = np.delete(np.load(X11_data_dir),missing,0)
X21 = np.delete(np.load(X21_data_dir),missing,0)
X31 = np.delete(np.load(X31_data_dir),missing,0)
X12 = np.delete(np.load(X12_data_dir),missing,0)
X22 = np.delete(np.load(X22_data_dir),missing,0)
X32 = np.delete(np.load(X32_data_dir),missing,0)
'''
	Eliminacion de -1 y seleccion de una sola estacion
'''
y1 = np.delete(y,missing,0)
y = y1[:,53]
#Y = np.expand_dims(Y,axis=1) ## Ver si hace falta hacer esta expansion

'''
	Normalizacion y estandarizacion del input
'''
u1 = np.mean(X1)
u2 = np.mean(X2)
u3 = np.mean(X3)
u11 = np.mean(X11)
u21 = np.mean(X21)
u31 = np.mean(X31)
u12 = np.mean(X12)
u22 = np.mean(X22)
u32 = np.mean(X32)
s1 = np.std(X1)
s2 = np.std(X2)
s3 = np.std(X3)
s11 = np.std(X11)
s21 = np.std(X21)
s31 = np.std(X31)
s12 = np.std(X12)
s22 = np.std(X22)
s32 = np.std(X32)
print("media de X1, X2 y X3: %f, %f, %f" % (u1, u2, u3))
print("dev.est de X1, X2 y X3: %f, %f, %f" % (s1, s2, s3))

X1_scaled = (X1 - u1) / s1
X2_scaled = (X2 - u2) / s2
X3_scaled = (X3 - u3) / s3 
X11_scaled = (X11 - u11) / s11
X21_scaled = (X21 - u21) / s21
X31_scaled = (X31 - u31) / s31
X12_scaled = (X12 - u12) / s12
X22_scaled = (X22 - u22) / s22
X32_scaled = (X32 - u32) / s32 

X1 = np.expand_dims(X1_scaled, axis=3)
X2 = np.expand_dims(X2_scaled, axis=3)
X3 = np.expand_dims(X3_scaled, axis=3)

X11 = np.expand_dims(X11_scaled, axis=3)
X21 = np.expand_dims(X21_scaled, axis=3)
X31 = np.expand_dims(X31_scaled, axis=3)

X12 = np.expand_dims(X12_scaled, axis=3)
X22 = np.expand_dims(X22_scaled, axis=3)
X32 = np.expand_dims(X32_scaled, axis=3)

x = np.concatenate((X1,X2,X3,X11,X21,X32,X12,X22,X32), axis = 3)


'''
	Quedarse con solo 2000 datos para tener 40% de lluvias
'''
indices_lluvias = np.where(y==1)[0]
indices_nolluvias = np.where(y==0)[0][0:1250]
Y = np.concatenate((y[indices_lluvias], y[indices_nolluvias]))
X = np.concatenate((x[indices_lluvias], x[indices_nolluvias]))

idxs = np.arange(X.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

#X = X[idxs, :, :, :]
#Y = Y[idxs]

np.save("/home/lac/datos_modelo/X_3alt_scaled_a.npy", X)
np.save("/home/lac/datos_lluvia/Y_int_Cerro_nonulls_balanced_a.npy", Y)
