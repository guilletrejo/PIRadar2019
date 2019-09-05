from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as snb
import sys
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
'''
    Parametros
'''
balance_ratio = 1.0
home = os.environ['HOME']
file = sys.argv[1]
shape = (68,54,3) # grilla de 96x144 con 3 canales
cutoff = 0.5 # Si el modelo tiene buena separabilidad, 0.5 debería funcionar bien
X_data_dir = home + "/datos_modelo/X_Comp.npy"
Y_data_dir = home + "/datos_lluvia/Y_Comp.npy"
model_dir = "{}".format(file)
'''
    Carga de datos y modelo
'''
model = load_model(model_dir)
X = np.load(X_data_dir)
Y = np.load(Y_data_dir)
Y = np.expand_dims(Y,axis=1)
print("--------------TESTEANDO CON DATOS DE COMPARACION---------------")
print("TOTAL MUESTRAS: " + str(X.shape[0]))
print("Dimension matriz entrada: " + str(X.shape))
print("Dimension matriz salida: " + str(Y.shape))

'''
    Contar cuantos 1 hay en total en la estacion.
'''
lluvias = np.where(Y==1)[0].size
nolluvias = np.where(Y==0)[0].size
print("Cant. de horas con lluvia: " + str(lluvias))
print("Cant. de horas sin lluvia: " + str(nolluvias))
total_real = lluvias+nolluvias
print("Total de datos: " + str(total_real))
print("Porcentaje de horas con lluvia: " + "{0:.2f}".format(lluvias/total_real))
print("Porcentaje de horas sin lluvia: " + "{0:.2f}".format(nolluvias/total_real))

y_pred = model.predict(X) # Obtiene las predicciones
Y_falses = y_pred[Y==0] # Se queda solo con las predicciones cuyo equivalente en Y_test es 0
Y_trues = y_pred[Y==1]

'''
    Testing (y_true = Y ; y_pred = y_pred)
'''
print(y_pred)
y_pred[y_pred>=cutoff] = 1
y_pred[y_pred<cutoff] = 0

TN, FP, FN, TP = confusion_matrix(Y,y_pred).ravel()
accuracy = (TP + TN) / (TP+TN+FP+FN)
precision = (TP) / (TP+FP)
recall = (TP) / (TP+FN)
especificity = (TN) / (TN+FP)
missclassific_rate = (FP + FN) / (TP+TN+FP+FN)
negative_precision = (TN) + (TN+FN)
porcentaje_unos = (TP+FP) / (TP+TN+FP+FN)
print("---------------")
print("True Positives: {}".format(TP))
print("True Negatives: {}".format(TN))
print("False Positives: {}".format(FP))
print("False Negatives: {}".format(FN))
print("---------------")
print("Missclassification Rate = {}".format(missclassific_rate))
print("Negative Precision = {}".format(negative_precision))
print("Precision = {}".format(precision))
print("Especificity = {}".format(especificity))
print("Porcentaje unos = {}".format(porcentaje_unos))
print("Accuracy = {}".format(accuracy))
print("Recall = {}".format(recall))

'''
    CURVA ROC

fpr, tpr, thresholds = roc_curve(Y, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(14,14))
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC (Receiver Operating Characteristic)')
plt.legend(loc="lower right")
plt.savefig("ROC_TrainVal.png")
'''
