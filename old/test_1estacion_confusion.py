from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as snb
import sys

'''
    Parametros
'''
balance_ratio = float(sys.argv[1])
dataset_type = str(sys.argv[2])
if(dataset_type != "cs" and dataset_type != "ss"):
    print("Por favor, ingresar cs o ss (con sinteticos o sin sinteticos)")
    exit()

home = os.environ['HOME']
shape = (96,144,3) # grilla de 96x144 con 3 canales
cutoff = 0.5 # Si el modelo tiene buena separabilidad, 0.5 debería funcionar bien
if(dataset_type=='ss'):
    X_data_dir = home + "/datos_modelo/X_" + str(0.0) + "Smote_nuevos.npy"
    Y_data_dir = home + "/datos_lluvia/Y_" + str(0.0) + "Smote_nuevos.npy"
else:
    X_data_dir = home + "/datos_modelo/X_" + str(1.0) + "Smote_nuevos.npy"
    Y_data_dir = home + "/datos_lluvia/Y_" + str(1.0) + "Smote_nuevos.npy"
model_dir = home + "/modelos/CerroObero/modeloVgg" + str(balance_ratio) + "Smote.h5"
'''
    Carga de datos y modelo
'''
model = load_model(model_dir)
X = np.load(X_data_dir)
Y = np.load(Y_data_dir)
Y = np.expand_dims(Y,axis=1)
print("------TESTEANDO CON RATIO: " + str(balance_ratio) + "---------------")
print("TOTAL MUESTRAS: " + str(X.shape[0]))
print(X.shape)
print(Y.shape)

y_pred = model.predict(X) # Obtiene las predicciones
Y_falses = y_pred[Y==0] # Se queda solo con las predicciones cuyo equivalente en Y_test es 0
Y_trues = y_pred[Y==1]

'''
    Testing (y_true = Y ; y_pred = y_pred)
'''

y_pred[y_pred>=cutoff] = 1
y_pred[y_pred<cutoff] = 0

TN, FP, FN, TP = confusion_matrix(Y,y_pred).ravel()
accuracy = (TP + TN) / (TP+TN+FP+FN)
precision = (TP) / (TP+FP)
recall = (TP) / (TP+FN)
especificity = (TN) / (TN+FP)
missclassific_rate = (FP + FN) / (TP+TN+FP+FN)
negative_precision = (TN) + (TN+FN)

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
print("Accuracy = {}".format(accuracy))
print("Recall = {}".format(recall))

'''
    CURVA ROC
'''
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
plt.savefig("ROC_{}_{}_nuevos.png".format(balance_ratio, dataset_type))