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

home = os.environ['HOME']
shape = (96,144,3) # grilla de 96x144 con 3 canales
X_data_dir = home + "/datos_modelo/X_" + str(0.0) + "Smote.npy"
Y_data_dir = home + "/datos_lluvia/Y_" + str(0.0) + "Smote.npy"
modelos = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
colores = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'grey', 'pink', 'orange', 'palegreen', 'greenyellow']
'''
    Carga de datos y modelo
'''
X = np.load(X_data_dir)
Y = np.load(Y_data_dir)
Y = np.expand_dims(Y,axis=1)
print("------TESTEANDO CON RATIO: " + str(balance_ratio) + "---------------")
print("TOTAL MUESTRAS: " + str(X.shape[0]))
print(X.shape)
print(Y.shape)

'''
    CURVA ROC para varios modelos
'''
plt.figure(figsize=(14,14))
for i in range(len(modelos)):
    model_dir = home + "/modelos/CerroObero/modeloVgg" + str(modelos[i]) + "Smote.h5"
    model = load_model(model_dir)
    y_pred = model.predict(X, verbose=1) # Obtiene las predicciones
    fpr, tpr, thresholds = roc_curve(Y, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC (Receiver Operating Characteristic)')
    plt.plot(fpr, tpr, color=colores[i], lw=1, label='%0.1f ROC curve (area = %0.2f)' % (modelos[i], roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    
plt.legend(loc="lower right")
plt.savefig("ROC_{}.png".format(balance_ratio))

