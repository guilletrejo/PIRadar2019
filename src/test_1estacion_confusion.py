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
if(dataset_type=='ss'):
    X_data_dir = home + "/datos_modelo/X_" + str(0.0) + "Smote.npy"
    Y_data_dir = home + "/datos_lluvia/Y_" + str(0.0) + "Smote.npy"
else:
    X_data_dir = home + "/datos_modelo/X_" + str(balance_ratio) + "Smote.npy"
    Y_data_dir = home + "/datos_lluvia/Y_" + str(balance_ratio) + "Smote.npy"
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
    Grafico de las probabilidades y calculo del cutoff optimo
'''
snb.set(rc={'figure.figsize':(20,15)})

# Realiza un histograma y lo ajusta con una curva KDE, y de ahi obtiene los valores x,y de la curva
plt_true = snb.distplot(Y_trues, bins=10, norm_hist=False, kde=True, color='blue')
x_true = plt_true.lines[0].get_xdata()
y_true = plt_true.lines[0].get_ydata()
plt_true.cla()
plt_false = snb.distplot(Y_falses, bins=10, norm_hist=False, kde=True, color='red')
x_false = plt_false.lines[0].get_xdata()
y_false = plt_false.lines[0].get_ydata()
plt_false.cla()

# Calcula la menor diferencia para obtener la interseccion de las 2 curvas
min_dif = 1
min_dif2 = 1
for i in range(y_true.size):
    for j in range(y_false.size):
        dif = np.abs(y_true[i] - y_false[j])
        dif2 = np.abs(x_true[i] - x_false[j])
        if(dif < min_dif and dif2 < min_dif2 and x_true[i] > 0 and x_false[j] > 0): 
            min_dif = dif
            min_dif2 = dif2
            best_i, best_j = i, j

cutoff = (x_true[best_i]+x_false[best_j])/2
print("best_cutoff_true: {}".format(x_true[best_i]))
print("best_cutoff_false: {}".format(x_false[best_j]))
# Grafica las curvas
plt_true = snb.distplot(Y_trues, bins=10, norm_hist=False, kde=True, color='blue')
plt_false = snb.distplot(Y_falses, bins=10, norm_hist=False, kde=True, color='red',  axlabel="Prediccion de la red")

# Grafica una linea en el cutoff optimo
maxid_t = best_i
plt.plot(x_true[maxid_t],y_true[maxid_t], '|', ms=2000)
maxid_f = best_j # The id of the peak (maximum of y data)
plt.plot(x_false[maxid_f],y_false[maxid_f], '|', ms=2000)
plt.plot([], [], ' ', label="Balance ratio: {}".format(balance_ratio))
plt.plot([], [], ' ', label="Cutoff óptimo: {0:.3f}".format(cutoff))
plt.legend(loc=9, fontsize='xx-large')

print("CUTOFF OPTIMO: {0:.3f}".format(cutoff))
plt.savefig("Cutoff_{}_{}.png".format(balance_ratio, dataset_type))

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

print("--------------")
print("True Positives: {}".format(TP))
print("True Negatives: {}".format(TN))
print("False Positives: {}".format(FP))
print("False Negatives: {}".format(FN))
print("---------------")
print("Accuracy = {}".format(accuracy))
print("Precision = {}".format(precision))
print("Recall = {}".format(recall))
print("Especificity = {}".format(especificity))
print("Missclassification Rate = {}".format(missclassific_rate))
print("Negative Precision = {}".format(negative_precision))

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
plt.savefig("ROC_{}_{}.png".format(balance_ratio, dataset_type))