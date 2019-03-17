from HFA_Datas import *
from HFA_CNN import *

Data_Loader = Load_Process_Data(450, 0.15)
x_train, x_val, x_test, y_train, y_val = Data_Loader.main('Data/Learn', 'Data/Test',1,0)

model_CNN_over = CNN('model_CNN', 15, 0.01, 'best_weights_over.hdf5') 
model_CNN_over.building_model(30433)
model_CNN_over.plotting_ROC(x_val, y_val)

y_pred_over = model_CNN_over.testing_model(x_test,0)
print('Nombre total de prédictions : {}'.format(len(y_pred_over)))
print('Nombre de prédictions Chirac : {}'.format(y_pred_over.count('C')))
print('%age de prédictions Chirac : {}'.format(round(100*y_pred_over.count('C')/len(y_pred_over),2)))
print('Nombre de prédictions Mitterand : {}'.format(y_pred_over.count('M')))
print('%age de prédictions Mitterand : {}'.format(round(100*y_pred_over.count('M')/len(y_pred_over),2)))

