import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import Input,Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Concatenate, Embedding
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

class CNN():
    
    def __init__(self, model_name, epochs, lrate, weights_filepath):
        self.name = model_name
        self.model = None
        self.filters_size = [3,4,5]
        self.shape = (450,)
        self.epochs = epochs
        self.lrate = lrate
        self.decay = self.lrate / self.epochs
        self.weights_filepath = weights_filepath
        
        print('.. CNN Loader initialized ..')

    
    def building_model(self, vocab_size): #construction du modèle
#        Argument
#                        vocab_size = taille du vocabulaire permettant de définir la couche d'embedding
        
        input_layer = Input(shape = self.shape)
        
        embedding_layer = Embedding(vocab_size, 100, input_length = 450, trainable = True)
        embedded_input = embedding_layer(input_layer)
        embedded_input = Dropout(0.5)(embedded_input)
                
        conv_layers = []   
        for fsz in self.filters_size: #Pour chaque valeur du filtre, on va créer 100 'features map'
            conv = Conv1D(100, fsz, input_shape=self.shape, padding = 'valid', 
                          activation = 'relu')(embedded_input)
            dropout = Dropout(0.5)(conv)
            max_pool = MaxPooling1D(pool_size = 450-fsz+1, padding='valid')(dropout)
            conv_layers.append(max_pool)
        print('.. conv layers implemented ..')
        
        merged_layer = Concatenate(axis=1)(conv_layers) #concaténation des 3 convolutions : 100 features map chacunes
        flattened_layer = Flatten()(merged_layer) #transformation des 'features map' en 1 vecteur 'aplati' (flattened)
        flattened_layer = Dropout(0.5)(flattened_layer)
        print('.. flattened layers implemented ..')
        
        dense_layer = Dense(128, activation='relu')(flattened_layer) #réseau MultiLayerPerceptron à 1 couche qui sert 
                                                                     #de classifieur
        print('.. first dense layer implemented ..')
        dense_layer = Dropout(0.5)(dense_layer)
        print('.. second dense layer implemented ..')
        
        final_layer = Dense(2, activation = 'softmax')(dense_layer) #couche finale qui sépare les 2 classes
        
        self.model = Model(input_layer, final_layer)
        
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print ('.. model is compiled and ready to be trained ..')
        
        print(self.model.summary()) 
        return(self.model)
     
    def training_model(self, X_train, X_val, Y_train, Y_val, vocab_size): #training du modèle
#        Arguments
#                        vocab_size = taille du vocabulaire permettant de définir la couche d'embedding
#                        X_train / Y_train = input / output de training
#                        X_val/Y_val = input / output de validation
                        
        self.building_model(vocab_size) #construction du modèle
        
        checkpoint = ModelCheckpoint(self.weights_filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        #permet de sauvegarder les poids du meilleur modèle en terme de valeur minimale de la loss sur la validation
        callback = [Metrics()] #permet d'ajouter les metrics : F1score, recall, precision
        callbacks_final = callbacks_list + callback

        print('.. training model ..')
        self.model.fit(X_train, Y_train, batch_size=100, epochs=self.epochs,verbose = 1, shuffle = True, 
                       validation_data=(X_val,Y_val), callbacks=callbacks_final)
        print('.. model is trained ..')
        
        # Cette partie de code sert uniquement si on souhaite plot l'accuracy et la loss pendant
        # le training. Néanmoins, on ne fait que 15epochs donc pas nécessaire ici.

        # print('.. compiling accuracy and loss ..')
        # training_acc = trained_model.history['acc']
        # validation_acc = trained_model.history['val_acc']
        # training_loss = trained_model.history['loss']
        # validation_loss = trained_model.history['val_loss']
        
        # print('.. plotting accuracy and loss ..')
        # epochs = range(len(training_acc))
        # plt.plot(epochs, training_acc, 'bo', label = 'Training Accuracy')
        # plt.plot(epochs, validation_acc, 'b', label = 'Validation Acurracy')
        # plt.title('Training and Validation Acurracy')
        # plt.legend()
        # plt.figure()
        
        # plt.plot(epochs, training_loss, 'bo', label = 'Training Loss')
        # plt.plot(epochs, validation_loss, 'b', label = 'Validation Loss')
        # plt.title('Training and Validation Loss')
        # plt.legend()
        # plt.show()
        # print ('.. accuracy and loss printed ..')
        
    def testing_model(self, X_test, is_ROC): #test du modèle sauvegardé
#        Arguments
#                        X_test = dataset de test
#                        is_ROC = 1 ou 0 selon si on test le modèle sur la validation

            self.model.load_weights(self.weights_filepath)
            Y_test = self.model.predict(X_test)   
            
            if is_ROC == 1: 
                Y_test = Y_test[:,1] #bout de code permettant de sortir les probabilités 
                #de la classe positive (Chirac) pour tracer la ROC
            else:
                Y_test = pd.DataFrame(np.argmax(Y_test, axis=1), columns=['prediction'])
                Y_test.prediction = Y_test.prediction.apply(lambda x: 'C' if x ==1 else 'M')
                Y_test = Y_test.prediction.tolist()

            return(Y_test)
    def plotting_ROC(self,x_val, y_val): #plot les courbes de ROC et precision-recall
#        Arguments
#                        X_val/Y_val = input / output de validation

        y_pred_ROC = self.testing_model(x_val,1)

        y_val_ROC = np.argmax(y_val, axis=1)

        fpr,tpr,thresholds = roc_curve(y_val_ROC, y_pred_ROC)
        plt.figure()
        plt.plot(fpr,tpr,label = 'ROC Curve')
        plt.xlabel("FPR")
        plt.ylabel("TPR (recall)")
        plt.legend(loc='best')
        plt.show()

        precision, recall, threshold = precision_recall_curve(y_val_ROC, y_pred_ROC)
        plt.plot(precision, recall, label = 'Precision-Recall Curve')
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.legend(loc='best')
        plt.show()

        print('Area Under the Curve : {}'.format(roc_auc_score(y_val_ROC, y_pred_ROC)))
                
    def main(self, X_train, X_test, X_val, Y_train, Y_val, vocab_size): #script du modèle
        
        self.training_model(X_train, X_val,Y_train, Y_val,vocab_size)

        Y_pred = self.testing_model(X_test,0)
        
        return(Y_pred)
        
class Metrics(Callback): #Classe qui permet de définir F1score, recall & precision
    
    def on_train_begin(self, logs={}):
     self.val_f1s = []
     self.val_recalls = []
     self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
     
     val_predict = self.model.predict(self.validation_data[0])
     val_predict = np.argmax(val_predict, axis=1)
     
     val_targ = self.validation_data[1]
     val_targ = np.argmax(val_targ, axis=1)

     _val_f1 = f1_score(val_targ, val_predict)
     _val_recall = recall_score(val_targ, val_predict)
     _val_precision = precision_score(val_targ, val_predict)

     self.val_f1s.append(_val_f1)
     self.val_recalls.append(_val_recall)
     self.val_precisions.append(_val_precision)

     print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))

 
