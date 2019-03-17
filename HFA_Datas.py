import pandas as pd
import os

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

      
class Load_Process_Data():
    
    def __init__(self, MAX_LENGTH, VALIDATION_SPLIT):
        self.MAX_LENGTH = MAX_LENGTH
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        
        print('.. Data Loader and Processer initialized ..')
        
    def import_data(self,repo_learn, repo_test): #importation des données
#        Arguments
#                    repo_learn = dossier de training
#                    repo_test = dossier de testing                    
        
        labels = pd.read_pickle(os.path.join(repo_learn,'labels.pkl'))
        sequences_train = pd.read_pickle(os.path.join(repo_learn,'sequences.pkl'))
        
        sequences_test = pd.read_pickle(os.path.join(repo_test,'sequences.pkl'))
        
        print('.. Datas imported ..')
        return (sequences_train, labels, sequences_test)
    
    
    def reshape_sequences(self, sequences): #fixe une longueur identique à toutes les séquences
#        Argument
#                    sequences = séquence à uniformiser
        
        sequences = pad_sequences(sequences, maxlen = self.MAX_LENGTH, padding='post')
        
        return(sequences)
        
    
    def labels_to_integer(self, y): #transforme les labels de 'C' et 'M' à 1 et 0 (respectivement)
#        Argument
#                    y = liste de labels à transformer
        
        df_y = pd.DataFrame({'label':y})
        df_y.label = df_y.label.apply(lambda x : 1 if x=="C" else 0)
        
        y_gold = df_y.label.tolist()
        
        return(y_gold)
    
    
    def main(self, repo_learn, repo_test, is_oversampling, is_undersampling): #prépare les inputs pour le CNN
#        Arguments
#                    repo_learn = dossier de training
#                    repo_test = dossier de testing 
#                    is_oversampling = 1 ou 0 selon si on veut oversampler nos datas
#                    is_undersampling = 1 ou 0 selon si on veut undersampler nos datas
        
        X_train, Y_train, X_test = self.import_data(repo_learn, repo_test)
        
        X_train = self.reshape_sequences(X_train)
        X_test = self.reshape_sequences(X_test)
        Y_train = self.labels_to_integer(Y_train)
        
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                          test_size = self.VALIDATION_SPLIT,
                                                          stratify = Y_train,
                                                          random_state = 13) #stratify permet de garder la même 
                                                                             #distribution qu'initialement
        if is_oversampling ==1 : 
            ros = RandomOverSampler(random_state=0, sampling_strategy = 0.6)
            #sampling_strategy correspond à un ratio de resampling, à quel point on veut plus d'échantillon
            #de la classe initialement minoritaire
            X_train, Y_train = ros.fit_resample(X_train, Y_train)
            
            print('.. Oversampling completed ..')
        
        if is_undersampling == 1 : 
            rus = RandomUnderSampler(random_state=0, sampling_strategy = 0.6)
            X_train, Y_train = rus.fit_resample(X_train, Y_train)
            
            print('.. Undersampling completed ..')
            
        Y_train = to_categorical(Y_train) #reshape nécessaire pour être un output d'un CNN keras
        Y_val = to_categorical(Y_val)
        
        print('.. Inputs are ready to be used ..')
        return(X_train, X_val, X_test, Y_train, Y_val)
            
