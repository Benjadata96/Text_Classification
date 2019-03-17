import pandas as pd
import os 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore") #permet d'éviter les messages d'alerte de Scikit
                                    #Learn si l'algorithme ne converge pas

class Logistic_Reg():
    
    def __init__(self, CV_FOLD, vectorizer, VALIDATION_SPLIT):
        self.CV_FOLD = CV_FOLD
        self.best_model = None
        self.vectorizer = vectorizer
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        
        
    def import_data(self,repo_learn,repo_test): #importation des données
#        Arguments
#                    repo_learn = dossier de training
#                    repo_test = dossier de testing           
        labels = pd.read_pickle(os.path.join(repo_learn,'labels.pkl'))
        sentences_train = pd.read_pickle(os.path.join(repo_learn,'sentences.pkl'))
            
        sentences_test = pd.read_pickle(os.path.join(repo_test,'sentences.pkl'))
        
        print('.. Datas imported ..')
        return(sentences_train, labels, sentences_test)
        
    def labels_to_integer(self, y): #transforme les labels de 'C' et 'M' à 1 et 0 (respectivement)
#        Argument
#                    y = liste de labels à transformer            
            df_y = pd.DataFrame({'label':y})
            df_y.label = df_y.label.apply(lambda x : 1 if x=="C" else 0)
            
            y_gold = df_y.label.tolist()
            
            return(y_gold)
            
    def training_pipeline(self, x_train, y_train, vectorizer): #implémente et entraîne
                                                               #la régression logistique
        pipe = make_pipeline(vectorizer, LogisticRegression())
        print('.. Pipeline created ..')
        
        param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10],
                      'logisticregression__solver':['lbfgs', 'liblinear', 'sag'],
                      'logisticregression__max_iter':[300]}
        
        grid = GridSearchCV(pipe, param_grid, cv=self.CV_FOLD)
        print('.. Fitting datas ..')
        grid.fit(x_train, y_train)
        
        print("Best cross-validation score: {:.2f}".format(grid.best_score_))
        print("Best parameters: ", grid.best_params_)
        
        self.best_model = grid.best_estimator_        
        
    def testing_model(self, x_test):
        
        y_pred = self.best_model.predict(x_test)
        y_pred = pd.DataFrame(y_pred, columns=['prediction'])
        y_pred.prediction = y_pred.prediction.apply(lambda x: 'C' if x ==1 else 'M')
        y_pred = y_pred.prediction.tolist()
        
        return(y_pred)

    def plotting_ROC(self,x_val,y_val):
        fpr,tpr,thresholds = roc_curve(y_val, self.best_model.decision_function(x_val))
        plt.figure()
        plt.plot(fpr,tpr,label = 'ROC Curve')
        plt.xlabel("FPR")
        plt.ylabel("TPR (recall)")
        plt.legend(loc='best')
        plt.show()

        precision, recall, threshold = precision_recall_curve(y_val, self.best_model.decision_function(x_val))
        plt.plot(precision, recall, label = 'Precision-Recall Curve')
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.legend(loc='best')
        plt.show()

        print('Area Under the Curve : {}'.format(roc_auc_score(y_val, self.best_model.decision_function(x_val))))
            
    def main(self, repo_learn, repo_test):
        
        x_train, y_train, x_test = self.import_data(repo_learn, repo_test)
        
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size = self.VALIDATION_SPLIT,
                                                          random_state = 13) #stratify permet de garder la même 
                                                                             #distribution qu'initialement        
        y_train = self.labels_to_integer(y_train)
        y_val = self.labels_to_integer(y_val)
        
        self.training_pipeline(x_train, y_train, self.vectorizer)
        
        y_pred = self.testing_model(x_test)
        
        return(x_train, x_val, x_test, y_train, y_val, y_pred)
        

