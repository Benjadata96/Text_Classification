Plusieurs modèles ont été implémentés pour résoudre le problème : un réseau de convolution et une régression logistique.

Le traitement de données pour le réseau de convolution est implémenté dans le fichier Datas.py
Le réseau de convolution est implémenté dans le fichier CNN.py
La régression logistique ainsi que le traitement de données associé est implémenté dans le fichier Logistic_Regression.py

Les fichiers 'best_weights' représentent les modèles CNN sauvegardés pour les 3 datasets : basique / oversampled / undersampled. 

Il est possible, que ce soit dans le rapport ou dans le code qu'il y ait un mix entre français et anglais, la plupart des termes techniques anglais étant utilisés ainsi en français.

Pour obtenir la solution, il suffit de lancer le script Script_Solution.py

Vous trouverez un fichier texte requirements.txt avec les versions des packages que j'ai utilisé.

# Setup Local
## create virtualenv
Link: <https://github.com/pyenv/pyenv>
    
	$ pyenv virtualenv 3.6.3 Chirac
	$ pyenv activate Chirac
## install libraries
    $ pip install --upgrade -r requirements.txt
## run locally
    $ python Script_Solution.py
## deactivate
    $ pyenv deactivate
