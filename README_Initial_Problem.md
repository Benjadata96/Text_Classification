## Problème

Développer un modèle capable d'identifier le locuteur (Chirac vs. Mitterrand) d'un segment de discours politique.

Un dataset labellisé est fourni pour l'apprentissage

## Data

Les données sont au format Pickle

Les `sequences` sont la version preprocessed des `sentences`. Le dictionnaire produit lors de la tokenization est fourni.

Les données de `Test/` ne sont pas labellisées

## Objectif

Prédire les labels de  `Test` (en respectant le format utilisé dans `Learn`)

