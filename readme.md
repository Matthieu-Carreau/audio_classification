# Détection d’activités d’oiseaux dans des enregistrements audio par réseaux de neurones frugaux


Ce dépôt contient le code pour un atelier pratique pour la 5e école technologique du RdEI.

Cet atelier est dans 2 parties :
- 1 - Pré-processing : représentation temps-fréquence
- 2 - Détection par réseau de neurones

Pour chacune des parties, un notebook (.ipynb) et son équivalent en script python (.py) sont disponibles.
Ces fichiers contiennent des lignes à compléter, indiquées par ```...```.
Les solutions se trouvent dans le dossier ```notebooks_complets```.

Pour suivre l'atelier, cloner le dépôt 

```
git clone https://github.com/Matthieu-Carreau/audio_classification.git
```

Créer l'environnement avec conda
```
conda env create -f env_rdei.yml
```

Alternativement, pour installer les dépendances avec pip uniquement : 
```
pip install numpy matplotlib torch torchaudio librosa pandas
```