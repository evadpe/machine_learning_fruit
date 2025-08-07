# 🍎 Guide complet : Entraînement et déploiement du modèle de fruits

## 📋 Vue d'ensemble

Ce projet vous permet d'entraîner votre propre modèle de classification de fruits avec le dataset Kaggle Fruits-360 et de l'intégrer dans votre application Django.

## 🚀 Installation et configuration

### 1. Installer les dépendances

```bash
# Depuis le dossier du projet
pip install -r requirements.txt
```

### 2. Configuration Kaggle API (optionnel)

Pour télécharger automatiquement le dataset :

1. Créez un compte sur [kaggle.com](https://kaggle.com)
2. Allez dans Account > API > Create New API Token
3. Téléchargez `kaggle.json`
4. Placez-le dans `~/.kaggle/kaggle.json`
5. Définissez les permissions : `chmod 600 ~/.kaggle/kaggle.json`

## 📊 Entraînement du modèle

### Étape 1 : Télécharger les données

```bash
cd ml_training
python download_data.py
```

**Alternative manuelle :**
1. Allez sur https://www.kaggle.com/datasets/moltean/fruits
2. Téléchargez le dataset
3. Décompressez dans `ml_training/data/`

### Étape 2 : Entraîner le modèle

```bash
python train_model.py
```

**Ce que fait l'entraînement :**
- ✅ Utilise MobileNetV2 avec transfer learning
- ✅ Augmentation de données automatique
- ✅ Validation croisée intégrée
- ✅ Sauvegarde automatique du meilleur modèle
- ✅ Génération de graphiques de performance

**Résultats attendus :**
- Précision : 85-95% selon le dataset
- Temps d'entraînement : 30-60 minutes (CPU/GPU)
- Fichiers générés :
  - `models/fruit_classifier.h5` (modèle principal)
  - `models/class_mapping.json` (mapping des classes)
  - `models/training_history.png` (graphiques)

## 🔧 Intégration dans Django

### Architecture

```
fruits_identification/
├── ml_models/
│   ├── __init__.py
│   └── predictor.py          # Interface de prédiction
├── views.py                  # Vues Django mises à jour
└── templates/
    └── identifier.html       # Template avec résultats ML

ml_training/
├── models/
│   ├── fruit_classifier.h5   # Votre modèle entraîné
│   └── class_mapping.json    # Classes du modèle
├── download_data.py          # Téléchargement dataset
└── train_model.py           # Script d'entraînement
```

### Fonctionnement automatique

1. **Upload d'image** → Formulaire Django avec drag & drop
2. **Préprocessing** → Redimensionnement et normalisation automatiques
3. **Prédiction** → Votre modèle entraîné ou mode démo
4. **Affichage** → Top 3 prédictions avec scores de confiance

## 🎯 Utilisation

### Mode démo (sans modèle)
- Fonctionne immédiatement
- Génère des prédictions aléatoires pour tester l'interface

### Mode production (avec modèle)
- Utilise automatiquement votre modèle entraîné
- Prédictions réelles basées sur vos données

### Test de l'application

```bash
python manage.py runserver
```

Allez sur http://127.0.0.1:8000 et testez l'upload !

## 📊 Performance et optimisation

### Métriques du modèle

Le script d'entraînement fournit :
- **Accuracy** : Précision globale
- **Top-5 Accuracy** : Précision dans le top 5
- **Confusion Matrix** : Détail par classe
- **Training Curves** : Évolution pendant l'entraînement

### Optimisations possibles

1. **Plus de données** :
   - Ajoutez vos propres images
   - Utilisez d'autres datasets complémentaires

2. **Architecture** :
   - Essayez d'autres modèles (ResNet, EfficientNet)
   - Ajustez les hyperparamètres

3. **Déploiement** :
   - Quantification du modèle pour la production
   - Mise en cache des prédictions

## 🐛 Dépannage

### Problèmes courants

**Erreur "No module named 'tensorflow'"**
```bash
pip install tensorflow
```

**Erreur de mémoire pendant l'entraînement**
- Réduisez `BATCH_SIZE` dans `train_model.py`
- Utilisez moins d'époques

**Modèle non trouvé**
- Vérifiez que `ml_training/models/fruit_classifier.h5` existe
- Relancez l'entraînement si nécessaire

**Mauvaise précision**
- Vérifiez la qualité de vos données
- Augmentez le nombre d'époques
- Activez l'affinage fin

### Logs et debugging

Les logs sont affichés dans la console Django :
```python
# Dans settings.py, ajoutez pour plus de logs :
LOGGING = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}
```

## 🎨 Personnalisation

### Ajouter de nouvelles classes

1. Ajoutez vos images dans `ml_training/data/Training/nouvelle_classe/`
2. Relancez l'entraînement
3. Le modèle détectera automatiquement les nouvelles classes

### Modifier l'interface

Les templates Django sont dans `templates/fruits_identification/` :
- `accueil.html` : Page d'accueil
- `identifier.html` : Page de résultats
- `base.html` : Template de base

### Changer le modèle

Dans `ml_training/train_model.py`, modifiez :
```python
# Remplacez MobileNetV2 par un autre modèle
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
```
