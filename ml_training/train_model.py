"""
Script d'entraînement du modèle de classification de fruits
Utilise TensorFlow/Keras avec transfer learning (MobileNetV2)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

class FruitClassifier:
    def __init__(self, data_dir, model_dir):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.train_dir = self.data_dir / "Training"
        self.test_dir = self.data_dir / "Test"
        
        self.model = None
        self.history = None
        self.class_names = None
        
    def prepare_data(self):
        """
        Prépare les générateurs de données avec augmentation
        """
        print("Préparation des données...")
        
        # Générateur pour l'entraînement avec augmentation de données
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=0.2  # 20% pour la validation
        )
        
        # Générateur pour les tests (sans augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Générateur d'entraînement
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        
        # Générateur de validation
        self.validation_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
        
        # Générateur de test
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        # Sauvegarder les noms de classes
        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"Données préparées :")
        print(f"   - {self.train_generator.samples} images d'entraînement")
        print(f"   - {self.validation_generator.samples} images de validation")
        print(f"   - {self.test_generator.samples} images de test")
        print(f"   - {self.num_classes} classes")
        
        # Sauvegarder le mapping des classes
        class_mapping = {i: name for name, i in self.train_generator.class_indices.items()}
        with open(self.model_dir / "class_mapping.json", 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        return True
    
    def create_model(self):
        """
        Crée le modèle avec transfer learning (MobileNetV2)
        """
        print("Création du modèle...")
        
        # Modèle de base pré-entraîné
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        
        # Geler les couches du modèle de base
        base_model.trainable = False
        
        # Ajouter des couches personnalisées
        model = keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compiler le modèle
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        self.model = model
        
        print("Modèle créé :")
        print(f"   - Modèle de base : MobileNetV2")
        print(f"   - Paramètres entraînables : {model.count_params():,}")
        print(f"   - Classes de sortie : {self.num_classes}")
        
        return model
    
    def train(self):
        """
        Entraîne le modèle
        """
        if self.model is None:
            raise ValueError("Modèle non créé. Appelez create_model() d'abord.")
        
        print("Début de l'entraînement...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_dir / "best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entraînement
        self.history = self.model.fit(
            self.train_generator,
            epochs=EPOCHS,
            validation_data=self.validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Entraînement terminé !")
        return self.history
    
    def fine_tune(self, epochs=5):
        """
        Affinage fin du modèle (débloquer les dernières couches)
        """
        print("Début de l'affinage fin...")
        
        # Débloquer les dernières couches du modèle de base
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Geler toutes les couches sauf les dernières
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompiler avec un taux d'apprentissage plus faible
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
        )
        
        # Continuer l'entraînement
        fine_tune_history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            verbose=1
        )
        
        # Combiner les historiques
        for key in self.history.history:
            self.history.history[key].extend(fine_tune_history.history[key])
        
        print("Affinage fin terminé !")
        return fine_tune_history
    
    def evaluate(self):
        """
        Évalue le modèle sur les données de test
        """
        print("Évaluation du modèle...")
        
        # Évaluation
        test_loss, test_accuracy, test_top5 = self.model.evaluate(
            self.test_generator, verbose=0
        )
        
        print(f"Résultats sur les données de test :")
        print(f"   - Précision (Top-1) : {test_accuracy:.4f}")
        print(f"   - Précision (Top-5) : {test_top5:.4f}")
        print(f"   - Perte : {test_loss:.4f}")
        
        return {
            'test_accuracy': test_accuracy,
            'test_top5_accuracy': test_top5,
            'test_loss': test_loss
        }
    
    def save_model(self, filename="fruit_classifier.keras"):
        """
        Sauvegarde le modèle
        """
        model_path = self.model_dir / filename
        self.model.save(model_path)
        print(f"Modèle sauvegardé : {model_path}")
        
        # Sauvegarder aussi au format H5 pour compatibilité
        h5_path = self.model_dir / "fruit_classifier.h5"
        self.model.save(h5_path)
        print(f"Modèle H5 sauvegardé : {h5_path}")
        
        # Sauvegarder au format SavedModel pour TensorFlow Serving
        try:
            savedmodel_path = self.model_dir / "fruit_classifier_savedmodel"
            self.model.export(savedmodel_path)
            print(f"Modèle SavedModel exporté : {savedmodel_path}")
        except Exception as e:
            print(f"WARNING: Export SavedModel échoué : {e}")
        
        return model_path
    
    def plot_training_history(self):
        """
        Affiche les courbes d'entraînement
        """
        if self.history is None:
            print("ERROR: Pas d'historique d'entraînement disponible")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Précision
        ax1.plot(self.history.history['accuracy'], label='Entraînement')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Précision du modèle')
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Précision')
        ax1.legend()
        ax1.grid(True)
        
        # Perte
        ax2.plot(self.history.history['loss'], label='Entraînement')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Perte du modèle')
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('Perte')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Graphiques sauvegardés : {self.model_dir / 'training_history.png'}")

def main():
    """
    Pipeline principal d'entraînement
    """
    print("Entraînement du classificateur de fruits")
    print("=" * 50)
    
    # Chemins
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    model_dir = current_dir / "models"
    
    # Vérifier que les données existent
    if not (data_dir / "Training").exists():
        print("ERROR: Données d'entraînement non trouvées !")
        print("Exécutez d'abord download_data.py pour télécharger le dataset.")
        return
    
    # Créer le classificateur
    classifier = FruitClassifier(data_dir, model_dir)
    
    try:
        # Pipeline d'entraînement
        classifier.prepare_data()
        classifier.create_model()
        classifier.train()
        
        # Évaluation
        results = classifier.evaluate()
        
        # Sauvegarde
        model_path = classifier.save_model()
        
        # Graphiques
        classifier.plot_training_history()
        
        print("\nEntraînement terminé avec succès !")
        print(f"Modèle sauvegardé dans : {model_path}")
        print(f"Précision finale : {results['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"ERROR: Erreur pendant l'entraînement : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
