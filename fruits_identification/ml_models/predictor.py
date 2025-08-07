"""
Prédicteur de fruits pour l'intégration Django
"""

import os
import json
import numpy as np
from PIL import Image
import logging
from typing import Tuple, List, Dict

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FruitPredictor:
    """
    Classe pour prédire les fruits à partir d'images avec le modèle entraîné
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.img_size = 224  # Taille attendue par MobileNetV2
        
        # Si aucun chemin spécifié, utiliser le modèle entraîné
        if model_path is None:
            # Chemin vers le modèle entraîné
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Remonter de fruits_identification/ml_models vers machine_learning
            machine_learning_dir = os.path.dirname(os.path.dirname(current_dir))
            ml_training_dir = os.path.join(machine_learning_dir, "ml_training")
            self.model_path = os.path.join(ml_training_dir, "models", "best_model.h5")
        
        if self.model_path and os.path.exists(self.model_path):
            self.load_model()
        else:
            logger.warning(f"Modèle non trouvé à : {self.model_path}. Utilisation du mode démo.")
            self._init_demo_mode()
    
    def load_model(self):
        """Charge le modèle TensorFlow"""
        try:
            import tensorflow as tf
            
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Modèle chargé depuis : {self.model_path}")
            
            # Charger le mapping des classes
            model_dir = os.path.dirname(self.model_path)
            class_mapping_path = os.path.join(model_dir, "class_mapping.json")
            
            if os.path.exists(class_mapping_path):
                with open(class_mapping_path, 'r') as f:
                    class_mapping = json.load(f)
                self.class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
                logger.info(f"Classes chargées : {len(self.class_names)} catégories")
            else:
                logger.warning("Fichier de mapping des classes non trouvé")
                
        except ImportError:
            logger.error("TensorFlow non installé. Installez avec : pip install tensorflow")
            self.model = None
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle : {e}")
            self.model = None
    
    def _init_demo_mode(self):
        """Initialise le mode démo avec quelques fruits populaires"""
        self.model = None
        self.class_names = [
            "Apple Braeburn", "Apple Granny Smith", "Apple Red",
            "Banana", "Orange", "Lemon", "Strawberry",
            "Kiwi", "Mango", "Pineapple"
        ]
        logger.info("Mode démo initialisé avec fruits populaires")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Préprocesse une image pour la prédiction
        """
        try:
            # Charger et redimensionner l'image
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = image.resize((self.img_size, self.img_size))
            
            # Convertir en array et normaliser
            img_array = np.array(image)
            img_array = img_array.astype(np.float32) / 255.0
            
            # Ajouter dimension batch
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Erreur lors du préprocessing : {e}")
            raise
    
    def predict(self, image_path: str) -> Dict[str, any]:
        """
        Prédit le fruit dans une image
        """
        try:
            if self.model is None:
                return self._demo_prediction()
            
            # Préprocesser l'image
            processed_image = self.preprocess_image(image_path)
            
            # Faire la prédiction
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            # Trouver la classe prédite
            predicted_class_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_class_idx]) * 100
            
            if len(self.class_names) > predicted_class_idx:
                predicted_class = self.class_names[predicted_class_idx]
            else:
                predicted_class = f"Classe_{predicted_class_idx}"
            
            # Top 3 prédictions
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_3_predictions = [
                {
                    'class': self.class_names[i] if len(self.class_names) > i else f"Classe_{i}",
                    'confidence': float(predictions[i]) * 100
                }
                for i in top_3_indices
            ]
            
            return {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_3': top_3_predictions,
                'model_type': 'tensorflow',
                'is_demo': False
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction : {e}")
            return {
                'success': False,
                'error': str(e),
                'predicted_class': 'Erreur',
                'confidence': 0.0
            }
    
    def _demo_prediction(self) -> Dict[str, any]:
        """Retourne une prédiction de démo"""
        import random
        
        demo_fruits = [
            ('Apple Braeburn', 94.7),
            ('Banana', 89.2),
            ('Orange', 85.6),
            ('Strawberry', 92.3),
            ('Tomato Cherry Red', 87.1),
            ('Grape Blue', 79.8),
            ('Lemon', 83.4),
            ('Avocado', 90.6)
        ]
        
        selected_fruit, confidence = random.choice(demo_fruits)
        
        # Top 3 factice
        other_fruits = [f for f, _ in demo_fruits if f != selected_fruit]
        random.shuffle(other_fruits)
        
        top_3 = [
            {'class': selected_fruit, 'confidence': confidence},
            {'class': other_fruits[0], 'confidence': confidence - random.uniform(10, 20)},
            {'class': other_fruits[1], 'confidence': confidence - random.uniform(20, 30)}
        ]
        
        return {
            'success': True,
            'predicted_class': selected_fruit,
            'confidence': confidence,
            'top_3': top_3,
            'model_type': 'demo',
            'is_demo': True
        }

def get_predictor() -> FruitPredictor:
    """
    Retourne une instance configurée du FruitPredictor
    """
    # Chemin vers le modèle entraîné
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    model_path = os.path.join(project_root, "ml_training", "models", "fruit_classifier.h5")
    
    return FruitPredictor(model_path=model_path)
