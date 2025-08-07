"""
Script pour télécharger et préparer le dataset Fruits-360 de Kaggle
"""

import os
import zipfile
import shutil
from pathlib import Path
import pandas as pd

def setup_kaggle():
    """
    Instructions pour configurer Kaggle API
    """
    print("📋 Configuration de Kaggle API")
    print("=" * 50)
    print("1. Créez un compte sur kaggle.com si vous n'en avez pas")
    print("2. Allez dans vos paramètres de compte > API")
    print("3. Cliquez sur 'Create New API Token'")
    print("4. Téléchargez le fichier kaggle.json")
    print("5. Placez-le dans ~/.kaggle/ (créez le dossier si nécessaire)")
    print("6. Définissez les permissions : chmod 600 ~/.kaggle/kaggle.json")
    print("7. Installez kaggle : pip install kaggle")
    print("=" * 50)

def download_dataset(data_dir):
    """
    Télécharge le dataset Fruits-360 depuis Kaggle
    """
    try:
        import kaggle
        print("📥 Téléchargement du dataset Fruits-360...")
        
        # Télécharger le dataset
        kaggle.api.dataset_download_files(
            'moltean/fruits', 
            path=data_dir, 
            unzip=True
        )
        print("✅ Téléchargement terminé !")
        return True
        
    except ImportError:
        print("❌ Kaggle API non installée. Installez avec : pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        print("\n🔧 Solutions possibles :")
        print("- Vérifiez votre configuration Kaggle API")
        print("- Assurez-vous d'avoir accepté les termes du dataset sur Kaggle")
        return False

def manual_download_instructions():
    """
    Instructions pour téléchargement manuel
    """
    print("\n📥 Téléchargement manuel")
    print("=" * 50)
    print("1. Allez sur : https://www.kaggle.com/datasets/moltean/fruits")
    print("2. Cliquez sur 'Download' (vous devez être connecté)")
    print("3. Décompressez le fichier fruits.zip")
    print("4. Copiez le contenu dans : ml_training/data/")
    print("5. La structure doit être :")
    print("   ml_training/data/")
    print("   ├── Training/")
    print("   │   ├── Apple_Braeburn/")
    print("   │   ├── Apple_Golden_1/")
    print("   │   └── ...")
    print("   └── Test/")
    print("       ├── Apple_Braeburn/")
    print("       └── ...")

def analyze_dataset(data_dir):
    """
    Analyse la structure du dataset
    """
    training_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Test")
    
    if not os.path.exists(training_dir):
        print("❌ Dossier Training non trouvé")
        return None
    
    # Compter les classes et images
    classes = []
    total_train_images = 0
    total_test_images = 0
    
    for class_name in os.listdir(training_dir):
        class_path = os.path.join(training_dir, class_name)
        if os.path.isdir(class_path):
            classes.append(class_name)
            train_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_train_images += train_count
            
            # Compter les images de test si le dossier existe
            test_class_path = os.path.join(test_dir, class_name)
            if os.path.exists(test_class_path):
                test_count = len([f for f in os.listdir(test_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_test_images += test_count
    
    print(f"\n📊 Analyse du dataset")
    print("=" * 50)
    print(f"Nombre de classes : {len(classes)}")
    print(f"Images d'entraînement : {total_train_images}")
    print(f"Images de test : {total_test_images}")
    print(f"Total d'images : {total_train_images + total_test_images}")
    
    # Afficher quelques exemples de classes
    print(f"\n🏷️ Exemples de classes :")
    for i, class_name in enumerate(sorted(classes)[:10]):
        print(f"  - {class_name}")
    if len(classes) > 10:
        print(f"  ... et {len(classes) - 10} autres")
    
    return {
        'classes': classes,
        'num_classes': len(classes),
        'train_images': total_train_images,
        'test_images': total_test_images
    }

def create_class_mapping(data_dir, output_file):
    """
    Crée un fichier de mapping des classes
    """
    training_dir = os.path.join(data_dir, "Training")
    
    if not os.path.exists(training_dir):
        return None
    
    classes = sorted([d for d in os.listdir(training_dir) 
                     if os.path.isdir(os.path.join(training_dir, d))])
    
    # Créer le mapping
    class_mapping = {i: class_name for i, class_name in enumerate(classes)}
    
    # Sauvegarder
    import json
    with open(output_file, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"✅ Mapping des classes sauvegardé dans : {output_file}")
    return class_mapping

def main():
    # Chemins
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    
    print("🍎 Préparation du dataset Fruits-360")
    print("=" * 50)
    
    # Créer le dossier data si nécessaire
    data_dir.mkdir(exist_ok=True)
    
    # Vérifier si le dataset existe déjà
    if os.path.exists(data_dir / "Training"):
        print("✅ Dataset déjà présent !")
        stats = analyze_dataset(data_dir)
        if stats:
            create_class_mapping(data_dir, current_dir / "class_mapping.json")
        return
    
    # Tentative de téléchargement automatique
    print("🔄 Tentative de téléchargement automatique...")
    if download_dataset(data_dir):
        stats = analyze_dataset(data_dir)
        if stats:
            create_class_mapping(data_dir, current_dir / "class_mapping.json")
    else:
        print("\n" + "="*50)
        setup_kaggle()
        print("\n" + "="*50)
        manual_download_instructions()
        print("\n" + "="*50)
        print("Après téléchargement manuel, relancez ce script pour l'analyse.")

if __name__ == "__main__":
    main()
