from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from .ml_models.predictor import get_predictor

# Create your views here.

def accueil(request):
    """Vue pour la page d'accueil de l'identification de fruits"""
    fruits_identifiables = [
        # Pommes (Apples)
        'Apple', 'Apple Braeburn', 'Apple Core', 'Apple Crimson Snow', 'Apple Golden', 
        'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red', 'Apple Red Delicious', 
        'Apple Red Yellow', 'Apple Rotten', 'Apple Hit', 'Apple Worm',
        
        # Fruits à noyau
        'Apricot', 'Cherry', 'Cherry Rainier', 'Cherry Sour', 'Cherry Wax Black',
        'Cherry Wax Red', 'Cherry Wax Yellow', 'Cherry Wax (not ripe)', 'Peach', 'Peach Flat',
        'Plum', 'Nectarine', 'Nectarine Flat',
        
        # Fruits tropicaux
        'Avocado', 'Avocado Black', 'Avocado Green', 'Avocado Ripe', 'Banana', 
        'Banana Lady Finger', 'Banana Red', 'Mango', 'Mango Red', 'Pineapple', 
        'Pineapple Mini', 'Papaya', 'Passion Fruit', 'Kiwi', 'Guava',
        
        # Agrumes
        'Orange', 'Lemon', 'Lemon Meyer', 'Lime', 'Grapefruit Pink', 'Grapefruit White',
        'Mandarine', 'Clementine', 'Tangelo', 'Kumquat',
        
        # Baies
        'Strawberry', 'Strawberry Wedge', 'Blackberry', 'Blackberry (half ripe)', 
        'Blackberry (not ripe)', 'Blueberry', 'Raspberry', 'Redcurrant', 'Gooseberry',
        'Huckleberry', 'Mulberry',
        
        # Poires
        'Pear', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster',
        'Pear Red', 'Pear Stone', 'Pear Williams',
        
        # Raisins
        'Grape Blue', 'Grape Pink', 'Grape White',
        
        # Melons
        'Cantaloupe', 'Watermelon', 'Melon Piel de Sapo',
        
        # Légumes fruits
        'Tomato', 'Tomato Cherry Maroon', 'Tomato Cherry Orange', 'Tomato Cherry Red',
        'Tomato Cherry Yellow', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow',
        'Tomato (not ripe)', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow',
        'Eggplant', 'Eggplant Long', 'Cucumber', 'Cucumber Ripe', 'Zucchini', 'Zucchini Dark',
        
        # Légumes racines
        'Carrot', 'Beetroot', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White',
        'Onion Red', 'Onion Red Peeled', 'Onion White', 'Ginger Root',
        
        # Légumes feuilles et fleurs
        'Cabbage Red', 'Cabbage White', 'Cauliflower', 'Kohlrabi',
        
        # Légumineuses et céréales
        'Beans', 'Corn', 'Corn Husk',
        
        # Noix et graines
        'Nut', 'Nut Forest', 'Nut Pecan', 'Chestnut', 'Hazelnut', 'Pistachio', 'Walnut',
        'Caju Seed',
        
        # Fruits exotiques
        'Cactus Fruit', 'Cactus Fruit Green', 'Cactus Fruit Red', 'Carambula', 'Cherimoya',
        'Cocos', 'Dates', 'Fig', 'Granadilla', 'Lychee', 'Mangostan', 'Maracuja',
        'Pepino', 'Physalis', 'Physalis with Husk', 'Pitahaya Red', 'Pomegranate',
        'Pomelo Sweetie', 'Quince', 'Rambutan', 'Salak', 'Tamarillo'
    ]
    
    context = {
        'titre': 'Identification de Fruits',
        'message': 'Bienvenue sur notre système d\'identification de fruits par machine learning!',
        'fruits': fruits_identifiables,
        'nombre_fruits': len(fruits_identifiables)
    }
    return render(request, 'fruits_identification/accueil.html', context)

def identifier_fruit(request):
    """Vue pour identifier un fruit"""
    print(f"DEBUG: Method: {request.method}")
    print(f"DEBUG: Content-Type: {request.META.get('CONTENT_TYPE', 'Not set')}")
    
    if request.method == 'POST':
        print(f"DEBUG: POST Request received")
        print(f"DEBUG: request.FILES keys: {list(request.FILES.keys())}")
        print(f"DEBUG: request.POST keys: {list(request.POST.keys())}")
        print(f"DEBUG: Raw request.FILES: {request.FILES}")
        
        # Vérifier si le fichier existe dans la requête
        if 'photo_fruit' not in request.FILES:
            print("ERROR: 'photo_fruit' not found in request.FILES")
            context = {
                'titre': 'Identification de Fruit - Erreur',
                'message': 'Aucun fichier détecté dans la requête.',
                'erreur': True,
                'erreur_message': 'Fichier manquant'
            }
            return render(request, 'fruits_identification/identifier.html', context)
        
        try:
            # Récupération du fichier uploadé
            photo = request.FILES['photo_fruit']
            print(f"DEBUG: Fichier reçu : {photo.name}, taille: {photo.size} bytes")
            
            # Validation du type de fichier
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            file_extension = os.path.splitext(photo.name)[1].lower()
            print(f"DEBUG: Extension : {file_extension}")
            
            if file_extension not in allowed_extensions:
                raise ValueError(f"Type de fichier non supporté : {file_extension}")
            
            # Sauvegarde temporaire du fichier
            file_name = f"temp_{photo.name}"
            file_path = default_storage.save(f"uploads/{file_name}", ContentFile(photo.read()))
            full_file_path = default_storage.path(file_path)
            print(f"DEBUG: Fichier sauvé : {full_file_path}")
            
            # Vérifier que le fichier existe
            if not os.path.exists(full_file_path):
                raise FileNotFoundError(f"Fichier non trouvé : {full_file_path}")
            
            # Initialisation du prédicteur
            print("DEBUG: Chargement du prédicteur...")
            predictor = get_predictor()
            print(f"DEBUG: Prédicteur chargé : {predictor.model is not None}")
            
            # Prédiction
            print("DEBUG: Début de la prédiction...")
            prediction_result = predictor.predict(full_file_path)
            print(f"DEBUG: Résultat : {prediction_result}")
            
            # Nettoyage du fichier temporaire
            if default_storage.exists(file_path):
                default_storage.delete(file_path)
                print("DEBUG: Fichier temporaire supprimé")
            
            # Préparation du contexte pour l'affichage
            context = {
                'titre': 'Identification de Fruit - Résultat',
                'message': f'Analyse de "{photo.name}" terminée !',
                'photo_name': photo.name,
                'photo_size': f"{photo.size / 1024:.1f} KB",
                'resultat': True,
                'prediction': prediction_result
            }
            
        except Exception as e:
            # Gestion des erreurs avec plus de détails
            print(f"ERROR: Erreur dans identifier_fruit : {e}")
            import traceback
            traceback.print_exc()
            
            context = {
                'titre': 'Identification de Fruit - Erreur',
                'message': 'Une erreur est survenue lors de l\'analyse.',
                'erreur': True,
                'erreur_message': str(e),
                'erreur_details': traceback.format_exc()
            }
        
        return render(request, 'fruits_identification/identifier.html', context)
    
    # Affichage du formulaire
    context = {
        'titre': 'Identifier un Fruit',
        'message': 'Téléchargez une photo de fruit ou légume pour l\'identifier',
        'formulaire': True
    }
    return render(request, 'fruits_identification/identifier.html', context)