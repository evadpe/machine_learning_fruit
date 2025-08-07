#!/usr/bin/env python3
"""
Script pour reformater les fichiers HTML avec une indentation propre
"""

import re
from pathlib import Path

def format_html_manually(file_path):
    """Formate manuellement un fichier HTML simple"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Ajouter des retours à la ligne après certaines balises
        content = re.sub(r'>\s*<', '>\n<', content)
        content = re.sub(r'<(html|head|body|div|form|h1|h2|h3)', r'\n<\1', content)
        content = re.sub(r'</(html|head|body|div|form|h1|h2|h3)>', r'</\1>\n', content)
        content = re.sub(r'{% csrf_token %}', '\n        {% csrf_token %}\n        ', content)
        content = re.sub(r'{% if', r'\n    {% if', content)
        content = re.sub(r'{% endif %}', r'\n    {% endif %}\n', content)
        
        # Nettoyer les espaces multiples
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = content.strip()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Manually formatted: {file_path}")
        
    except Exception as e:
        print(f"Error formatting {file_path}: {e}")

def main():
    """Formate tous les fichiers HTML du projet"""
    base_dir = Path("/home/noahrd0/Documents/projet_groupe_ml/machine_learning")
    html_files = list(base_dir.rglob("*.html"))
    
    for html_file in html_files:
        format_html_manually(html_file)
    
    print(f"Formatted {len(html_files)} HTML files")

if __name__ == "__main__":
    main()
