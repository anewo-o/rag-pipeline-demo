# File: chatos_search.py
#
# Description: Script de test pour la fonctionnalité de chat avec Ollama LLM.
#
import sys
from ollama import Client

# Initialisation du Client
try:
    client = Client(host='http://localhost:11434')
    client.list() 
    print("Connexion à Ollama réussie.")
except Exception as e:
    print(f"Erreur de connexion à Ollama (http://localhost:11434): {e}")
    print("Veuillez vous assurer qu'Ollama est en cours d'exécution.")
    sys.exit(1)

# Fonction Chat
def test_chat():
    """
    Teste la fonctionnalité de chat en envoyant un prompt au LLM.
    """
    print("--- Test de la fonction Chat ---")
    
    # Un prompt pertinent pour votre projet
    prompt = "Génère-moi une commande shell pour créer un dossier nommé 'Projets' dans mon répertoire personnel."
    
    try:
        response = client.chat(model='mistral:7b-instruct', messages=[
          {
            'role': 'user',
            'content': prompt,
          },
        ])
        
        print(f"Prompt envoyé : '{prompt}'")
        print("\nRéponse du LLM :\n", response['message']['content'])
        
    except Exception as e:
        print(f"\nErreur lors du test du chat: {e}")
        print("Veuillez vous assurer que le modèle 'mistral:7b-instruct' est téléchargé.")
        print("Exécutez : 'ollama pull mistral:7b-instruct'")

# Point d'entrée
if __name__ == "__main__":
    test_chat()