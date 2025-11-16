# Fichier: chatos_search.py
#
# Description: Script de test pour la fonctionnalité de recherche sémantique 
#
import sys
import os
from ollama import Client
from scipy.spatial.distance import cosine

# Initialisation du Client
try:
    client = Client(host='http://localhost:11434')
    client.list() 
    print("Connexion à Ollama réussie.")
except Exception as e:
    print(f"Erreur de connexion à Ollama (http://localhost:11434): {e}")
    print("Veuillez vous assurer qu'Ollama est en cours d'exécution.")
    sys.exit(1)

# Fonctions de Recherche Sémantique

def embed(text: str):
    """
    Génère un embedding pour un texte donné.
    """
    try:
        response = client.embed(model='embeddinggemma:300m', input=text)
        return response['embeddings'][0]
    except Exception as e:
        print(f"\nErreur lors de la génération de l'embedding pour '{text}': {e}")
        print("Veuillez vous assurer que le modèle 'embeddinggemma:300m' est téléchargé.")
        print("Exécutez : 'ollama pull embeddinggemma:300m'")
        return None

def search(query: str, index: dict):
    """
    Recherche un query dans l'index en utilisant la similarité cosinus.
    """
    q_emb = embed(query)
    if q_emb is None:
        return []
    
    results = [
        (1 - cosine(q_emb, emb), fname)
        for fname, emb in index.items()
        if emb is not None # L'embedding du fichier a réussi
    ]
    return sorted(results, reverse=True)

def test_search():
    """
    Teste la fonctionnalité de recherche en indexant un dossier et en lançant une requête.
    """
    print("\n--- Test de la fonction Recherche ---")
    
    # Indexation 
    print("Indexation des fichiers...")
    index = {}
    
    # Vérification que le dossier 'files' existe au même niveau que le script
    files_dir = os.path.join(os.getcwd(), "files")
    
    if not os.path.isdir(files_dir):
        print(f"Erreur: Le dossier '{files_dir}' n'existe pas.")
        print("Veuillez le créer et y ajouter des fichiers pour le test.")
        return

    files = [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f))]
    
    if not files:
        print(f"Le dossier '{files_dir}' est vide. Veuillez y ajouter des fichiers.")
        return

    for file in files:
        file_embedding = embed(file) 
        if file_embedding:
            index[file] = file_embedding
            
    print(f"Indexation terminée. {len(index)} fichiers indexés.")
    
    if not index:
        print("Aucun fichier n'a pu être indexé. Arrêt du test.")
        return

    # Recherche 
    query = "oiseau" # requête de test
    print(f"\nRecherche pour : '{query}'")
    
    results = search(query, index)
    
    if results:
        print("Résultats:")
        for score, fname in results[:5]:
            print(f"  {fname}: {score:.3f}")
    else:
        print("Aucun résultat trouvé.")


if __name__ == "__main__":
    test_search()