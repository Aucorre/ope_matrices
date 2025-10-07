import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import random

# ========== Paramètres ==========
IMAGE_PATH = "barbara.jpg"  # Remplacez par le chemin de votre image
PATCH_SIZE = 8           # Taille des patchs (ex: 8x8)
N_PATCHES = 1000          # Nombre de patches à extraire (moins = plus rapide)
MAX_DISPLAY = 5          # Nombre max de formes principales à afficher

def extract_random_patches(image, patch_size, n_patches):
    """Extraire n_patches patchs aléatoires d'une image en niveaux de gris."""
    h, w = image.shape[:2]
    patches = []
    locations = []
    for _ in range(n_patches):
        y = random.randint(0, h - patch_size)
        x = random.randint(0, w - patch_size)
        patch = image[y:y+patch_size, x:x+patch_size]
        patches.append(patch.flatten())
        locations.append((y, x))
    return np.array(patches), locations

def plot_principal_components(pca_result, original_image, patch_size, patch_locations):
    """Afficher les composantes principales sous forme de patchs et montrer un emplacement de patch."""
    n_components = min(MAX_DISPLAY, pca_result.n_components_)
    components = pca_result.components_[:n_components]
    fig, axes = plt.subplots(n_components, 2, figsize=(8, 4 * n_components))
    axes = np.atleast_2d(axes)
    for i in range(n_components):
        component_reshaped = components[i].reshape(patch_size, patch_size)
        axes[i, 0].imshow(component_reshaped, cmap='gray')
        axes[i, 0].set_title(f"Forme principale #{i+1}")
        axes[i, 0].axis("off")
        if i == 0 and patch_locations:
            y, x = patch_locations[0]
            img_with_patch = original_image.copy()
            img_with_patch = cv2.cvtColor(img_with_patch, cv2.COLOR_GRAY2RGB)
            img_with_patch[y:y+patch_size, x:x+patch_size, 0] = 255  # Surlignement rouge
            axes[i, 1].imshow(img_with_patch)
            axes[i, 1].set_title("Emplacement du patch")
            axes[i, 1].axis("off")
        else:
            axes[i, 1].set_visible(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    random.seed(42)  # Pour reproductibilité
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erreur : Image non trouvée. Utilisez un fichier comme 'doggies.jpg' dans le même dossier.")
        exit()
    patches_flat, patch_locations = extract_random_patches(img, PATCH_SIZE, N_PATCHES)
    if len(patches_flat) == 0:
        print("Aucun patch n'a été extrait.")
        exit()
    pca = PCA(n_components=min(MAX_DISPLAY, patches_flat.shape[0]))
    pca.fit(patches_flat)
    print(f"ACP réalisée avec {len(patches_flat)} patchs de taille {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Les {MAX_DISPLAY} premières formes principales (vecteurs propres) sont les plus représentatives des textures locales.")
    plot_principal_components(pca, img, PATCH_SIZE, patch_locations)
