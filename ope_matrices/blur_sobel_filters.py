import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger une image
image = cv2.imread('doggies.jpg')  # Remplacez 'image.jpg' par le chemin de votre image
if image is None:
    print("Erreur : Impossible de lire l'image.")
    exit()

# Convertir en mode gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- Filtre floutant (moyenne 3x3) ---
kernel_blur = np.ones((3, 3), np.float32) / 9.0
blurred = cv2.filter2D(gray, -1, kernel_blur)

# --- Détection de contours avec le filtre Sobel ---
sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]], np.float32)
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], np.float32)

grad_x = cv2.filter2D(gray, -1, sobel_x)
grad_y = cv2.filter2D(gray, -1, sobel_y)

# Intensité du gradient (pour les contours)
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Affichage
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title("Image originale (gris)")

plt.subplot(1, 3, 2)
plt.imshow(blurred, cmap='gray')
plt.title("Image floutée")

plt.subplot(1, 3, 3)
plt.imshow(magnitude, cmap='gray')
plt.title("Magnitude des gradients (contours détectés)")

plt.show()
