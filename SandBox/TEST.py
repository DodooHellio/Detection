import cv2

# Charger la vidéo
input_video_path = './Assets/GX010087.MP4'
cap = cv2.VideoCapture(input_video_path)

# Obtenir les informations de la vidéo (largeur, hauteur, etc.)
width = int(cap.get(3))
height = int(cap.get(4))

# Définir la résolution réduite
new_width = int(width * 0.5)  # Vous pouvez ajuster ce facteur selon vos besoins
new_height = int(height * 0.5)

# Définir le chemin pour la nouvelle vidéo
output_video_path = './Assets/GX010087_smallerrez.MP4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Format de compression vidéo
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(5), (new_width, new_height))



while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Redimensionner le cadre
    small_frame = cv2.resize(frame, (new_width, new_height))

    # Votre code de reconnaissance faciale ici (si nécessaire)

    # Écrire le cadre redimensionné dans la nouvelle vidéo
    out.write(small_frame)

    # Afficher le résultat
    cv2.imshow('Frame', small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()