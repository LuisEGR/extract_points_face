# coding=utf-8
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob

parser = argparse.ArgumentParser()
parser.add_argument("source")
args = parser.parse_args()



def shape_to_np(shape):
    # Inicializamos la lista de coordenadas
    coords = np.zeros((68, 2), dtype="int")
    
    # Iteramos sobre las 68 coordenadas de los
    # puntos característicos y las agregamos al arreglo
    # como una tupla, para facilitar la graficación
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
    # Retornamos la lista de coordenadas (x, y)
    return coords

# Inicializamos el detector de rostros
detector_rostros = dlib.get_frontal_face_detector()

# Inicializamos el detector de puntos característicos pre-entrenado
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def extract_landmarks(img_url):
    # Cargamos la imágen
    # img_url = "./id2_6000_swwp2s.mpg_snapshot_00.00.jpg"
    image = cv2.imread(img_url)

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Imagen original')
    # plt.show()

    # Reescalamos la imagen con un ancho de 1000
    image = imutils.resize(image, width=1000)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Imagen original')
    # plt.show()

    # imagen transformada a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # plt.imshow(gray, cmap='gray')
    # plt.title('Escala de grises')
    # plt.show()

    # Detectamos los rostros en la imagen
    rects = detector_rostros(gray, 1)
    # print(rects)


    # Por cada rostro detectado se localizan los puntos característicos
    for (i, rect) in enumerate(rects):
        # Se extraen los puntos característicos y se convierten a un
        # arreglo de NumPy para graficarlo.
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        # print(shape)
    
        # print(shape.tolist())
        np.savetxt(img_url + '.data', shape, delimiter=',', fmt='%d')



files = glob.glob(args.source, recursive=True)

for file in files:
    print("Processing: " + file)
    extract_landmarks(file)