"""
    Reconocimiento Facial Multimagen .- Este programa encuentra todos los rostros de una fotografía.
    Desarrollado por Mauro Rivera
"""
import cv2
import os
import imutils
import json
import sys

# Variables Generales
nameFileMethodEigenFace = 'modeloEigenFace.xml'   # Nombre Archivo a Generar del Metodo EigenFace
nameFileMethodFisherFace = 'modeloFisherFace.xml' # Nombre Archivo a Generar del Metodo FisherFace
nameFileMethodLBPHFace = 'modeloLBPHFace.xml'     # Nombre Archivo a Generar del Metodo LBPHFace
valueEstimateMethodEigenFace = 5700               # Valor de Confianza en Metodo EigenFace
valueEstimateMethodFisherFace = 500               # Valor de Confianza en Metodo FisherFace
valueEstimateMethodLBPHFace = 70                  # Valor de Confianza en Metodo LBPHFace
cascadeClassifier = 'libs/haarcascade_frontalface_alt.xml' # Nombre de archivo clasificador de RF
dataPath = 'C:/Users/mrivera/Desktop/python/MRIRecocimientoFacial/data/' # Directorio Rostros de Persona Analizada
trainingPath = 'training/'                        # Directorio de entrenamiento
typeModel = 2                                     # Tipo de Metodo de Entrenamiento (1=EigenFace,2=FisherFace,3=LBPHFace)
videoPath = 'video/Julio.mp4'                     # Nombre Video a Analizar
imagePaths = os.listdir(dataPath)                 # Lista de Directorios del dataPath
dimens = 150                                      # Dimension de imagen del rostro ejem: 150x150
nameFrame = 'Detectando Reconocimiento Facial'    # Titulo del Frame
widthFrame = 640                                  # Dimension del tamaño de pantalla del Frame
unknownFace = 'Desconocido'                       # Nombre de Etiqueta a perfiles no reconocidos
fileNameOutputJson = 'output/detectFacial.json'   # Nombre de archivo JSON 
labelFirtsLevelJson = 'persons'                   # Etiqueta Primer Nivel Estructura JSON
labelSecondLevelJson = 'name'                     # Etiqueta Segundo Nivel Estructura JSON
data = {}                                         # Informacion a almacenar en Json
data[labelFirtsLevelJson] = []                    # Estructura Json

#Lectura de perfiles de Reconocimiento Facial
def faceRecognizer(type, image):
	if type == 1:
		face_recognizer = cv2.face.EigenFaceRecognizer_create()
		nameModel = nameFileMethodEigenFace
		valueEstimate = valueEstimateMethodEigenFace 
	if type == 2:
		face_recognizer = cv2.face.FisherFaceRecognizer_create()
		nameModel = nameFileMethodFisherFace
		valueEstimate = valueEstimateMethodFisherFace 
	if type == 3:
		face_recognizer = cv2.face.LBPHFaceRecognizer_create()
		nameModel = nameFileMethodLBPHFace
		valueEstimate = valueEstimateMethodLBPHFace 

	#Cargamos nuestro classificador de Haar:
	face_recognizer.read(f'{trainingPath}{nameModel}')
	faceClassif = cv2.CascadeClassifier(cascadeClassifier)	 
	 
	#Cargamos la imagen y la convertimos a grises:
	img = cv2.imread(image)
	img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	 
	#Buscamos los rostros:
	faces = faceClassif.detectMultiScale(img_gris, 1.3, 5)
	  
	#Ahora recorremos los rostros 'faces' se dibuja rectángulos sobre la imagen original y se reconoce el rostro:
	for (x,y,w,h) in faces:
		auxFrame = img_gris.copy()
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(dimens,dimens),interpolation= cv2.INTER_CUBIC)
		result = face_recognizer.predict(rostro)
		
		if result[1] < valueEstimate:
			data[labelFirtsLevelJson].append({labelSecondLevelJson: f'{imagePaths[result[0]]}'})

	cv2.destroyAllWindows()

#Crea archivo Salida Estructura Json 
def savedJson():
	with open(fileNameOutputJson, 'w') as file:
	    json.dump(data, file, indent=4)

#Ejecucion Reconocimiento Facial
if len(sys.argv) > 1:
	if os.path.exists(sys.argv[1]):
		faceRecognizer(typeModel,sys.argv[1])
		savedJson()
	else:
		print(f'Error: El directorio de la imagen a analizar no existe')	
else:
	print(f'Error: Este programa necesita un parámetro: El directorio de la imagen a analizar')