import cv2
import os
import numpy as np

# Variables Generales
dataPath = 'C:/Users/mrivera/Desktop/python/MRIRecocimientoFacial/data/' #Directorio Rostros de Persona Analizada
typeModel = 3                                     # Tipo de Metodo de Entrenamiento (1=EigenFace,2=FisherFace,3=LBPHFace)
peopleList = os.listdir(dataPath)                 # Lista de Directorios del dataPath
nameMethodEigenFace = 'EigenFace'                 # Nombre del Metodo EigenFace
nameMethodFisherFace = 'FisherFace'               # Nombre del Metodo FisherFace
nameMethodLBPHFace = 'LBPHFace'                   # Nombre del Metodo LBPHFace
nameFileMethodEigenFace = 'modeloEigenFace.xml'   # Nombre Archivo a Generar del Metodo EigenFace
nameFileMethodFisherFace = 'modeloFisherFace.xml' # Nombre Archivo a Generar del Metodo FisherFace
nameFileMethodLBPHFace = 'modeloLBPHFace.xml'     # Nombre Archivo a Generar del Metodo LBPHFace
valueEstimateMethodEigenFace = 5700               # Valor de Confianza en Metodo EigenFace
valueEstimateMethodFisherFace = 500               # Valor de Confianza en Metodo FisherFace
valueEstimateMethodLBPHFace = 70                  # Valor de Confianza en Metodo LBPHFace
labelMetodo = 'Metodo Usado: '                    # Etiqueta Metodo
trainingPath = 'training/'                        # Directorio de entrenamiento
message1 = 'Lista de Perfiles: '                  # Mensaje de Consola 1
message2 = 'Leyendo Perfiles...'                  # Mensaje de Consola 2
message3 = 'Entrenando...'                        # Mensaje de Consola 3
message4 = 'Modelo Almacenado...'                 # Mensaje de Consola 4

#Lectura de perfiles de Reconocimiento Facial
def readProfile(type):
	labels = [] #Lista de Etiquetas de Personas
	facesData = [] #Lista de Perfiles
	label = 0 #Etiqueta de la Persona
	print(message2)
	for nameDir in peopleList:
		personPath = f'{dataPath}{nameDir}'

		for fileName in os.listdir(personPath):
			labels.append(label)
			facesData.append(cv2.imread(personPath+'/'+fileName,0))
		label += 1

	# Métodos para entrenar el reconocedor
	if type == 1:
		print(f'{labelMetodo}{nameMethodEigenFace}')
		face_recognizer = cv2.face.EigenFaceRecognizer_create()
		nameModel = nameFileMethodEigenFace
	if type == 2:
		print(f'{labelMetodo}{nameMethodFisherFace}')
		face_recognizer = cv2.face.FisherFaceRecognizer_create()
		nameModel = nameFileMethodFisherFace
	if type == 3:
		print(f'{labelMetodo}{nameMethodLBPHFace}')
		face_recognizer = cv2.face.LBPHFaceRecognizer_create()
		nameModel = nameFileMethodLBPHFace

	# Entrenando el reconocedor de rostros
	print(message3)
	face_recognizer.train(facesData, np.array(labels))

	# Almacenando el modelo obtenido
	face_recognizer.write(f'{trainingPath}{nameModel}')
	print(message4)

#Ejecucion del programa
print(f'{message1}{peopleList}')
readProfile(typeModel)
