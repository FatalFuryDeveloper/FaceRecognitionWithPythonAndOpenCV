import cv2
import os
import imutils

# Variables Generales
nameFileMethodEigenFace = 'modeloEigenFace.xml'   # Nombre Archivo a Generar del Metodo EigenFace
nameFileMethodFisherFace = 'modeloFisherFace.xml' # Nombre Archivo a Generar del Metodo FisherFace
nameFileMethodLBPHFace = 'modeloLBPHFace.xml'     # Nombre Archivo a Generar del Metodo LBPHFace
valueEstimateMethodEigenFace = 5700               # Valor de Confianza en Metodo EigenFace
valueEstimateMethodFisherFace = 500               # Valor de Confianza en Metodo FisherFace
valueEstimateMethodLBPHFace = 70                  # Valor de Confianza en Metodo LBPHFace
cascadeClassifier = 'haarcascade_frontalface_default.xml' # Nombre de archivo clasificador de RF
dataPath = 'C:/Users/mrivera/Desktop/python/MRIRecocimientoFacial/data/' # Directorio Rostros de Persona Analizada
trainingPath = 'training/'                        # Directorio de entrenamiento
typeModel = 2                                     # Tipo de Metodo de Entrenamiento (1=EigenFace,2=FisherFace,3=LBPHFace)
videoPath = 'video/Julio.mp4'                     # Nombre Video a Analizar
imagePaths = os.listdir(dataPath)                 # Lista de Directorios del dataPath
dimens = 150                                      # Dimension de imagen del rostro ejem: 150x150
nameFrame = 'Detectando Reconocimiento Facial'    # Titulo del Frame
widthFrame = 640                                  # Dimension del tamaño de pantalla del Frame
unknownFace = 'Desconocido'                       # Nombre de Etiqueta a perfiles no reconocidos 

#Lectura de perfiles de Reconocimiento Facial
def faceRecognizer(type):
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

	# Leyendo el modelo
	face_recognizer.read(f'{trainingPath}{nameModel}')
	faceClassif = cv2.CascadeClassifier(f'{cv2.data.haarcascades}{cascadeClassifier}')

	#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
	cap = cv2.VideoCapture(videoPath)

	while True:
		ret,frame = cap.read()
		if ret == False: break
		frame =  imutils.resize(frame, widthFrame)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		auxFrame = gray.copy()
		faces = faceClassif.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(dimens,dimens),interpolation= cv2.INTER_CUBIC)
			result = face_recognizer.predict(rostro)

			cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

			if result[1] < valueEstimate:
				cv2.putText(frame,f'{imagePaths[result[0]]}',(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			else:
				cv2.putText(frame,unknownFace,(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
			
		cv2.imshow(nameFrame,frame)
		k = cv2.waitKey(1)
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

#Ejecucion Reconocimiento Facial
faceRecognizer(typeModel)