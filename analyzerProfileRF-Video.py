import cv2
import os
import imutils
import numpy as np

# Variables Generales
namePerson = 'Luisito'                              # Nombre de Persona a Analizar
typeRun = 1                                       # Tipo de Ejecucion de Video (Other=Stream Video,1=Video Almacenado)
personPath = 'C:/Users/mrivera/Desktop/python/MRIRecocimientoFacial/data/' + namePerson #Directorio Rostros de Persona Analizada
maxImage = 300                                    # Cantidad Maxima de Imagenes
prefixImage = 'Luisito_'                            # Prefijo del Nombre de la Imagen
typeImage = '.jpg'                                # Tipo de Extension de la imagen
videoPath = 'video/Luisito.mp4'                     # Nombre Video a Analizar
dimens = 150                                      # Dimension de imagen del rostro ejem: 150x150
nameFrame = 'Analizando Perfil de la Persona'     # Titulo del Frame
widthFrame = 640                                  # Dimension del tamaño de pantalla del Frame
cascadeClassifier = 'haarcascade_frontalface_default.xml' # Nombre de archivo clasificador de RF
message0 = 'Carpeta creada: '                     # Mensaje de Consola 1
message1 = 'Inicia Analisis Perfil de Persona'    # Mensaje de Consola 2
message2 = 'Directorio de Perfil: '               # Mensaje de Consola 3
message3 = 'Cantidad de Perfiles: '               # Mensaje de Consola 4

# Crea directorios con el nombre de la persona analizada
def createPath(path):
	if not os.path.exists(personPath):
		print(f'{message0}{personPath}')
		os.makedirs(personPath)

# Crea y almacena el perfil de la persona a analizar
def savedProfile(type):
	if type == 1:
		cap = cv2.VideoCapture(videoPath) #Captura video almacenado
	else:
		cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) #Captura video de camara o stream

	faceClassif = cv2.CascadeClassifier(f'{cv2.data.haarcascades}{cascadeClassifier}')
	count = 0

	while True:
		ret, frame = cap.read()
		if ret == False: break
		frame =  imutils.resize(frame, widthFrame)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		auxFrame = frame.copy()
		faces = faceClassif.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(dimens,dimens),interpolation=cv2.INTER_CUBIC)
			cv2.imwrite(f'{personPath}/{prefixImage}{count}{typeImage}',rostro)
			count += 1
		cv2.imshow(nameFrame,frame)

		k =  cv2.waitKey(1)
		if k == 27 or count >= maxImage:break

	cap.release()
	cv2.destroyAllWindows()

#Ejecucion del programa
print(message1)
createPath(personPath)
savedProfile(typeRun)
print(f'{message2}{personPath}')
print(f'{message3}{maxImage}')
