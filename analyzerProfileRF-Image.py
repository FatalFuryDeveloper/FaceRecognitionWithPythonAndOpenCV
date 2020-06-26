import cv2
import os
import imutils
import numpy as np

# Variables Generales
namePerson = 'Chica'                              # Nombre de Persona a Analizar
typeRun = 1                                       # Tipo de Ejecucion de Video (Other=Stream Video,1=Video Almacenado)
personPath = 'C:/Users/mrivera/Desktop/python/MRIRecocimientoFacial/data/' + namePerson #Directorio Rostros de Persona Analizada
maxImage = 300                                    # Cantidad Maxima de Imagenes
prefixImage = 'Chica_'                            # Prefijo del Nombre de la Imagen
typeImage = '.jpg'                                # Tipo de Extension de la imagen
imagePath = 'test/9.jpg'                     # Nombre Video a Analizar
dimens = 150                                      # Dimension de imagen del rostro ejem: 150x150
nameFrame = 'Analizando Perfil de la Persona'     # Titulo del Frame
widthFrame = 600                                  # Dimension del tamaño de pantalla del Frame
cascadeClassifier = 'libs/haarcascade_frontalface_alt.xml' # Nombre de archivo clasificador de RF
defaultCascadeClassifier = 'haarcascade_frontalface_default.xml' # Nombre de archivo clasificador de RF
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
def savedProfile(image):
	#Cargamos la imagen y la convertimos a grises:
	img = cv2.imread(image)
	img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	 
	#Buscamos los rostros:
	#faceClassif1 = cv2.CascadeClassifier(cascadeClassifier)
	faceClassif = cv2.CascadeClassifier(f'{cv2.data.haarcascades}{defaultCascadeClassifier}')
	faces = faceClassif.detectMultiScale(img_gris, 1.3, 5) 
	count = 0
	auxFrame = img.copy()

	#Ahora recorremos los rostros 'faces' se dibuja rectángulos sobre la imagen original y se reconoce el rostro:
	for (x,y,w,h) in faces:	
		cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(dimens,dimens),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(f'{personPath}/{prefixImage}{count}{typeImage}',rostro)
		count += 1
	img2 =  imutils.resize(img, widthFrame)

	#Abrimos una ventana con el resultado:
	cv2.imshow(nameFrame,img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#Ejecucion del programa
print(message1)
createPath(personPath)
savedProfile(imagePath)
print(f'{message2}{personPath}')
print(f'{message3}{maxImage}')
