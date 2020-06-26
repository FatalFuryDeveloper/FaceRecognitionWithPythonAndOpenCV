# Reconocimiento Facial 👨👩 | Python - OpenCV
La funcionalidad del aplicativo es el reconocimiento facial de diferentes perfiles (Rostros o Personas), con los metodos utilizados en la libreria OpenCV (EigenFaces, FisherFaces y LBPHFaces).

## Demo
* Reconocimiento Facial en Videos con Python y OpenCV: https://youtu.be/5OneeUnM_n8 
* Reconocimiento Facial MultiRostros en Python con OpenCV con Frame: https://youtu.be/PUibIf3Fnks
* Reconocimiento Facial MultiRostros en Python con OpenCV sin Frame: https://youtu.be/YPrVNGnLmFU

[![ScreenShot](https://github.com/FatalFuryDeveloper/FaceRecognitionWithPythonAndOpenCV/blob/master/demo/main.jpg)](https://youtu.be/5OneeUnM_n8)

## Requerimientos
Las siguientes librerias de python son necesarias para la ejecucion del aplicativo:
* pip install **opencv-python**
* pip install **imutils**
* pip install **numpy**
* pip install **opencv_contrib_python**

## Scripts:
|Nombre                                     |Descripcion |
|-------------------------------------------|------------|
|analyzerProfileRF-Image.py                 |Este script analiza perfiles de los rostros a capturar de las personas que deseamos almacenar para proceder a reconocer. |
|analyzerProfileRF-Video.py                 |Este script entrenamos el reconocedor de rostros con los siguientes: EigenFaces, FisherFaces y LBPHFaces. |
|detectProfileRF-Video.py                   |Este script ejecuta el reconocimiento facial de videos. |
|detectProfileRF-MultiImageWithFrame.py     |Este script ejecuta el reconocimiento facial de imagenes con multiples rostros y usando un frame para visualizar resultado, y genera un json con los nombres de los rostros detectados. |
|detectProfileRF-MultiImageWithoutFrame.py  |Este script ejecuta el reconocimiento facial de imagenes con multiples rostros, y genera un json con los nombres de los rostros detectados. |
|test.py                                    |Este script se realizaba para pruebas entre videos e imagenes
|trainingMethodRF.py                        |Este script entrena un modelo de OpenCV (EigenFaces, FisherFaces y LBPHFaces).

**Nota:** Los script **detectProfileRF-MultiImageWithFrame.py** y **detectProfileRF-MultiImageWithoutFrame.py** necesitan recibir un argumento

## Example Run in python 
* python detectProfileRF-ImageWithoutFrame.py test.jpg
* python detectProfileRF-ImageWithoutFrame.py C:\Users\mrivera\Desktop\python\MRIRecocimientoFacial\test.jpg