## Reconocimiento Facial 👨👩 | Python - OpenCV
La funcionalidad del aplicativo es el reconocimiento facial de diferentes perfiles (Rostros o Personas), con los metodos utilizados en la libreria OpenCV (EigenFaces, FisherFaces y LBPHFaces).

# Requerimientos
Las siguientes librerias de python son necesarias para la ejecucion del aplicativo:
pip install opencv-python
pip install imutils
pip install numpy
pip install opencv_contrib_python

# Scripts:
analyzerProfileRF.py: Este script analiza perfiles de los rostros a capturar de las personas que deseamos almacenar para proceder a reconocer.
trainingMethodRF.py: Este script entrenamos el reconocedor de rostros con los siguientes: EigenFaces, FisherFaces y LBPHFaces.
detectProfileRF-Video.py: Este script ejecuta el reconocimiento facial de videos.
detectProfileRF-MultiImageWithFrame.py: Este script ejecuta el reconocimiento facial de imagenes con multiples rostros y usando un frame para visualizar resultado, y genera un json con los nombres de los rostros detectados.
detectProfileRF-MultiImageWithoutFrame.py: Este script ejecuta el reconocimiento facial de imagenes con multiples rostros, y genera un json con los nombres de los rostros detectados.

Nota: Los script etectProfileRF-MultiImageWithFrame.py y detectProfileRF-MultiImageWithoutFrame.py necesitan recibir un argumento

#Example Run in python 
python detectProfileRF-ImageWithoutFrame.py test.jpg
python detectProfileRF-ImageWithoutFrame.py C:\Users\mrivera\Desktop\python\MRIRecocimientoFacial\test.jpg