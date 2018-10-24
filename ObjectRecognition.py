# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:17:13 2015

@author: José María Sola Durán
"""

import cv2
import os
import numpy as np

# Se ha creado una clase Python, llamada ImageFeature
# que contendrá para cada una de las imágenes de la base de datos,
# la información necesaria para computar el reconocimiento de objetos.
class ImageFeatureMine(object):
    def __init__(self, nameFile, desc):
        #Nombre del fichero
        self.nameFile = nameFile
        #KeyPoints de la imagen una vez aplicado el algoritmo de detección de features
        #Descriptores de las features detectadas
        self.desc = desc

#Funcion encargada de calcular, para cada uno de los métodos de calculo de features,
#las features de cada una de las imagenes del directorio "modelos"
def loadModelsFromDirectoryMine():
    #El método devuelve un diccionario. La clave es el algoritmo de features
    #mientras que el valor es una lista con objetos del tipo ImageFeature
    #donde se almacenan todos los datos de las features de las imagenes de la
    #Base de datos.
    dataBase = []
    #Se ha limitado el número de features a 250, para que el algoritmo vaya fluido.
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
    for imageFile in os.listdir("./ModelosMine"):
        imageFile = imageFile#Se carga la imagen con la OpenCV
        colorImage = cv2.imread("./ModelosMine/" + str(imageFile))
        #Pasamos la imagen a escala de grises
        currentImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
        #Realizamos un resize de la imagen, para que la imagen comparada sea igual
        kp, desc = sift.detectAndCompute(currentImage, None)
        #Se cargan las features con SIFT
        dataBase.append(ImageFeatureMine(imageFile, desc))
    return dataBase

class ImageFeatureEnemy(object):
    def __init__(self, nameFile, desc):
        #Nombre del fichero
        self.nameFile = nameFile
        #KeyPoints de la imagen una vez aplicado el algoritmo de detección de features
        #Descriptores de las features detectadas
        self.desc = desc

#Funcion encargada de calcular, para cada uno de los métodos de calculo de features,
#las features de cada una de las imagenes del directorio "modelos"
def loadModelsFromDirectoryEnemy():
    #El método devuelve un diccionario. La clave es el algoritmo de features
    #mientras que el valor es una lista con objetos del tipo ImageFeature
    #donde se almacenan todos los datos de las features de las imagenes de la
    #Base de datos.
    dataBase = []
    #Se ha limitado el número de features a 250, para que el algoritmo vaya fluido.
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
    for imageFile in os.listdir("./ModelosEnemy"):
        imageFile = imageFile#Se carga la imagen con la OpenCV
        colorImage = cv2.imread("./ModelosEnemy/" + str(imageFile))
        #Pasamos la imagen a escala de grises
        currentImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
        #Realizamos un resize de la imagen, para que la imagen comparada sea igual
        kp, desc = sift.detectAndCompute(currentImage, None)
        #Se cargan las features con SIFT
        dataBase.append(ImageFeatureEnemy(imageFile, desc))
    return dataBase

class ImageFeatureTemplate(object):
    def __init__(self, nameFile, grayImage):
        #Nombre del fichero
        self.nameFile = nameFile
        #Shape de la imagen
        self.grayImage = grayImage

def loadModelsFromDirectoryTemplate():
    dataBase = []
    for imageFile in os.listdir("C:/Users/Fernandoo/Downloads/object-recognition-opencv-python-master-a9b184d6cd19eef59de976135d359d3f8d6f9783/p5-objrecon/templates"):
        grayImage = cv2.imread("C:/Users/Fernandoo/Downloads/object-recognition-opencv-python-master-a9b184d6cd19eef59de976135d359d3f8d6f9783/p5-objrecon/templates/" + str(imageFile), 0)
        dataBase.append(ImageFeatureTemplate(imageFile, grayImage))
    return dataBase
