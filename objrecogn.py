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
class ImageFeature(object):
    def __init__(self, nameFile, shape, imageBinary, kp, desc):
        #Nombre del fichero
        self.nameFile = nameFile
        #Shape de la imagen
        self.shape = shape
        #Datos binarios de la imagen
        self.imageBinary = imageBinary
        #KeyPoints de la imagen una vez aplicado el algoritmo de detección de features
        self.kp = kp
        #Descriptores de las features detectadas
        self.desc = desc
        #Matchings de la imagen de la base de datos con la imagen de la webcam
        self.matchingWebcam = []
        #Matching de la webcam con la imagen actual de la base de datos.
        self.matchingDatabase = []
    #Permite vaciar los matching calculados con anterioridad, para una nueva imagen
    def clearMatchingMutuos(self):
        self.matchingWebcam = []
        self.matchingDatabase = []

#Funcion encargada de calcular, para cada uno de los métodos de calculo de features,
#las features de cada una de las imagenes del directorio "modelos"
def loadModelsFromDirectory():
    #El método devuelve un diccionario. La clave es el algoritmo de features
    #mientras que el valor es una lista con objetos del tipo ImageFeature
    #donde se almacenan todos los datos de las features de las imagenes de la
    #Base de datos.
    dataBase = dict([('SIFT', [])])
    #Se ha limitado el número de features a 250, para que el algoritmo vaya fluido.
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
    for imageFile in os.listdir("C:/Users/Fernandoo/Downloads/object-recognition-opencv-python-master-a9b184d6cd19eef59de976135d359d3f8d6f9783/p5-objrecon/modelos"):
        imageFile = imageFile#Se carga la imagen con la OpenCV
        colorImage = cv2.imread("C:/Users/Fernandoo/Downloads/object-recognition-opencv-python-master-a9b184d6cd19eef59de976135d359d3f8d6f9783/p5-objrecon/modelos/" + str(imageFile))
        #Pasamos la imagen a escala de grises
        currentImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
        #Realizamos un resize de la imagen, para que la imagen comparada sea igual
        kp, desc = sift.detectAndCompute(currentImage, None)
        #Se cargan las features con SIFT
        dataBase["SIFT"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
    return dataBase
