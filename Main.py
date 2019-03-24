#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Paquetes estándar utilizados:
import sys
import cv2
import time
import numpy as np
import json
from websocket import create_connection

# Paquetes propios:
import ObjectRecognition as orec
import Utility as util
import Firebase
import DigitRecognition
import Detector

def connect(local_id):
    while True:
        try:
            ws = create_connection("ws://localhost:8080")
            msg = {
                "type": 'auth',
                "payload": str(local_id)
            }
            msg = json.dumps(msg)
            print(msg)
            ws.send(msg)
            return ws
            break
        except Exception:
            print("exception")
            time.sleep(3)

# Programa principal:
if __name__ == '__main__':
    ws = connect(1)
    
    #Obtenemos las imagenes de OBS
    #videoinput = cv2.VideoCapture(1)
    #videoinput.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)
    #videoinput.set(cv2.CAP_PROP_FRAME_HEIGHT, 1620)

    #Cargamos la base de datos de los modelos (Crear para Ally y Enemy por separado)
    selectedDataBaseMine = orec.loadModelsFromDirectoryMine()
    selectedDataBaseEnemy = orec.loadModelsFromDirectoryEnemy()
    #Cargar base de datos de los buffs
    selectedBuffsDatabase = orec.loadModelsFromDirectoryTemplate()
    # Creación del detector de features, según método (sólo al principio):
    detector = cv2.xfeatures2d.SIFT_create()

    #Crear diccionarios
    JSON_detected = util.JSON_detected
    JSON_detected_copy = util.JSON_detected_copy
    
    #Crear basedMatcher
    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

    #Se importa los 2 detectores de digitos (Uno detecta el simbolo "%"" y el otro no)
    model = cv2.ml.KNearest_create()
    model.train(DigitRecognition.samples, cv2.ml.ROW_SAMPLE, DigitRecognition.responses)

    modelPercent = cv2.ml.KNearest_create()
    modelPercent.train(DigitRecognition.samplesPercent, cv2.ml.ROW_SAMPLE, DigitRecognition.responsesPercent)

    #Crear diccionario de imagenes por recortar
    dict_allySkills_boundaries = util.dict_allySkills_boundaries
    dict_allyHealthMana_boundaries = util.dict_allyHealthMana_boundaries
    dict_allyItems_boundaries = util.dict_allyItems_boundaries
    dict_allyBuffs_boundaries = util.dict_allyBuffs_boundaries
    dict_allyKpAndNameFile = util.dict_allyKpAndNameFile

    frame = cv2.imread("full.png")

    number = 0
    while True:
        start_time = time.time()
        print(str(number) + " Frame")
        #ret, frame = videoinput.read()
        if frame is None:
            print('End of video input')
            break
        # Pasamos imagen de entrada a grises:
        imgin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, threshold_imgin = cv2.threshold(imgin, 125, 255, cv2.THRESH_BINARY)
        retval_1, threshold_imgin_1 = cv2.threshold(imgin, 90, 255, cv2.THRESH_BINARY)
        #print("FrameGrabbed --- %s seconds ---" % (time.time() - start_time))
        #Cambiar dict con items detectados (Cambiar nombres de funciones)
        #print("Ally:")
        #Crear un selectedDatabase exclusivo para los objetos aliados
        #Si es gris poner false, si esta en cd poner pass
        start_time1 = time.time()
        JSON_detected = Detector.ItemDetector(dict_allyItems_boundaries, imgin, selectedDataBaseMine, flann, JSON_detected, detector, "mine", dict_allyKpAndNameFile, frame)
        #print("ItemDetector --- %s seconds ---" % (time.time() - start_time1))
        #Rendimiento eficiente
        start_time2 = time.time()
        JSON_detected = Detector.AllySkillPoints(dict_allySkills_boundaries, threshold_imgin, JSON_detected)
        #print("AllySkillPoints --- %s seconds ---" % (time.time() - start_time2))
        #Rendimiento eficiente
        start_time3 = time.time()
        JSON_detected = Detector.AllyStatsDetector(imgin, JSON_detected, modelPercent, number, threshold_imgin_1)
        #print("AllyStatsDetector --- %s seconds ---" % (time.time() - start_time3))
        #Rendimiento eficiente
        start_time4 = time.time()
        JSON_detected = Detector.AllyHealthDetector(dict_allyHealthMana_boundaries, threshold_imgin, JSON_detected, model, number)
        #print("AllyHealthDetector --- %s seconds ---" % (time.time() - start_time4))
        #Rendimiento eficiente
        start_time5 = time.time()
        JSON_detected = Detector.BuffDetector(imgin, selectedBuffsDatabase, JSON_detected, dict_allyBuffs_boundaries, "mine")
        #print("BuffDetector --- %s seconds ---" % (time.time() - start_time5))
        #print("Enemy:")
        #Medir rendimiento
        start_time6 = time.time()
        JSON_detected = Detector.EnemyDetector(JSON_detected, model, number, modelPercent, imgin, flann, selectedDataBaseEnemy, detector, frame, threshold_imgin, threshold_imgin_1)
        #print("EnemyDetector --- %s seconds ---" % (time.time() - start_time6))
        #Checo si son diferentes
        number += 1
        if JSON_detected != JSON_detected_copy:
            print(JSON_detected)
            for side in JSON_detected:
                for obj in JSON_detected[side]:
                    if obj == "items":
                        for items in JSON_detected[side][obj]:
                            JSON_detected_copy[side][obj][items] = JSON_detected[side][obj][items]
                    else:
                        JSON_detected_copy[side][obj] = JSON_detected[side][obj]   
            msg = {
                "type": 'jsonData',
                "payload": json.dumps(JSON_detected_copy)
            }
            msg = json.dumps(msg)
            try:
                ws.send(msg)
            except Exception:
                time.sleep(3)
                ws = connect(1)
            print("diferentes... mensaje enviado")
        x = 1 - (time.time() - start_time)
        if x <= 0:
            pass
        else:
            time.sleep(x)
        print("LoopEnd --- %s seconds ---" % (time.time() - start_time))
    # Cerrar ventana(s) y fuente(s) de vídeo:
    videoinput.release()
    cv2.destroyAllWindows()