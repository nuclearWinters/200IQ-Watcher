#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Paquetes estándar utilizados:
import sys
import cv2
import time
import numpy as np
import pyrebase

# Paquetes propios:
import objrecogn as orec
import utility as util

# Programa principal:
if __name__ == '__main__':

    config = util.config
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    auth = firebase.auth()
    user = auth.sign_in_with_email_and_password("armandonarcizoruedaperez@gmail.com", "armando123")
    local_id = user["localId"]
    
    videoinput = cv2.VideoCapture(1)
    videoinput.set(cv2.CAP_PROP_FRAME_WIDTH, 1960)
    videoinput.set(cv2.CAP_PROP_FRAME_HEIGHT, 1620)
    #Cargamos la base de datos de los modelos
    dataBaseDictionary = orec.loadModelsFromDirectory()
    # Creación del detector de features, según método (sólo al principio):
    detector = cv2.xfeatures2d.SIFT_create()

    #Crear diccionarios
    JSON_detected = util.json_base
    JSON_detected_copy = util.json_base1

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    selectedDataBase = dataBaseDictionary["SIFT"]

    #Crear diccionario de imagenes por recortar
    dict_health_bars = util.dict_health_bars
    dict_skills = util.dict_skills_boundaries

    #######   training part    ###############
    samples = np.loadtxt('./data/generalsamples.data', np.float32)
    responses = np.loadtxt('./data/generalresponses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    samplesPercent = np.loadtxt('./data/generalsamplesPercent.data', np.float32)
    responsesPercent = np.loadtxt('./data/generalresponsesPercent.data', np.float32)
    responsesPercent = responsesPercent.reshape((responsesPercent.size, 1))

    modelPercent = cv2.ml.KNearest_create()
    modelPercent.train(samplesPercent, cv2.ml.ROW_SAMPLE, responsesPercent)

    health_detector_boundaries = util.health_detector_boundaries
    dict_allyItems_boundaries = util.dict_allyItems_boundaries

    number = 0
    while True:
        ret, frame = videoinput.read()
        if frame is None:
            print('End of video input')
            break
        # Pasamos imagen de entrada a grises:
        print(number)
        imgin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, threshold_imgin = cv2.threshold(imgin, 130, 255, cv2.THRESH_BINARY)
        #Cambiar dict con items detectados
        print("Ally:")
        JSON_detected = util.detector_items_and_enemyChampion(dict_allyItems_boundaries, imgin, frame, selectedDataBase, flann, JSON_detected, detector, "mine")
        JSON_detected = util.dict_skills(dict_skills, threshold_imgin, JSON_detected)
        JSON_detected = util.stats_detector(frame, JSON_detected, modelPercent, number)
        JSON_detected = util.health_detector(health_detector_boundaries, frame, JSON_detected, model, number)
        print("Enemy:")
        JSON_detected = util.enemy_detector(frame, JSON_detected, model, number, modelPercent, imgin, flann, selectedDataBase, detector)
        #Checo si son diferentes
        number += 1
        if JSON_detected != JSON_detected_copy:
            for side in JSON_detected:
                for obj in JSON_detected[side]:
                    if obj == "items":
                        for items in JSON_detected[side][obj]:
                            JSON_detected_copy[side][obj][items] = JSON_detected[side][obj][items]
                    else:
                        JSON_detected_copy[side][obj] = JSON_detected[side][obj]   
            results = db.child("users").child(local_id).child("inGame").update(JSON_detected_copy, user['idToken'])
            print(JSON_detected_copy)
            print("diferentes... mensaje enviado")
        time.sleep(1)
    # Cerrar ventana(s) y fuente(s) de vídeo:
    videoinput.release()
    cv2.destroyAllWindows()