import cv2
import numpy as np

import configFirebase

config = configFirebase.config

json_base = {
    "enemy": {
        "items": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
        },
        "level": False,
        "champion": False,
        "health_bar": 1,
        "currentHealth": 0,
        "currentMana": 0,
        "maxHealth": 0,
        "mana_bar": 1,
        "maxMana": 0,
        "AD": 0,
        "AP": 0,
        "Armor": 0,
        "MR": 0,
        "ASP": 0,
        "CDR": 0,
        "Crit": 0,
        "MS": 0
    },
    "mine": {
        "items": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
        },
        "level": False,
        "health_bar": 1,
        "maxHealth": 0,
        "currentHealth": 0,
        "mana_bar": 1,
        "maxMana": 0,
        "currentMana": 0,
        "skills": [0,0,0,0],
        "AD": 0,
        "AP": 0,
        "Armor": 0,
        "MR": 0,
        "ASP": 0,
        "CDR": 0,
        "Crit": 0,
        "MS": 0
    }
}

json_base1 = {
    "enemy": {
        "items": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
        },
        "level": False,
        "champion": False,
        "health_bar": 1,
        "currentHealth": 0,
        "currentMana": 0,
        "maxHealth": 0,
        "mana_bar": 1,
        "maxMana": 0,
        "AD": 0,
        "AP": 0,
        "Armor": 0,
        "MR": 0,
        "ASP": 0,
        "CDR": 0,
        "Crit": 0,
        "MS": 0
    },
    "mine": {
        "items": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
        },
        "level": False,
        "health_bar": 1,
        "currentHealth": 0,
        "currentMana": 0,
        "maxHealth": 0,
        "mana_bar": 1,
        "maxMana": 0,
        "skills": [0,0,0,0],
        "AD": 0,
        "AP": 0,
        "Armor": 0,
        "MR": 0,
        "ASP": 0,
        "CDR": 0,
        "Crit": 0,
        "MS": 0
    }
}

dict_health_bars = {
    "enemy_health_bar": [[37,38, 202,343], [45,342], [20,340], [27,27,16]],
    "mine_health_bar": [[1027,1028, 683,1095], [1035,1094], [1010,1050], [30,31,15]]
}

health_detector_boundaries = [[1545,1564,1175,1560],[1576,1596,1175,1560]]

dict_skills_boundaries = [
    [1517,1531, 1093,1185],
    [1517,1531, 1193,1285],
    [1517,1531, 1293,1385],
    [1517,1531, 1393,1485]]

dict_allyItems_boundaries = [
    [1422, 1469, 1696, 1760],
    [1422, 1469, 1769, 1833],
    [1422, 1469, 1841, 1905],
    [1491, 1538, 1696, 1760],
    [1491, 1538, 1769, 1833],
    [1491, 1538, 1841, 1905]]

dict_enemyItemsAndChampion_boundaries = [
    [122, 166, 209, 250],
    [122, 166, 254, 295],
    [122, 166, 298, 339],
    [122, 166, 342, 383],
    [122, 166, 386, 427],
    [122, 166, 430, 471],
    [22, 82, 206, 299]]

def enemy_detector(frame, JSON_detected, model, number, modelPercent, imgin, flann, selectedDataBase, detector):
    img_rgb = frame[0:205,0:554]
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('./data/enemyMatcher.png',0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = .9
    loc = np.where( res >= threshold)
    if len(loc[0]) > 0:
        #Barras de vida y mana:
        enemy_HPMana_bars = [[64,79,305,516], [91,106,305,516]]
        enemy_stats = [[23,46,47,105],[23,46,136,194],[61,84,47,105],[61,84,136,194],[101,124,47,105],[101,124,136,194],[140,163,47,105],[140,163,136,194],[87,110,272,293]]
        JSON_detected = extractorDeVida(enemy_HPMana_bars, frame, model, JSON_detected, number, "enemy")
        JSON_detected = extractorDeStats(enemy_stats, img_rgb, modelPercent, JSON_detected, "enemy", number)
        JSON_detected = detector_items_and_enemyChampion(dict_enemyItemsAndChampion_boundaries, imgin, frame, selectedDataBase, flann, JSON_detected, detector, "enemy")
        #for obj, lv, champion in dicts:
        return JSON_detected
    else:
        return JSON_detected

def detector_items_and_enemyChampion(List_boundaries_new, imgin, frame, selectedDataBase, flann, JSON_detected, detector, side):
    for i, boundary in enumerate(List_boundaries_new):
        bestImage = None
        bestIndex = None
        matchesMask = 0
        image = imgin[boundary[0]:boundary[1], boundary[2]:boundary[3]]
        image_color = frame[boundary[0]:boundary[1], boundary[2]:boundary[3]]
        B, G, R = image_color[8,16]
        #Detectamos features, y medimos tiempo:
        kp, desc = detector.detectAndCompute(image, None)
        for index, imageDatabase in enumerate(selectedDataBase):
            kp2 = imageDatabase.kp
            des2 = imageDatabase.desc
            matches = flann.knnMatch(desc,des2,k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            if len(good)>5:
                src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                if mask is not None:
                    matchesMask_this = len(mask.ravel().tolist())
                    if (matchesMask_this > matchesMask):
                        bestIndex = index
                        matchesMask = matchesMask_this
            else:
                pass
        if not bestIndex is None:
            bestImage = selectedDataBase[bestIndex]
            name = bestImage.nameFile.split(".")
            name = name[0].split(",")
            print(name[0])
            if i == 6:
                JSON_detected[side]["champion"] = name[0]
            else:
                JSON_detected[side]["items"][i] = name[0]
    return JSON_detected

def dict_skills(dict_skills, threshold_imgin, JSON_detected):
    for i, skill in enumerate(dict_skills):
        img_count_skills = threshold_imgin[skill[0]:skill[1],skill[2]:skill[3]]
        im2, contours, hierarchy = cv2.findContours(img_count_skills,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        JSON_detected["mine"]["skills"][i] = len(contours)
    return JSON_detected

def health_detector(health_detector_boundaries, frame, JSON_detected, model, number):
    ############################# testing part  #########################
    JSON_detected = extractorDeVida(health_detector_boundaries, frame, model, JSON_detected, number, "mine")
    return JSON_detected

def stats_detector(frame, JSON_detected, model, number):
    im = frame[1418:1613,539:787]
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('./data/statsDetector.png',0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = .9
    loc = np.where( res >= threshold)
    if len(loc[0]) > 0:
        stats_boundaries = [[1435,1461,584,687],[1435,1461,726,787],[1481,1507,584,687],[1481,1507,726,787],[1527,1553,584,687],[1527,1553,726,787],[1573,1599,584,687],[1573,1599,726,787],[1561,1595,925,963]]
        JSON_detected = extractorDeStats(stats_boundaries, frame, model, JSON_detected, "mine", number)
        return JSON_detected
    else:
        return JSON_detected


def extractorDeVida(health_detector_boundaries, frame, model, JSON_detected, number, stringSide):
    for index, bar in enumerate(health_detector_boundaries):
        im = frame[bar[0]:bar[1], bar[2]:bar[3]]
        imCopy = im.copy()
        # Convert BGR to HSV
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        #Convertir a gris y luego obtener contornos
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        retval, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        #Quitar azul brilloso, minimo 50 para no afectar las barras de mana
        lower_green = np.array([80,40,0])
        upper_green = np.array([100,100,255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)
        thresh = cv2.bitwise_and(thresh,thresh,mask=mask_inv)
        #Eliminar colores saturados (Quiza eliminar!)
        lower_green = np.array([0,120,0])
        upper_green = np.array([180,255,255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)
        thresh = cv2.bitwise_and(thresh,thresh,mask=mask_inv)
        #Buscar bounding boxes grandes y dividirlos para que no se peguen los contornos
        images, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            pass
        else:
            hierarchy0 = hierarchy[0]
            for i, cnts in enumerate(contours):
                if hierarchy0[i][3] == -1:
                    [x, y, w, h] = cv2.boundingRect(cnts)
                    if cv2.contourArea(cnts) < 15:
                        cv2.drawContours(thresh, contours, i, (0,0,0), cv2.FILLED)
                    if w <= 5:
                        cv2.drawContours(thresh, contours, i, (0,0,0), cv2.FILLED) 
                    if w >= 20 and 30 >= w:
                        cv2.line(thresh, (int(x+(w/2)),y), (int(x+(w/2)),y+h),(0,0,0),1)
                    elif w >= 35 and 45 >= w:
                        cv2.line(thresh, (int(x+(w/3)),y), (int(x+(w/3)),y+h),(0,0,0),1)
                        cv2.line(thresh, (int(x+(2*(w/3))),y), (int(x+(2*(w/3))),y+h),(0,0,0),1)
                    elif w >= 47 and 57 >= w:
                        cv2.line(thresh, (int(x+(w/4)),y), (int(x+(w/4)),y+h),(0,0,0),1)
                        cv2.line(thresh, (int(x+(2*(w/4))),y), (int(x+(2*(w/4))),y+h),(0,0,0),1)
                        cv2.line(thresh, (int(x+(3*(w/4))),y), (int(x+(3*(w/4))),y+h),(0,0,0),1)
            #Buscar contornos otra vez para clasificarlos
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hierarchy0 = hierarchy[0]
            #Volver a buscar contornos para eliminar los desechados
            if hierarchy is None:
                pass
            else:
                hierarchy0 = hierarchy[0]
                for i, cnts in enumerate(contours):
                    if hierarchy0[i][3] == -1:
                        [x, y, w, h] = cv2.boundingRect(cnts)
                        if cv2.contourArea(cnts) < 15:
                            cv2.drawContours(thresh, contours, i, (0,0,0), cv2.FILLED)
                        if w <= 5:
                            cv2.drawContours(thresh, contours, i, (0,0,0), cv2.FILLED)
            #Rellenar contornos pequeños con negro
            im = cv2.bitwise_and(im,im,mask = thresh)
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hierarchy0 = hierarchy[0]
            #Crear bounding boxes de cada contorno
            boundingBoxes1 = [cv2.boundingRect(c) for c in contours]
            #Combinar los bounding boxes con los contorno, ordenarlas de izquierda a derecha y regresar "cnts" o contornos en orden
            (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes1),
            	key=lambda b:b[1][0], reverse=False))
            #Falta ordenar la jerarquia en el mismo orden que los contornos "cnts", se utilizan los bounding boxes en el orden original
            (hierarchy0, boundingBoxes) = zip(*sorted(zip(hierarchy0, boundingBoxes1),
            	key=lambda b:b[1][0], reverse=False))
            stringResult = ""
            numbers = [0,1,2,3,4,5,6,7,8,9]
            previous_x = 0
            for i, cnt in enumerate(cnts):
                if hierarchy0[i][3] == -1:
                    [x, y, w, h] = cv2.boundingRect(cnt)
                    roi = thresh[y:y + h, x:x + w]
                    roismall = cv2.resize(roi, (10, 10))
                    roismall = roismall.reshape((1, 100))
                    roismall = np.float32(roismall)
                    retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
                    if previous_x == 0:
                        pass
                    else:
                        diff_x = previous_x - x
                        if diff_x < -25:
                            stringResult += "/"
                    previous_x = x
                    if results[0][0] in numbers:
                        stringResult += str(int(results[0][0]))
                        string = str(int((results[0][0])))
                    else:
                        stringResult += "/"
                        string = "/"
            stringResult = stringResult.split("/")
            #print(stringResult)
            if len(stringResult) == 2:
                #print(number)
                if index == 0:
                    health_img = "{}_health_img_{}.png".format(stringSide, number)
                    health_img_full = "{}_health_img_full_{}.png".format(stringSide, number)
                    cv2.imwrite(health_img, im)
                    cv2.imwrite(health_img_full, imCopy)
                    percentage = int(stringResult[0])/int(stringResult[1])
                    percentage = round(percentage,3)
                    JSON_detected[stringSide]["health_bar"] = percentage
                    JSON_detected[stringSide]["currentHealth"] = stringResult[0]
                    JSON_detected[stringSide]["maxHealth"] = stringResult[1]
                elif index == 1:
                    mana_img = "{}_mana_img_{}.png".format(stringSide, number)
                    mana_img_full = "{}_mana_img_full_{}.png".format(stringSide, number)
                    cv2.imwrite(mana_img, im)
                    cv2.imwrite(mana_img_full, imCopy)
                    percentage = int(stringResult[0])/int(stringResult[1])
                    percentage = round(percentage,3)
                    JSON_detected[stringSide]["mana_bar"] = percentage
                    JSON_detected[stringSide]["currentMana"] = stringResult[0]
                    JSON_detected[stringSide]["maxMana"] = stringResult[1]
            else:
                #print(number)
                if index == 0:
                    health_img = "{}_health_img_{}.png".format(stringSide, number)
                    health_img_full = "{}_health_img_full_{}.png".format(stringSide, number)
                    cv2.imwrite(health_img, im)
                    cv2.imwrite(health_img_full, imCopy)
                elif index == 1:
                    mana_img = "{}_mana_img_{}.png".format(stringSide, number)
                    mana_img_full = "{}_mana_img_full_{}.png".format(stringSide, number)
                    cv2.imwrite(mana_img, im)
                    cv2.imwrite(mana_img_full, imCopy)
    return JSON_detected

def extractorDeStats(enemy_stats, frame, model, JSON_detected, side, number):
    for index, bar in enumerate(enemy_stats):
        im = frame[bar[0]:bar[1], bar[2]:bar[3]]
        imCopy = im.copy()
        #Convertir a gris y luego obtener contornos
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        retval, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
        #Buscar bounding boxes grandes y dividirlos para que no se peguen los contornos
        images, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            pass
        else:
            hierarchy0 = hierarchy[0]
            for i, cnts in enumerate(contours):
                if hierarchy0[i][3] == -1:
                    [x, y, w, h] = cv2.boundingRect(cnts)
                    if cv2.contourArea(cnts) < 10:
                        cv2.drawContours(thresh, contours, i, (0,0,0), cv2.FILLED)
                    if w >= 20 and 30 >= w:
                        cv2.line(thresh, (int(x+(w/2)),y), (int(x+(w/2)),y+h),(0,0,0),1)
                    elif w >= 35 and 45 >= w:
                        cv2.line(thresh, (int(x+(w/3)),y), (int(x+(w/3)),y+h),(0,0,0),1)
                        cv2.line(thresh, (int(x+(2*(w/3))),y), (int(x+(2*(w/3))),y+h),(0,0,0),1)
                    elif w >= 47 and 57 >= w:
                        cv2.line(thresh, (int(x+(w/4)),y), (int(x+(w/4)),y+h),(0,0,0),1)
                        cv2.line(thresh, (int(x+(2*(w/4))),y), (int(x+(2*(w/4))),y+h),(0,0,0),1)
                        cv2.line(thresh, (int(x+(3*(w/4))),y), (int(x+(3*(w/4))),y+h),(0,0,0),1)
            #Buscar contornos otra vez para clasificarlos
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hierarchy0 = hierarchy[0]
            #Volver a buscar contornos para eliminar los desechados
            if hierarchy is None:
                pass
            else:
                hierarchy0 = hierarchy[0]
                for i, cnts in enumerate(contours):
                    if hierarchy0[i][3] == -1:
                        [x, y, w, h] = cv2.boundingRect(cnts)
                        if cv2.contourArea(cnts) < 5:
                            cv2.drawContours(thresh, contours, i, (0,0,0), cv2.FILLED)
            #Rellenar contornos pequeños con negro
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hierarchy0 = hierarchy[0]
            #Crear bounding boxes de cada contorno
            boundingBoxes1 = [cv2.boundingRect(c) for c in contours]
            #Combinar los bounding boxes con los contorno, ordenarlas de izquierda a derecha y regresar "cnts" o contornos en orden
            (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes1),
            	key=lambda b:b[1][0], reverse=False))
            #Falta ordenar la jerarquia en el mismo orden que los contornos "cnts", se utilizan los bounding boxes en el orden original
            (hierarchy0, boundingBoxes) = zip(*sorted(zip(hierarchy0, boundingBoxes1),
            	key=lambda b:b[1][0], reverse=False))
            stringResult = ""
            numbers = [0,1,2,3,4,5,6,7,8,9]
            for i, cnt in enumerate(cnts):
                if hierarchy0[i][3] == -1:
                    [x, y, w, h] = cv2.boundingRect(cnt)
                    roi = thresh[y:y + h, x:x + w]
                    roismall = cv2.resize(roi, (10, 10))
                    roismall = roismall.reshape((1, 100))
                    roismall = np.float32(roismall)
                    retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
                    if results[0][0] in numbers:
                        stringResult += str(int(results[0][0]))
                        string = str(int((results[0][0])))
                    elif results[0][0] == 37:
                        stringResult += "%"
                        string = "%"
                    else:
                        stringResult += "/"
                        string = "/"
            #stringResult = stringResult.split("/")
            print(stringResult)
            if index == 0:
                img = "{}_AD_{}.png".format(side, number)
                cv2.imwrite(img, imCopy)
                inted = int(stringResult)
                JSON_detected[side]["AD"] = inted
            elif index == 1:
                img = "{}_AP_{}.png".format(side, number)
                cv2.imwrite(img, imCopy)
                inted = int(stringResult)
                JSON_detected[side]["AP"] = inted
            elif index == 2:
                img = "{}_Armor_{}.png".format(side, number)
                cv2.imwrite(img, imCopy)
                inted = int(stringResult)
                JSON_detected[side]["Armor"] = inted
            elif index == 3:
                img = "{}_MR_{}.png".format(side, number)
                cv2.imwrite(img, imCopy)
                inted = int(stringResult)
                JSON_detected[side]["MR"] = inted
            elif index == 4:
                img = "{}_ASP_{}.png".format(side, number)
                cv2.imwrite(img, imCopy)
                inted = int(stringResult)/100
                JSON_detected[side]["ASP"] = inted
            elif index == 5:
                img = "{}_CDR_{}.png".format(side, number)
                cv2.imwrite(img, imCopy)
                stringResult = stringResult.split("%")
                inted = int(stringResult[0])/100
                JSON_detected[side]["CDR"] = inted
            elif index == 6:
                img = "{}_Crit_{}.png".format(side, number)
                cv2.imwrite(img, imCopy)
                stringResult = stringResult.split("%")
                inted = int(stringResult[0])/100
                JSON_detected[side]["Crit"] = inted
            elif index == 7:
                img = "{}_MS_{}.png".format(side, number)
                cv2.imwrite(img, imCopy)
                inted = int(stringResult)
                JSON_detected[side]["MS"] = inted
            elif index == 8:
                img = "{}_LV_{}.png".format(side, number)
                cv2.imwrite(img, imCopy)
                inted = int(stringResult)
                JSON_detected[side]["level"] = inted
    return JSON_detected