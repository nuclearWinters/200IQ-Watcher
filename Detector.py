import cv2
import numpy as np
import time
import Utility as util

def ItemDetector(List_boundaries_new, imgin, selectedDataBase, flann, JSON_detected, detector, side, dict_allyKpAndNameFile, frameRead):
    for i, boundary in enumerate(List_boundaries_new):
        bestDesc = None 
        bestImage = None
        lenGood = 0
        image = imgin[boundary[0]:boundary[1], boundary[2]:boundary[3]]
        image_color = frameRead[boundary[0]:boundary[1], boundary[2]:boundary[3]]
        hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)
        px = hsv[24, 7]
        #print(px)
        if all([
            px[0] == 0,
            px[1] == 0,
            px[2] < 10,
        ]):
            if i == 6:
                JSON_detected[side]["champion"] = False
            else:
                JSON_detected[side]["items"][i] = False
        elif all([
            px[0] <= 150 and px[0] >= 140,
            px[1] <= 200 and px[0] >= 50,
            px[2] <= 200 and px[0] >= 50,
        ]):
            pass
        else:
            #Detectamos features, y medimos tiempo:
            kp, desc = detector.detectAndCompute(image, None)
            if not dict_allyKpAndNameFile[i]["desc"] is None:
                des2 = dict_allyKpAndNameFile[i]["desc"]
                matches = flann.knnMatch(desc,des2,k=2)
                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)
                if len(good)>8:
                    bestImage = dict_allyKpAndNameFile[i]["nameFile"]
                    bestDesk = dict_allyKpAndNameFile[i]["desc"]
            else:
                for index, imageDatabase in enumerate(selectedDataBase):
                    des2 = imageDatabase.desc
                    matches = flann.knnMatch(desc,des2,k=2)
                    good = []
                    for m,n in matches:
                        if m.distance < 0.7*n.distance:
                            good.append(m)
                    if len(good)>8:
                        lenGood_this = len(good)
                        if lenGood_this > lenGood:
                            bestImage = imageDatabase.nameFile
                            bestDesk = imageDatabase.desc
                    else:
                        pass
            if not bestImage is None:
                name = bestImage.split(".")
                name = name[0].split(",")
                if i == 6:
                    JSON_detected[side]["champion"] = int(name[0])
                    dict_allyKpAndNameFile[i]["desc"] = bestDesk
                else:
                    JSON_detected[side]["items"][i] = int(name[0])
                    dict_allyKpAndNameFile[i]["desc"] = bestDesk
    return JSON_detected

def AllySkillPoints(dict_skills, threshold_imgin, JSON_detected):
    for i, skill in enumerate(dict_skills):
        img_count_skills = threshold_imgin[skill[0]:skill[1],skill[2]:skill[3]]
        #stringT = "skillsT" + str(i) + ".png"
        #cv2.imwrite(stringT, img_count_skills)
        im2, contours, hierarchy = cv2.findContours(img_count_skills,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        JSON_detected["mine"]["skills"][i] = len(contours)
    return JSON_detected

def AllyStatsDetector(frame, JSON_detected, model, number, thresthreshold_imgin_1):
    img_gray = frame[1418:1613,539:787]
    template = cv2.imread('./data/statsDetector.png',0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = .9
    loc = np.where( res >= threshold)
    if len(loc[0]) > 0:
        stats_boundaries = [[1435,1461,584,687],[1435,1461,726,787],[1481,1507,584,687],[1481,1507,726,787],[1527,1553,584,687],[1527,1553,726,787],[1573,1599,584,687],[1573,1599,726,787],[1561,1595,925,963]]
        divisions = [[20, 30], [35, 45], [47, 57]]
        JSON_detected = extractorDeStats(stats_boundaries, thresthreshold_imgin_1, model, JSON_detected, "mine", number, divisions)
        return JSON_detected
    else:
        return JSON_detected

def AllyHealthDetector(health_detector_boundaries, frame, JSON_detected, model, number):
    JSON_detected = extractorDeVida(health_detector_boundaries, frame, model, JSON_detected, number, "mine")
    return JSON_detected

def EnemyDetector(JSON_detected, model, number, modelPercent, imgin, flann, selectedDataBase, detector, frameRead, threshold_imgin, threshold_imgin_1):
    img_gray = imgin[168:314,186:503]
    template = cv2.imread('./data/enemyMatcher.png',0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = .9
    loc = np.where( res >= threshold)
    if len(loc[0]) > 0:
        #Barras de vida y mana:
        enemy_HPMana_bars = [[68,83,305,516], [95,111,305,516]]
        enemy_stats = [[23,46,47,105],[23,46,136,194],[61,84,47,105],[61,84,136,194],[101,124,47,105],[101,124,136,194],[140,163,47,105],[140,163,136,194],[87,110,272,293]]
        dict_enemyItemsAndChampion_boundaries = [[122, 166, 209, 250],[122, 166, 254, 295],[122, 166, 298, 339],[122, 166, 342, 383],[122, 166, 386, 427],[122, 166, 430, 471],[22, 82, 206, 299]]
        divisions = [[15, 25], [30, 40], [42, 52]]
        JSON_detected = extractorDeVida(enemy_HPMana_bars, threshold_imgin, model, JSON_detected, number, "enemy")
        JSON_detected = extractorDeStats(enemy_stats, threshold_imgin_1, modelPercent, JSON_detected, "enemy", number, divisions)
        JSON_detected = ItemDetector(dict_enemyItemsAndChampion_boundaries, imgin, selectedDataBase, flann, JSON_detected, detector, "enemy", util.dict_enemyKpAndNameFile, frameRead)
        #for obj, lv, champion in dicts:
        return JSON_detected
    else:
        return JSON_detected

def extractorDeVida(health_detector_boundaries, frame, model, JSON_detected, number, stringSide):
    for index, bar in enumerate(health_detector_boundaries):
        #im = frame[bar[0]:bar[1], bar[2]:bar[3]]
        #imCopy = im.copy()
        # Convert BGR to HSV
        #hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        #Convertir a gris y luego obtener contornos
        #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #retval, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        thresh = frame[bar[0]:bar[1], bar[2]:bar[3]]
        #Quitar azul brilloso, minimo 50 para no afectar las barras de mana
        #lower_green = np.array([80,40,0])
        #upper_green = np.array([100,100,255])
        #mask = cv2.inRange(hsv, lower_green, upper_green)
        #mask_inv = cv2.bitwise_not(mask)
        #thresh = cv2.bitwise_and(thresh,thresh,mask=mask_inv)
        #Eliminar colores saturados (Quiza eliminar!)
        #lower_green = np.array([0,120,0])
        #upper_green = np.array([180,255,255])
        #mask = cv2.inRange(hsv, lower_green, upper_green)
        #mask_inv = cv2.bitwise_not(mask)
        #thresh = cv2.bitwise_and(thresh,thresh,mask=mask_inv)
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
                    if w <= 2:
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
                        if w <= 2:
                            cv2.drawContours(thresh, contours, i, (0,0,0), cv2.FILLED)
            #Rellenar contornos pequeños con negro
            #im = cv2.bitwise_and(im,im,mask = thresh)
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
            #cv2.imwrite("test" + str(index) + str(stringSide) + str(number) + ".png", thresh)
            if len(stringResult) == 2:
                #print(number)
                if index == 0:
                    #cv2.imwrite("testHealth" + str(stringSide) + str(number) + ".png", thresh)
                    if int(stringResult[0]) == 0:
                        JSON_detected[stringSide]["health_bar"] = 0
                    
                    else:
                        percentage = int(stringResult[0])/int(stringResult[1])
                        percentage = round(percentage,3)
                        JSON_detected[stringSide]["health_bar"] = percentage
                    JSON_detected[stringSide]["currentHealth"] = int(stringResult[0])
                    JSON_detected[stringSide]["maxHealth"] = int(stringResult[1])
                elif index == 1:
                    #cv2.imwrite("testMana" + str(stringSide) + str(number) + ".png", thresh)
                    if int(stringResult[0]) == 0:
                        JSON_detected[stringSide]["mana_bar"] = 0
                    else:
                        percentage = int(stringResult[0])/int(stringResult[1])
                        percentage = round(percentage,3)
                        JSON_detected[stringSide]["mana_bar"] = percentage
                    JSON_detected[stringSide]["currentMana"] = int(stringResult[0])
                    JSON_detected[stringSide]["maxMana"] = int(stringResult[1])
            else:
                #print(number)
                if index == 0:
                    pass
                    #health_img = "{}_health_img_{}.png".format(stringSide, number)
                    #health_img_full = "{}_health_img_full_{}.png".format(stringSide, number)
                    #cv2.imwrite(health_img, im)
                    #cv2.imwrite(health_img_full, imCopy)
                elif index == 1:
                    pass
                    #mana_img = "{}_mana_img_{}.png".format(stringSide, number)
                    #mana_img_full = "{}_mana_img_full_{}.png".format(stringSide, number)
                    #cv2.imwrite(mana_img, im)
                    #cv2.imwrite(mana_img_full, imCopy)
    return JSON_detected

def extractorDeStats(enemy_stats, frame, model, JSON_detected, side, number, divisions):
    for index, bar in enumerate(enemy_stats):
        thresh = frame[bar[0]:bar[1], bar[2]:bar[3]]
        #gray = frame[bar[0]:bar[1], bar[2]:bar[3]]
        #imCopy = im.copy()
        #retval, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
        #Buscar bounding boxes grandes y dividirlos para que no se peguen los contornos
        images, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            pass
        else:
            hierarchy0 = hierarchy[0]
            for i, cnts in enumerate(contours):
                if hierarchy0[i][3] == -1:
                    [x, y, w, h] = cv2.boundingRect(cnts)
                    if cv2.contourArea(cnts) < 20:
                        cv2.drawContours(thresh, contours, i, (0,0,0), cv2.FILLED)
                    if w >= divisions[0][0] and divisions[0][1] >= w:
                        cv2.line(thresh, (int(x+(w/2)),y), (int(x+(w/2)),y+h),(0,0,0),1)
                    elif w >= divisions[1][0] and divisions[1][1] >= w:
                        cv2.line(thresh, (int(x+(w/3)),y), (int(x+(w/3)),y+h),(0,0,0),1)
                        cv2.line(thresh, (int(x+(2*(w/3))),y), (int(x+(2*(w/3))),y+h),(0,0,0),1)
                    elif w >= divisions[2][0] and divisions[2][1] >= w:
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
                        if cv2.contourArea(cnts) < 20:
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
            #print(stringResult)
            #cv2.imwrite("test" + str(index) + str(side) + str(number) + ".png", thresh)
            if index == 0:
                #img = "{}_AD1_{}.png".format(side, number)
                #imgThresh = "{}_AD2_{}.png".format(side, number)
                #cv2.imwrite(imgThresh, thresh)
                #cv2.imwrite(img, imCopy)
                try:
                    inted = int(stringResult)
                    JSON_detected[side]["AD"] = inted
                except Exception:
                    pass
            elif index == 1:
                #img = "{}_AP1_{}.png".format(side, number)
                #imgThresh = "{}_AP2_{}.png".format(side, number)
                #cv2.imwrite(imgThresh, thresh)
                #cv2.imwrite(img, imCopy)
                try:
                    inted = int(stringResult)
                    JSON_detected[side]["AP"] = inted
                except Exception:
                    pass
            elif index == 2:
                #img = "{}_Armor1_{}.png".format(side, number)
                #imgThresh = "{}_Armor2_{}.png".format(side, number)
                #cv2.imwrite(imgThresh, thresh)
                #cv2.imwrite(img, imCopy)
                try:
                    inted = int(stringResult)
                    JSON_detected[side]["Armor"] = inted
                except Exception:
                    pass
            elif index == 3:
                #imgThresh = "{}_MR2_{}.png".format(side, number)
                #cv2.imwrite(imgThresh, thresh)
                #print(stringResult)
                try:
                    inted = int(stringResult)
                    JSON_detected[side]["MR"] = inted
                except Exception:
                    pass
            elif index == 4:
                #img = "{}_ASP1_{}.png".format(side, number)
                #imgThresh = "{}_ASP2_{}.png".format(side, number)
                #cv2.imwrite(imgThresh, thresh)
                #cv2.imwrite(img, imCopy)
                try:
                    inted = int(stringResult)/100
                    JSON_detected[side]["ASP"] = inted
                except Exception:
                    pass
            elif index == 5:
                #img = "{}_CDR1_{}.png".format(side, number)
                #imgThresh = "{}_CDR2_{}.png".format(side, number)
                #cv2.imwrite(imgThresh, thresh)
                #cv2.imwrite(img, imCopy)
                try:
                    stringResult = stringResult.split("%")
                    inted = int(stringResult[0])/100
                    JSON_detected[side]["CDR"] = inted
                except Exception:
                    pass
            elif index == 6:
                #img = "{}_Crit1_{}.png".format(side, number)
                #imgThresh = "{}_Crit2_{}.png".format(side, number)
                #cv2.imwrite(imgThresh, thresh)
                #cv2.imwrite(img, imCopy)
                try:
                    stringResult = stringResult.split("%")
                    inted = int(stringResult[0])/100
                    JSON_detected[side]["Crit"] = inted
                except Exception:
                    pass
            elif index == 7:
                #img = "{}_MS1_{}.png".format(side, number)
                #imgThresh = "{}_MS2_{}.png".format(side, number)
                #cv2.imwrite(imgThresh, thresh)
                #cv2.imwrite(img, imCopy)
                try:
                    inted = int(stringResult)
                    JSON_detected[side]["MS"] = inted
                except Exception:
                    pass
            elif index == 8:
                #img = "{}_LV1_{}.png".format(side, number)
                #imgThresh = "{}_LV2_{}.png".format(side, number)
                #cv2.imwrite(imgThresh, thresh)
                #cv2.imwrite(img, imCopy)
                try:
                    inted = int(stringResult)
                    JSON_detected[side]["level"] = inted
                except Exception:
                    pass
    return JSON_detected

def BuffDetector(frame, selectedBuffsDatabase, JSON_detected, Boundaries, side):
    JSON_detected[side]["buffs"] = []
    img_gray = frame[Boundaries[0]:Boundaries[1],Boundaries[2]:Boundaries[3]]
    for index, image in enumerate(selectedBuffsDatabase):   
        res = cv2.matchTemplate(img_gray,image.grayImage,cv2.TM_CCOEFF_NORMED)
        threshold = .7
        loc = np.where( res >= threshold)
        if len(loc[0]) > 0:
            print(image.nameFile)
            JSON_detected[side]["buffs"].append(image.nameFile)
    if not JSON_detected[side]["buffs"]:
        JSON_detected[side]["buffs"].append(False)
    return JSON_detected