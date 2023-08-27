# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 21:00:38 2021
車牌辨識系統
@author: 10820109 10827231 10827238

"""

import sys
import cv2
import numpy as np
from skimage import measure

def cut_img(f):
    contours, h = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    num = []
    for (i, cnt) in enumerate(contours) :
        count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        num.append([x, y, f[y:y + h, x:x + w]])
    return num

def sort_img(outputname, num, index1=0, index2=1):
    num.sort(key=lambda s: (s[index1], s[index2]))
    for i in range(len(num)):
        cv2.imwrite(f'{outputname}{i}.jpg', num[i][2])

def CClabeling_filter(f, lower=0, upper=20000):
    openingimg = f.copy()
    labels = measure.label(openingimg, background=0)
    mask = np.zeros(openingimg.shape, dtype="uint8")
    label_size = 0
    for (i, label) in enumerate(np.unique(labels)):

        # 如果label=0，表示它為背景
        if label == 0: pass
            # print("[INFO] label: 0(background)")
        else:
            labelMask = np.zeros(openingimg.shape, np.uint8)
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            if numPixels > lower and numPixels < upper:
                label_size += 1
                # 放到剛剛建立的空圖中
                mask = cv2.add(mask, labelMask)

    return mask, label_size


if len(sys.argv) == 2:
    img_n = sys.argv[1]
    # 讀取彩色的圖片
    img = cv2.imread(img_n)
    # cv2.imshow("test",img)
    # cv2.waitKey(0)
    find = False
    img = cv2.resize(img, [480,480], cv2.INTER_LINEAR)
    if img is not None:

        # thy = round(img.shape[0] * 0.45)
        # thh = round(img.shape[0] * 0.85)
        # thx = img.shape[1] // 3
        # img = img[thy:thh, thx:thx * 2]

        # 轉換為灰度圖
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("test",img1)
        # cv2.waitKey(0)

        # 用Sobel進行邊緣檢測
        # # 高斯模糊
        img2 = cv2.GaussianBlur(img1, (5, 5), 10)
        # cv2.imshow("test",img2)
        # cv2.waitKey(0)

        # Laplacian進行邊緣檢測
        img3 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=1)
        # cv2.imshow("test",img3)
        # cv2.waitKey(0)
        img4 = cv2.Canny(img3, 110, 100)
        # cv2.imshow("test",img4)
        # cv2.waitKey(0)

        # 進行二值化處理
        i, img5 = cv2.threshold(img4, 0, 255, cv2.THRESH_BINARY)
        # cv2.imshow("test",img5)
        # cv2.waitKey(0)

        # 可以侵蝕和擴張
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 8))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
        img_d1 = cv2.dilate(img5, kernel1, 0)  # 膨脹
        img_e1 = cv2.erode(img_d1, kernel2, 4)   # 侵蝕

        out_img, label_size = CClabeling_filter(img_e1, lower=1000, upper=8000)

        # cv2.imshow("test",out_img)
        # cv2.waitKey(0)

        # # # 迴圈找到所有的輪廓
        i, j = cv2.findContours(out_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result = None
        c = 0
        for i1 in i:
            x, y, w, h = cv2.boundingRect(i1)
            ## car license size =
            # ( 32 * 15 )舊版 w/h = 2.133
            # ( 38 * 16 )新版 w/h = 2.375
            print( 'h = {}, w = {}, ratio = {}'.format( h, w, w/h ) )
            ratio = w/h
            # if ratio > 2.0 and ratio < 2.5:
            #     c += 1
            lincense_loc = img[y:y + h, x:x + w]
            cv2.imshow("test", lincense_loc)
            cv2.waitKey(0)
        find = True


    else:
        print("Can't found the license.")

else:
    print("Requir an image.")

if find == True :
    # 固定定每張車牌大小 ==============================================================
    rc = 200
    rr = 550
    area = rc * rr
    inimg = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY) #以灰階讀取
    inimg = cv2.resize( inimg, [rr,rc], cv2.INTER_LINEAR )
    #cv2.imshow("img", inimg )
    #===============================================================================


    # Black-Hat 加強車牌字體輪廓======================================================
    se = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (50,50) )
    top = cv2.morphologyEx( inimg, cv2.MORPH_BLACKHAT, se )
    #===============================================================================

    # 增加每個pixel的強度  ===========================================================
    c, r = top.shape[:2]
    #cv2.imshow("inimg", top )
    for x in range( c ) :
        for y in range( r ) :
            if( top[x,y] > 60) :
                pixel = top[x,y] * 2
                if( pixel > 255 ) : top[x,y] = 255
                else : top[x,y] = pixel
    #===============================================================================


    # 形態學處裡=====================================================================
    #cv2.imshow("top", top )
    ret, threshold = cv2.threshold(top, 127, 255, cv2.THRESH_BINARY ) # 二值圖
    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (5,10) )
    threshold = cv2.morphologyEx( threshold, cv2.MORPH_OPEN, kernel ) # 開運算 把連在一起的字斷開
    kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (5,5) )
    threshold = cv2.morphologyEx( threshold, cv2.MORPH_OPEN, kernel ) # 把多餘的雜訊和字體斷開
    kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (2,2) )
    threshold = cv2.morphologyEx( threshold, cv2.MORPH_CLOSE, kernel ) # 閉運算 把弄斷的字補好
    #cv2.imshow("thresh", threshold)

    # 利用CClabeling把每個label找出來，並用"面積"分類出有可能是字體的label
    lower = 400 # label pixel的下限
    upper = threshold.shape[0]*threshold.shape[1]/12 # label pixel的上限
    size = 0
    cc = 0 # 預防無窮迴圈的計數器
    while ( size < 7 and cc < 10 ): # 至少要標出七個符合的label
        out, size = CClabeling_filter(threshold, lower, upper)
        # 只留下 lower < img_pixel < upper 的 label
        # 逐漸縮小範圍找出更準確的label
        cc += 1
        lower += 50
        upper -= 10

    # 找出目前所剩餘的label的contours(輪廓)
    # 利用"輪廓的大小和長寬比"來過濾出是車牌字體的label
    contours, _ = cv2.findContours( out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    for (i, cnt) in enumerate( contours ) :
        x, y, w, h = cv2.boundingRect( cnt )
        if ( w/h >= 0.6 or cv2.contourArea(cnt) < 1000 or cv2.contourArea(cnt) > 10000  ):
            # 符合上述條件的代表不是正確的字體 將他移除
            out[y:y+h, x:x+w ] = 0

    #cv2.imshow("fff", out)
    #cv2.waitKey(0)
    #=======================================================================================

    # 將車牌上面的字體依照label的位置切下來=====================================================
    num = []
    num = cut_img(out)
    outputname = "num"
    sort_img(outputname, num) # 依照 Y 座標排序 (由左至右)

    # 切模板字體 並排序 ( 左上到右下 )
    temp = cv2.imread("../DIP10820109/template_fonts.bmp", 0)
    if temp is not None:
        ret, thresh = cv2.threshold(temp, 250, 255, cv2.THRESH_BINARY)
        label = []
        label = cut_img(thresh)
        #print(len(label))
        outputname = "temp"
        sort_img(outputname, label, 1, 0)
        #print(len(label))

        table = [[0, 'A'], [1, 'B'], [2, 'C'], [3, 'D'], [4, 'E'], [5, 'F'], [6, 'G'],
                 [7, 'H'], [8, 'I'], [9, 'J'], [10, 'L'], [11, 'K'], [12, 'M'], [13, 'N'],
                 [14, 'O'], [15, 'P'], [16, 'R'], [17, 'T'], [18, 'S'], [19, 'Q'], [20, 'U'],
                 [21, 'V'], [22, 'W'], [23, 'X'], [24, 'Y'], [25, 'Z'], [26, '3'], [27, '1'],
                 [28, '2'], [29, '4'], [30, '5'], [31, '6'], [32, '7'], [33, '8'], [34, '9'],
                 [35, '0']]
        #========================================================================================

        # 將車牌下的字體和模板的字體做比較============================================================
        # map -> 切下來的車牌有多少個 pixel 的 Level 是 255 (白色)
        # checkmap -> 模板和車牌有多少個 pixel 的 Level 是相同的
        # checkmap/map * 100 相似度百分比
        ans = ""
        c = 0 # 計數器
        n = 0
        while n < len(num) :  # for letter
            same = []
            NUM = False
            if c == 3 : break # 前三個字體是英文
            for t in range(26):
                relabel = cv2.resize( label[t][2], [num[n][2].shape[1], num[n][2].shape[0]])
                checkmap = 0
                map = 0
                for y in range(num[n][2].shape[0]):
                    for x in range(num[n][2].shape[1]):
                        if num[n][2][y][x] == 255 :
                            map += 1
                        if relabel[y][x] == num[n][2][y][x]:
                            checkmap += 1

                percent = checkmap/map * 100
                if percent > 99 : # 超過99%相同存起來
                    #print("[{}]percent:".format(table[t][1]), percent)
                    same.append( [table[t][1],percent] )
                    NUM = True

            same.sort( key=lambda s: s[1], reverse = True ) #依照相似度排大小(大到小)
            if NUM == True :
                c += 1
                ans += same[0][0]
            n += 1

        ans += '-'
        c = 0
        while n < len(num) :  # for NUM
            same = []
            NUM = False
            if c == 4 : break

            for t in range(26,len(label)):
                relabel = cv2.resize( label[t][2], [num[n][2].shape[1], num[n][2].shape[0]])
                checkmap = 0
                map = 0
                for y in range(num[n][2].shape[0]):
                    for x in range(num[n][2].shape[1]):
                        if num[n][2][y][x] == 255 :
                            map += 1
                        if relabel[y][x] == num[n][2][y][x]:
                            checkmap += 1

                percent = checkmap/map * 100
                if percent > 99 :
                    #print("[{}]percent:".format(table[t][1]), percent)
                    same.append( [table[t][1],percent] )
                    NUM = True

            #print( NUM )
            same.sort( key=lambda s: s[1], reverse = True )
            if NUM == True :
                c += 1
                #print(same[0])
                ans += same[0][0]

            n += 1

        print(f"License Plate: {ans}")
        cv2.imwrite(f'license/{ans}.jpg', img7)

    else:
        print("Can't read in \"template_fonts.bmp\".")

