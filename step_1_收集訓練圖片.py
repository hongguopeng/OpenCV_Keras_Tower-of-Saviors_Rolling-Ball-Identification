import cv2
import numpy as np
import csv
import os

#-------------------------副程式-------------------------#
def get_location_func(event , x , y , flags , param):
    global img_catch , count , x0 , x1 , x2 , x3 , y0 , y1 , y2 , y3
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img_catch , (x , y) , 3 , (255 , 255 , 255) , -1)
        cv2.imshow('Image Mouse Click' , img_catch)
        print('x:{:1d} , y:{:1d}'.format(x , y))
        if count == 0:
            x0 = x 
            y0 = y
        if count == 1:
            x1 = x 
            y1 = y
        if count == 2:
            x2 = x 
            y2 = y
        if count == 3:
            x3 = x 
            y3 = y  
        count = count + 1   
#-------------------------副程式-------------------------#


#-------------------------主程式-------------------------#   
grid_h , grid_w , grid_pixel , offset = 5 , 6 , 50 , 10
look_table = ['31' , '25' , '04' , '40' , '13' , '44']
cam = cv2.VideoCapture(1)

# 若update = 1的話，則分別對圖片取四個角落取邊界點，並將四個邊界點的座標存成txt檔，並利用這四個邊界點的座標進行透視轉換
# 若update = 0的話，則讀取已存檔的txt檔，並將txt檔中的四個邊界點的座標進行透視轉換
update = 0
if update == 1:
    
    # 測試camera以確保camera有拍到清晰的圖片(40 ⇔ 44)
    while True:
        ret_val , img_catch = cam.read()
        cv2.imshow('Testing camera' , img_catch)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows() 
    
    
    # 對左鍵雙擊的點座標，並將這些點座標存成txt檔(48 ⇔ 69)
    count , x0 , x1 , x2 , x3 , y0 , y1 , y2 , y3 = 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0
                       
    while True:
        ret_val , img = cam.read()
        cv2.imshow('Shooting' , img)
        cv2.setMouseCallback('Shooting' , get_location_func)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    directory = os.path.join('./image_location')
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join('./image_location' , 'image_location.txt')
    f_out = open(path , 'w+')
    f_out.write(str(x0) + '\n')
    f_out.write(str(y0) + '\n')
    f_out.write(str(x1) + '\n')
    f_out.write(str(y1) + '\n')
    f_out.write(str(x2) + '\n')
    f_out.write(str(y2) + '\n')
    f_out.write(str(x3) + '\n')
    f_out.write(str(y3) + '\n')
    f_out.close()
        
     
    # 透視轉換，也就是將原本斜斜的圖片轉一個角度變成上視圖(73 ⇔ 76)
    pts1 = np.float32([[x0 , y0] , [x1 , y1] , [x2 , y2] , [x3 , y3]])
    pts2 = np.float32([[0 , 0] , [300 , 0] , [300 , 250] ,[0 , 250] ])
    M = cv2.getPerspectiveTransform(pts1 , pts2)    
    img_transform = cv2.warpPerspective(img_catch , M , (300 , 250))

    while True:
        cv2.imshow('Transform Image' , img_transform)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

#-----------------------------------------------------------------------------#    
    
elif update == 0:
    
    # 讀取點座標之txt檔以獲取點座標(87 ⇔ 103)
    path = os.path.join('./image_location' , 'image_location.txt')            
    f_in = open(path , 'r')
    lines = f_in.readlines()
    try:
        location = [line.replace('\n' , '') for line in lines]
    except FileNotFoundError:
        location = []
    
    if len(location) != 0:
        x0 = int(location[0])
        y0 = int(location[1])
        x1 = int(location[2])
        y1 = int(location[3])
        x2 = int(location[4])
        y2 = int(location[5])
        x3 = int(location[6])
        y3 = int(location[7])
               
        
        folder_index = 0
        while True:
            ret_val , img = cam.read()
            cv2.imshow('Shooting' , img)
                
            if cv2.waitKey(20) & 0xFF == ord('c'):
                print('Image Collecting ...')
                
                # 透視轉換，也就是將原本斜斜的圖片轉一個角度變成上視圖(115 ⇔ 119)
                pts1 = np.float32([[x0 , y0] , [x1 , y1] , [x2 , y2] , [x3 , y3]])
                pts2 = np.float32([[0 , 0] , [300 , 0] , [300 , 250] ,[0 , 250] ])
                M = cv2.getPerspectiveTransform(pts1 , pts2)    
                img_transform = cv2.warpPerspective(img , M , (300 , 250)) 
                img_transform_gray = cv2.cvtColor(img_transform , cv2.COLOR_BGR2GRAY)


                # 用一般比對圖片的方式辨識轉珠種類(123 ⇔ 138)
                thresh_mean = cv2.adaptiveThreshold(img_transform_gray , 255 , cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY , 11 , 2)
                arr = np.zeros([grid_h , grid_w] , dtype = int)
                for ii in range(0 , grid_h):
                    for jj in range(0 , grid_w):
                        img_cut = thresh_mean[ii * grid_pixel : (ii + 1) * grid_pixel , 
                                              jj * grid_pixel : (jj + 1) * grid_pixel]
                        gmax_val = 0
                        for k in range(0 , 6):
                            path_template = os.path.join('./template' , '{}.jpg'.format(look_table[k]))
                            template = cv2.imread(path_template , 0)
                            res = cv2.matchTemplate(img_cut , template , cv2.TM_CCOEFF_NORMED)
                            min_val , max_val , min_loc , max_loc = cv2.minMaxLoc(res)
                            if max_val > gmax_val:
                                gmax_val = max_val
                                index = k + 1
                        arr[ii][jj] = index

                # 將神魔之塔圖片中的每個轉珠切割成小張的圖，並存檔(141 ⇔ 150)
                folder_img_cut = os.path.join('./train_image/{}'.format(folder_index))
                if not os.path.exists(folder_img_cut):
                    os.makedirs(folder_img_cut)
                for ii in range(0 , grid_h):
                    for jj in range(0 , grid_w):
                        # 切割出來的轉珠圖片有一些要當作template，所以要讓切割出來的轉珠圖片的width與height各自減少offset
                        img_cut = thresh_mean[ii * grid_pixel + offset : (ii + 1) * grid_pixel - offset ,
                                              jj * grid_pixel + offset : (jj + 1) * grid_pixel - offset] 
                        path_cut = os.path.join(folder_img_cut , '{}{}.jpg'.format(ii , jj))
                        cv2.imwrite(path_cut , img_cut)

                # 將神魔之塔圖片中的每個轉珠，所比對出的真正的label存成csv檔(153 ⇔ 157)
                path_label = os.path.join(folder_img_cut , 'label.csv')
                label_file = open(path_label , 'w+')
                writer = csv.writer(label_file , lineterminator = '\n')
                writer.writerows(arr.tolist())        
                label_file.close()

                # 將轉珠真正的label直接印在神魔之塔圖片(160 ⇔ 167)
                for ii in range(0 , grid_h):
                    for jj in range(0 , grid_w):   
                        cv2.putText(img_transform , str(arr[ii][jj]) , 
                                    ( int((jj + 0.5) * grid_pixel) , int((ii + 0.5) * grid_pixel) ) ,
                                    cv2.FONT_HERSHEY_COMPLEX , 1 , (125 , 255 , 125) , 2 , cv2.LINE_AA)
                cv2.imshow('Correct Label' , img_transform)        
                path_correct_img = os.path.join(folder_img_cut , 'detect.jpg')
                cv2.imwrite(path_correct_img , img_transform)

                folder_index = folder_index + 1

            elif cv2.waitKey(20) & 0xFF == ord('q'): break

        cv2.destroyAllWindows()

    elif len(location) == 0:
        print('Error:File dose not appear to exist')
#-------------------------主程式-------------------------#
