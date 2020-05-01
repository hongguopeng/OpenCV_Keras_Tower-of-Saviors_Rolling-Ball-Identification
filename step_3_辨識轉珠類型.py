import numpy as np
import cv2
import os
from keras.models import model_from_json

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

        
        while True:
            ret_val , img = cam.read()
            cv2.imshow('Shooting' , img)
                
            if cv2.waitKey(20) & 0xFF == ord('c'):
                
                # 透視轉換，也就是將原本斜斜的圖片轉一個角度變成上視圖(113 ⇔ 118)
                pts1 = np.float32([[x0 , y0] , [x1 , y1] , [x2 , y2] , [x3 , y3]])
                pts2 = np.float32([[0 , 0] , [300 , 0] , [300 , 250] ,[0 , 250] ])
                M = cv2.getPerspectiveTransform(pts1 , pts2)    
                img_transform_1 = cv2.warpPerspective(img , M , (300 , 250))
                img_transform_2 = img_transform_1.copy()
                img_transform_gray = cv2.cvtColor(img_transform_1 , cv2.COLOR_BGR2GRAY)

                
                # 用一般比對圖片的方式辨識轉珠種類(122 ⇔ 137)
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

                # 將轉珠真正的label直接印在神魔之塔圖片(140 ⇔ 145)
                for ii in range(0 , grid_h):
                    for jj in range(0 , grid_w):   
                        cv2.putText(img_transform_1 , str(arr[ii][jj]) ,
                                    ( int((jj + 0.5) * grid_pixel) , int((ii + 0.5) * grid_pixel) ) , 
                                    cv2.FONT_HERSHEY_COMPLEX , 1 , (125 , 255 , 125) , 2 , cv2.LINE_AA)
                cv2.imshow('Correct Label' , img_transform_1)

                
                # 載入訓練好的model來辨識轉珠類型(149 ⇔ 169)
                x_test = []
                for ii in range(0 , grid_h):
                    for jj in range(0 , grid_w):
                        img_cut = thresh_mean[ii * grid_pixel + offset : (ii + 1) * grid_pixel - offset ,
                                              jj * grid_pixel + offset : (jj + 1) * grid_pixel - offset]
                        x_test = np.append(x_test , img_cut)				
			     
                x_test = x_test.reshape(30 , 30 , 30 , 1)
                x_test = x_test / 255
                model_json_path = os.path.join('.' , 'saved_model.json')
                model_h5_path = os.path.join('.' , 'saved_model_weights.h5')
                json_file = open(model_json_path , 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights(model_h5_path)      
                loaded_model.compile(loss = 'categorical_crossentropy' ,
                                     optimizer = 'adam' , 
                                     metrics = ['accuracy'])
                prediction = loaded_model.predict_classes(x_test)
                arr_test = prediction.reshape(5 , 6)

                # 將轉珠預測的label直接印在神魔之塔圖片(172 ⇔ 177)
                for ii in range(0 , grid_h):
                    for jj in range(0 , grid_w):
                        cv2.putText(img_transform_2 , str(arr_test[ii][jj] + 1) ,
                                    ( int((jj + 0.5) * grid_pixel) , int((ii + 0.5) * grid_pixel) ) ,
                                    cv2.FONT_HERSHEY_COMPLEX , 1 , (50 , 50 , 100) , 2 , cv2.LINE_AA)
                cv2.imshow('Predict Label' , img_transform_2)

            elif cv2.waitKey(20) & 0xFF == ord('q'): break

        cv2.destroyAllWindows()

    elif len(location) == 0:
        print('Error:File dose not appear to exist')
#-------------------------主程式-------------------------#   
