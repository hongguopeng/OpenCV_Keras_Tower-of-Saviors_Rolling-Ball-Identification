import numpy as np
import cv2
import csv
import os
from keras.models import model_from_json
import matplotlib.pyplot as plt


class towers_of_saviors(object):
    def __init__(self , grid_h , grid_w , grid_pixel , offset):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.grid_pixel = grid_pixel
        self.offset = offset
        self.look_table = ['31' , '25' , '04' , '40' , '13' , '44']
        self.cam = cv2.VideoCapture(1)

    @staticmethod
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

    # 測試camera以確保camera有拍到清晰的圖片
    def test_webcam(self):
        print('Webcam Testing ...')
        while True:
            ret_val , img = self.cam.read()
            cv2.imshow('Testing camera' , img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        print('Testing Finish !')
        return img
        
    # 對左鍵雙擊的點座標，並將這些點座標存成txt檔
    def left_button_double_click(self):
        while True:
            ret_val , img = self.cam.read()
            cv2.imshow('Shooting' , img)
            cv2.setMouseCallback('Shooting' , self.get_location_func)
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

    # 透視轉換，也就是將原本斜斜的圖片轉一個角度變成上視圖
    def perspective_transform(self , img , x0 , x1 , x2 , x3 , y0 , y1 , y2 , y3 , show):
        pts1 = np.float32([[x0 , y0] , [x1 , y1] , [x2 , y2] , [x3 , y3]])
        pts2 = np.float32([[0 , 0] , [300 , 0] , [300 , 250] ,[0 , 250] ])
        M = cv2.getPerspectiveTransform(pts1 , pts2)
        self.img_transform_1 = cv2.warpPerspective(img , M , (300 , 250))
        self.img_transform_2 = self.img_transform_1.copy()
        self.img_transform_gray = cv2.cvtColor(self.img_transform_1 , cv2.COLOR_BGR2GRAY)               
        if show == 1:
            while True:
                cv2.imshow('Transform Image' , self.img_transform_1)
                cv2.resizeWindow('Transform Image' ,  500 , 400)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cv2.destroyAllWindows()

    # 讀取點座標之txt檔以獲取點座標
    def read_txt(self):
        path = os.path.join('./image_location' , 'image_location.txt')            
        f_in = open(path , 'r')
        lines = f_in.readlines()
        try:
            location = [line.replace('\n' , '') for line in lines]
            x0 = int(location[0])
            y0 = int(location[1])
            x1 = int(location[2])
            y1 = int(location[3])
            x2 = int(location[4])
            y2 = int(location[5])
            x3 = int(location[6])
            y3 = int(location[7])
            return x0 , x1 , x2 , x3 , y0 , y1 , y2 , y3    
        except FileNotFoundError:
            print('Error:File dose not appear to exist')

    # 用一般比對圖片的方式辨識轉珠種類，並且得到正確的轉珠label
    def compare_template_image(self):
        thresh_mean = cv2.adaptiveThreshold(self.img_transform_gray , 255 , cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY , 11 , 2)
        self.arr = np.zeros([self.grid_h , self.grid_w] , dtype = int)
        for ii in range(0 , self.grid_h):
            for jj in range(0 , self.grid_w):
                img_cut = thresh_mean[ii * self.grid_pixel : (ii + 1) * self.grid_pixel ,
                                      jj * self.grid_pixel : (jj + 1) * self.grid_pixel]
                gmax_val = 0
                for k in range(0 , 6):
                    path_template = os.path.join('./template' , '{}.jpg'.format(self.look_table[k]))
                    template = cv2.imread(path_template , 0)
                    res = cv2.matchTemplate(img_cut , template , cv2.TM_CCOEFF_NORMED)
                    min_val , max_val , min_loc , max_loc = cv2.minMaxLoc(res)
                    if max_val > gmax_val:
                        gmax_val = max_val
                        index = k + 1
                self.arr[ii][jj] = index  
        
        for ii in range(0 , self.grid_h):
            for jj in range(0 , self.grid_w):   
                cv2.putText(self.img_transform_1 , str(self.arr[ii][jj]) ,
                            ( int((jj + 0.5) * self.grid_pixel) , int((ii + 0.5) * self.grid_pixel) ) , 
                            cv2.FONT_HERSHEY_COMPLEX , 1 , (125 , 255 , 125) , 2 , cv2.LINE_AA)
        cv2.imshow('Correct Label' , self.img_transform_1)
        cv2.resizeWindow('Correct Label' ,  500 , 400)

    def save_cut_image_label(self , folder_index):
        # 將神魔之塔圖片中的每個轉珠切割成小張的圖，並存檔
        folder_img_cut = os.path.join('./train_image/{}'.format(folder_index))
        if not os.path.exists(folder_img_cut):
           os.makedirs(folder_img_cut)
        thresh_mean = cv2.adaptiveThreshold(self.img_transform_gray , 255 , cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY , 11 , 2)
        for ii in range(0 , self.grid_h):
            for jj in range(0 , self.grid_w):
                # 切割出來的轉珠圖片有一些要當作template，所以要讓切割出來的轉珠圖片的width與height各自減少offset
                img_cut = thresh_mean[ii * self.grid_pixel + self.offset : (ii + 1) * self.grid_pixel - self.offset ,
                                      jj * self.grid_pixel + self.offset : (jj + 1) * self.grid_pixel - self.offset] 
                path_cut = os.path.join(folder_img_cut , '{}{}.jpg'.format(ii , jj))
                cv2.imwrite(path_cut , img_cut)
        
        path_correct_img = os.path.join(folder_img_cut , 'detect.jpg')
        cv2.imwrite(path_correct_img , self.img_transform_1)

        # 將神魔之塔圖片中的每個轉珠所比對出的真正的label存成csv檔
        path_label = os.path.join(folder_img_cut , 'label.csv')
        label_file = open(path_label , 'w+')
        writer = csv.writer(label_file , lineterminator = '\n')
        writer.writerows(self.arr.tolist())        
        label_file.close()

    # 載入訓練好的model來辨識轉珠類型
    def recognize_image(self):
        thresh_mean = cv2.adaptiveThreshold(self.img_transform_gray , 255 , cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY , 11 , 2)
        x_test = []
        for ii in range(0 , self.grid_h):
            for jj in range(0 , self.grid_w):
                img_cut = thresh_mean[ii * self.grid_pixel + self.offset : (ii + 1) * self.grid_pixel - self.offset ,
                                      jj * self.grid_pixel + self.offset : (jj + 1) * self.grid_pixel - self.offset]
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
				   
        for ii in range(0 , grid_h):
            for jj in range(0 , grid_w):   
                cv2.putText(self.img_transform_2 , str(arr_test[ii][jj] + 1) ,
                            ( int((jj + 0.5) * self.grid_pixel) , int((ii + 0.5) * self.grid_pixel) ) ,
                            cv2.FONT_HERSHEY_COMPLEX , 1 , (50 , 150 , 255) , 2 , cv2.LINE_AA)
        cv2.imshow('Predict Label' , self.img_transform_2)
        cv2.resizeWindow('Predict Label' ,  500 , 400)

    def show_predict_correct(self):
        fig , ax = plt.subplots(1 , 2 , figsize = (20 , 10))

        ax[0].imshow(self.img_transform_1)
        ax[0].set_title('Correct Label' , size = 50)
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(self.img_transform_2)
        ax[1].set_title('Predict label' , size = 50)
        ax[1].set_xticks([])
        ax[1].set_yticks([])


global img_catch , count , x0 , x1 , x2 , x3 , y0 , y1 , y2 , y3
count , x0 , x1 , x2 , x3 , y0 , y1 , y2 , y3 = 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0
grid_h , grid_w , grid_pixel , offset = 5 , 6 , 50 , 10
TOS = towers_of_saviors(grid_h , grid_w , grid_pixel , offset)

update = 0
mode = 'Image Identifying'
if update == 1:
    img_catch = TOS.test_webcam()
    TOS.left_button_double_click()
    TOS.perspective_transform(img_catch , x0 , x1 , x2 , x3 , y0 , y1 , y2 , y3 , 1)

elif update == 0:
    _ = TOS.test_webcam()
    cv2.destroyAllWindows()
    x0 , x1 , x2 , x3 , y0 , y1 , y2 , y3 = TOS.read_txt()

    folder_index = 0
    while True:
        ret_val , img = TOS.cam.read()
        cv2.imshow('Shooting' , img)

        if cv2.waitKey(20) & 0xFF == ord('c'):
            TOS.perspective_transform(img , x0 , x1 , x2 , x3 , y0 , y1 , y2 , y3 , 0)
            if mode == 'Image Collecting':
                print('Image Collecting ...')
                TOS.compare_template_image()
                TOS.save_cut_image_label(folder_index)
                folder_index += 1
                
            elif mode == 'Image Identifying':
                print('Image Identifying ...')
                TOS.compare_template_image()
                TOS.recognize_image()
                TOS.show_predict_correct()
                
        elif cv2.waitKey(20) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
