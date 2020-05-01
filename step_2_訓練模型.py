import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten
from keras.layers import Conv2D , MaxPooling2D
from keras.models import model_from_json
import csv
import os
import cv2

batch_size = 32
num_classes = 6        # 神魔之塔有6種轉珠
folder_num = 15        # 拿前15個folder的資料當測試集
val_num = 30           # 驗證集image數目
val_folder_index = 16  # 第16個folder為測試集

#--------------------準備x_train y_train x_val y_val--------------------#
x_train = []
for folder_index in range(0 , folder_num):
    for i in range(0 , 5):
        for j in range(0 , int(val_num / 6) + 1):
            img_name = os.path.join('./train_image/{}'.format(folder_index) , '{}{}.jpg'.format(i , j))
            print(img_name)
            image = cv2.imread(img_name , 0)
            x_train = np.append(x_train , image)


y_train = []
for folder_index in range(0 , folder_num):
    label_name = os.path.join('./train_image/{}'.format(folder_index) , 'label.csv')
    print(label_name) 
    csv_file = open(label_name , 'r')
    reader = csv.reader(csv_file , lineterminator = '\n')
    for row in reader:
        for j in range(0 , len(row)):
            row[j] = int(row[j]) - 1
        y_train =  np.append(y_train , row)  


x_val = []
for i in range(0 , 5):
    for j in range(0 , int(val_num / 6) + 1):
        img_name = os.path.join('./train_image/{}'.format(val_folder_index) , '{}{}.jpg'.format(i , j))
        print(img_name)
        image = cv2.imread(img_name , 0)
        x_val = np.append(x_val , image)
    

label_name = os.path.join('./train_image/{}'.format(val_folder_index) , 'label.csv')
y_val = []
csv_file = open(label_name , 'r')
reader = csv.reader(csv_file , lineterminator = '\n')
for row in reader:
    for j in range(0 , len(row)):
        row[j] = int(row[j]) - 1
    y_val = np.append(y_val , row)


img_rows , img_cols = 30 , 30
x_train = x_train.reshape(folder_num * 30 , img_rows , img_cols , 1)
x_val = x_val.reshape(val_num , img_rows , img_cols, 1)
input_shape = (img_rows , img_cols , 1)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train = x_train / 255
x_val = x_val / 255
print('x_train shape:{}'.format(x_train.shape))
print('x_val shape:{}'.format(x_val.shape))
y_train = keras.utils.to_categorical(y_train , num_classes)
y_val = keras.utils.to_categorical(y_val , num_classes)
#--------------------準備x_train y_train x_val y_val--------------------#


#--------------------驗證模型準確度--------------------#
model = Sequential()
model.add(Conv2D(32 , kernel_size=(3 , 3),
                 activation = 'relu',
                 input_shape = input_shape))

model.add(Conv2D(64 , (3 , 3) , activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2 , 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128 , activation = 'relu'))
model.add(Dropout(0.5))
model.add( Dense(num_classes , activation = 'softmax') )

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adam(),
              metrics = ['accuracy'])

epochs = 10
model.fit(x_train , y_train,
          validation_data = (x_val , y_val) , 
          batch_size = batch_size,
          epochs = epochs,
          verbose = 1)
#--------------------驗證模型準確度--------------------#


# serialize model to JSON
model_json = model.to_json()
model_json_path = os.path.join('.' , 'saved_model.json')
saved_model = open(model_json_path , 'w+')  
saved_model.write(model_json)
saved_model.close()
model_h5_path = os.path.join('.' , 'saved_model_weights.h5')
model.save_weights(model_h5_path) 

# load weights into new model
json_file = open(model_json_path , 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_h5_path)
loaded_model.compile(loss = 'categorical_crossentropy' ,
                     optimizer = 'adam' , 
                     metrics = ['accuracy'])
score = loaded_model.evaluate(x_val , y_val , verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])