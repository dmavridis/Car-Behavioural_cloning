## Model to drive the simulator car in autonomous mode

# Imports

import pandas
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

from PIL import Image
import cv2
import base64
from io import BytesIO
 

### Image transfromations

# Shift image 
def shift_image(image, steer):
    '''
    Randomly shifts an image to the four directions
    Y shifting is tighter (+/- 10px)
    X Shifting is wider -30 to +70 px, the non symmetry is more of a legacy choice
    Steering should be updated for x-shifting
    
    Returns: Shifted image and updated streering value
    ''' 
    x_val = np.random.randint(0,100) - 50
    y_val = np.random.randint(0,20) - 10 
  
    steer = steer + 0.2*x_val/50

    rows,cols = np.shape(image)[0],  np.shape(image)[1]
    M = np.float32([[1,0,x_val],[0,1,y_val]])
    image_tr = cv2.warpAffine(image,M,(cols,rows))
    
    return image_tr,steer

def randomize_brightness(image):
    '''
    Convert the rgb image to hsv format and increase the brightness by a random value 0 - 100 
    
    Returns: darker image
    '''
    
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    value = np.random.randint(0, 60)

    image[:,:,2] += value
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def flip_image(image, steer):
    '''
    Flips an image if a random generated value allows it
    
    Returns: Input or flipped image and corresponding steering
    '''
	
    flip_on = np.random.randint(2)
    if flip_on:
        image = cv2.flip(image,1)
        steer = - steer
    return image, steer


    return image, steer
    
def resize_image(image):
    '''
    Cuts 30 and 25 pixels from the top and bottom of the image
    Converts the image to 64x64, using interpolation
    
    Returns: 64x64x3 image
    '''
    shape = image.shape
    image = image[30:shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(64, 64), interpolation=cv2.INTER_AREA)    
    return image


def preprocess_image_file_train(df_line):
    '''
    Input: Line of the dataframe that has the left, center and right image paths,
    the steering, the throttle and the speed
    
    Output: A 64x64 randomly transormed image with the various transformations
    '''
    camera = np.random.randint(-1,2) # -1: left, 0: center, 1:right
    steer_value = 0.3
    camera_view = {'-1': 'left', '0': 'center', '1': 'right'}

    img_file = df_line[camera_view[str(camera)]]
    steer = df_line['steering'] - camera*steer_value

    # Reading the selected image
    img_file = str.strip(img_file)
	
    with open(img_file, "rb") as imageFile:
        str1 = base64.b64encode(imageFile.read())
        image = Image.open(BytesIO(base64.b64decode(str1)))
        image = np.asarray(image)

    image, steer = shift_image(image, steer)
    image = randomize_brightness(image)
    image, steer = flip_image(image, steer)
    image = resize_image(image)

    return image, steer

# Function for the fit_generator that provides a se
def train_data_batch(data, batch_size = 64):
    ''' 
    Function for the fit generator
    '''
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data.iloc[i_line]     
            x,y = preprocess_image_file_train(line_data)

            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering

# Keras CNN model
def get_model():
    ''' 
    Keras model
    '''
    ch, row, col = 3, 64, 64
    model = Sequential()
    
    model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode="same", input_shape=( row, col, ch)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(128))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1, name= 'Steering_Angle'))

    model.summary()
    model.compile(optimizer="adam", loss="mse")

    return model

def main():

    ### Get the input data locations
    df = pandas.read_csv('driving_log.csv')
    model = get_model()
    ### Training the data 
    
    model.fit_generator(
        train_data_batch(df),
        samples_per_epoch=300*64,
        nb_epoch = 6)

    print("Fitting done")
    
    model.save_weights("./model.h5", True)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    print("Model Saved")

if __name__ == "__main__": 
    main()