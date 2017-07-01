from PIL import ImageGrab
import numpy as np
import cv2
from time import sleep
import threading
import win32api,win32con
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from time import sleep

"""
This piece of code was used to generate training data
i=0
while i<600:
    sleep(0.0000001)
    img = ImageGrab.grab(bbox=(385,320,580,450)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img_array = np.array(img) #this is the array obtained from conversion
    frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("test", frame)
    #cv2.waitKey(0)
    cv2.imwrite("filename{}.jpg".format(i),frame)
    break
    i+=1
"""


class SnapTaker:

    def __init__(self, capture_box=(385,320,580,450)):
        self.capture_box = capture_box

    def takesnap(self):
        img=ImageGrab.grab(bbox=self.capture_box)
        img_array=np.array(img)
        img_array = cv2.resize(img_array, (128,128), interpolation = cv2.INTER_AREA)
        #frame=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
        return img_array


class jumper:

    def __init__(self,duration=0.1):
        self.duration=duration

    def key_down(self):
        win32api.keybd_event(0x20 ,0,1,0)


    def key_up(self):
        win32api.keybd_event(0x20,0,win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP,0)

    def press(self):
        self.key_down()
        sleep(self.duration)
        self.key_up()

    def jump(self):
        threading.Thread(target=self.press).start()

classifier=None
def load_model():
    global classifier
    model_file=open('Trex_game/model.json', 'r')
    loaded_model=model_file.read()
    model_file.close()
    classifier=model_from_json(loaded_model)
    #load weights into new model
    classifier.load_weights("Trex_game/weights.hdf5")
    print("Model loaded successfully")


#get ready to roll
b=jumper()
load_model()

while True:
    a=SnapTaker(capture_box=(210,339,405,467))
    snap=a.takesnap()
    snap=snap.reshape((1,)+snap.shape)
    test_datagen = ImageDataGenerator()
    m=test_datagen.flow(snap,batch_size=1)
    y_pred=classifier.predict_generator(m,1)
    if y_pred>0.5:
        print("NO JUMP")
    else:
        b.jump()
        print("JUMP")


