"""This code was used to train the model using about 600 images from the game"""

from keras.models import Sequential,model_from_json
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#initialize the cnn
classifier=Sequential()

#convolutional layers: note currently only one filter has been added
classifier.add(Convolution2D(1, 3, 3, input_shape = (128,128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#once more lol
classifier.add(Convolution2D(1, 3, 3, input_shape = (128,128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#remove individual pixels to submit to the input layer
classifier.add(Flatten())

#two hidden layers with rectified linear function:
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 128, activation = 'relu'))

#one output layer: (sigmoid function) with one neuron
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compiling the classifier
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#time to generate training and test sets
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.,
                                   horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 1,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128,128),
                                            batch_size = 1,
                                            class_mode = 'binary')


#finally, start training
classifier.fit_generator(training_set,
                         samples_per_epoch = 624,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 84)

#after 5 epochs, accuracy was found to be 0.9647 on training and 0.9405 on test set resp.

#saving the weights
classifier.save_weights("weights.hdf5",overwrite=True)

#note that values are close to zero when it is to Jump and close to 1 when not to Jump


from keras.preprocessing.image import img_to_array,load_img

#testing it to a random image from the test set
img = load_img('dataset/test_set/jump/filename553.jpg',target_size=(128,128))
x=img_to_array(img)
x=x.reshape((1,)+x.shape)
test_datagen = ImageDataGenerator(rescale=1./255)
m=test_datagen.flow(x,batch_size=1)
y_pred=classifier.predict_generator(m,1)

#saving the model itself in json format:
model_json = classifier.to_json()
with open("model.json", "w") as model_file:
    model_file.write(model_json)
print("Model has been saved.")


"""to Load the model with weights:
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("weights.hdf5")
print("Loaded model from disk")

#compile again
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#test an image again
img = load_img('dataset/test_set/not_jump/filename543.jpg',target_size=(128,128))
x=img_to_array(img)
x=x.reshape((1,)+x.shape)
test_datagen = ImageDataGenerator(rescale=1./255)
m=test_datagen.flow(x,batch_size=1)
y_pred=classifier.predict_generator(m,1)
"""




