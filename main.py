import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

model = load_model("/Users/r_prateek/Desktop/Birds-Class/birds.h5")

train_path = "/Users/r_prateek/Desktop/Birds-Class/birdspeciesdata/train"

datagen = ImageDataGenerator(
    rescale = 1./255,
)

train_data = datagen.flow_from_directory(
    train_path,
    target_size = ((224,224)),
   
)

ci = train_data.class_indices
classes = {v: k for k, v in ci.items()}

path = input('Please enter path of image of bird to classify')

inp = Image.open(path)
img = inp.resize((224,224))
img = np.array(img)/255.0
img = np.reshape(img, [1,224,224,3])

predictions = model.predict(img)

top_values, top_indices = tf.nn.top_k(predictions, k=3)

values = np.array(top_values)
indices = np.array(top_indices)

print('Input Image: \n\n\n')
inp.show()

print('Probabilities: \n')

for i in range(3):
    print(classes[indices[0][i]] + " : ", end = "")
    print(values[0][i] * 100)
    print()

