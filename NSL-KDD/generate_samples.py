import tensorflow as tf
import numpy as np
from models import cgan

#label_mapping: {'dos': 0, 'probe': 2, 'r2l': 3, 'u2r': 4}
labels = np.array([0,2,3,4])
n = len(labels)
rand_noise_dim = 32
noise = np.random.normal(0,1,(n,rand_noise_dim))
#print("noise:",noise)
filepath = 'trained_generator/gen.h5'
model = tf.keras.models.load_model(filepath)
print(model.predict([noise,labels]))
