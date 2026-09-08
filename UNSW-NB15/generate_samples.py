import tensorflow as tf
import numpy as np
from models import cgan

# {'Analysis': 0, 'Backdoor': 1, 'DoS': 2, 'Exploits': 3, 'Fuzzers': 4, 'Generic': 5, 'Reconnaissance': 7, 'Shellcode': 8, 'Worms': 9}
labels = np.array([0,1,2,3,4,5,7,8,9])
n = len(labels)
rand_noise_dim = 32
noise = np.random.normal(0,1,(n,rand_noise_dim))
filepath = 'trained_generators/gen.h5'
model = tf.keras.models.load_model(filepath)
print(model.predict([noise,labels]))
