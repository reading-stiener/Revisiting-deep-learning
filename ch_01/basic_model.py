import tensorflow as tf

W = tf.Variable(tf.ones(shape=(2,2), name="W"))
b = tf.Variable(tf.zeros(shape=(2), name="b"))

@tf.function
def model(x):
  return W * x + b

out_a = model([1,0])
print(out_a)


# single layer network
from tensorflow import keras
NB_CLASSES = 10
RESHAPED = 784
model = tf.keras.models.Sequential()
model.add(
  keras.layers.Dense(NB_CLASSES, input_shape=(RESHAPED,), kernel_initializer="zeros", name="dense_layer", activation="softmax")
)


# multi layer network
 