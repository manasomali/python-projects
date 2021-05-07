import tensorflow as tf
import numpy as np

"""
Let's Load Data:
REF: https://www.tensorflow.org/tutorials/load_data/text
"""
batch_size = 10
#batch_size = 32
file = 'transcricoes_comnum_comsubes'
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "input/"+str(file)+"/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337
)
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "input/"+str(file)+"/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337
)
raw_test_ds  = tf.keras.preprocessing.text_dataset_from_directory(
    "input/"+str(file)+"/test",
    batch_size=batch_size
)

print(
    "Number of batches in raw_train_ds: %d" % tf.data.experimental.cardinality(raw_train_ds)
)
print(
    "Number of batches in raw_val_ds: %d" % tf.data.experimental.cardinality(raw_val_ds)
)
print(
    "Number of batches in raw_test_ds: %d" % tf.data.experimental.cardinality(raw_test_ds)
)

"""
Let's preview a few samples:
"""
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(2):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])


"""
## Prepare the data
"""
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

VOCAB_SIZE = 2000
MAX_SEQUENCE_LENGTH = 150

binary_vectorize_layer = TextVectorization(
    output_mode='binary')

int_vectorize_layer = TextVectorization(
    output_mode='int')

tfidf_vectorize_layer = TextVectorization(
    output_mode='tf-idf')

def binary_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return binary_vectorize_layer(text), label

def int_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return int_vectorize_layer(text), label

def tfidf_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return tfidf_vectorize_layer(text), label


train_text = raw_train_ds.map(lambda text, labels: text)

binary_vectorize_layer.adapt(train_text)
int_vectorize_layer.adapt(train_text)
tfidf_vectorize_layer.adapt(train_text)


# Retrieve a batch (of 32 docs and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_doc, first_label = text_batch[1], label_batch[1]

print("Doc", first_doc)
print("Label", first_label)

print("'binary' vectorized question:", 
      binary_vectorize_text(first_doc, first_label)[0])
print("'int' vectorized question:",
      int_vectorize_text(first_doc, first_label)[0])
print("'tfidf' vectorized question:",
      tfidf_vectorize_text(first_doc, first_label)[0])

print("100 ---> ", int_vectorize_layer.get_vocabulary()[100])
print("200 ---> ", int_vectorize_layer.get_vocabulary()[200])
print("Vocabulary size: {}".format(len(int_vectorize_layer.get_vocabulary())))
# Model constants.
binary_train_ds = raw_train_ds.map(binary_vectorize_text)
binary_val_ds = raw_val_ds.map(binary_vectorize_text)
binary_test_ds = raw_test_ds.map(binary_vectorize_text)

int_train_ds = raw_train_ds.map(int_vectorize_text)
int_val_ds = raw_val_ds.map(int_vectorize_text)
int_test_ds = raw_test_ds.map(int_vectorize_text)

tfidf_train_ds = raw_train_ds.map(tfidf_vectorize_text)
tfidf_val_ds = raw_val_ds.map(tfidf_vectorize_text)
tfidf_test_ds = raw_test_ds.map(tfidf_vectorize_text)


"""
## Build a model
We choose a simple 1D convnet starting with an `Embedding` layer.
"""


from tensorflow.keras import layers
from tensorflow.keras import losses

binary_model = tf.keras.Sequential([layers.Dense(12)])
binary_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])
history = binary_model.fit(
    binary_train_ds, validation_data=binary_val_ds, epochs=10)


tfidf_model = tf.keras.Sequential([layers.Dense(12)])
tfidf_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])
history = tfidf_model.fit(
    tfidf_train_ds, validation_data=tfidf_val_ds, epochs=10)


def create_model(vocab_size, num_labels):
  model = tf.keras.Sequential([
      layers.Embedding(vocab_size, 64, mask_zero=True),
      layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
      layers.GlobalMaxPooling1D(),
      layers.Dense(num_labels)
  ])
  return model

VOCAB_SIZE=len(int_vectorize_layer.get_vocabulary())
int_convnet_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=12)
int_convnet_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])
history = int_convnet_model.fit(int_train_ds, validation_data=int_val_ds, epochs=10)

print("Linear model on binary vectorized data:")
print(binary_model.summary())

print("Linear model on tfidf vectorized data:")
print(tfidf_model.summary())

print("ConvNet model on int vectorized data:")
print(int_convnet_model.summary())

binary_loss, binary_accuracy = binary_model.evaluate(binary_test_ds)
tfidf_loss, tfidf_accuracy = tfidf_model.evaluate(tfidf_test_ds)
int_convnet_loss, int_convnet_accuracy = int_convnet_model.evaluate(int_test_ds)

print("Binary Linear model accuracy: {:2.2%}".format(binary_accuracy))
print("Tfidf Linear  model accuracy: {:2.2%}".format(tfidf_accuracy))
print("Int ConvNet model accuracy: {:2.2%}".format(int_convnet_accuracy))
