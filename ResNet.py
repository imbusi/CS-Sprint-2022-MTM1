import numpy as np
#import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub

#Provide a training_folder, the labels in a dictionary to train the model, the validation split and number of training epochs
#returns the trained ResNet50 model
def CreateResNet(label_dictionary):

    #Change for different model
    feature_extractor_model = "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1"
    #Change for different model

    num_classes = len(label_dictionary)

    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

    model = tf.keras.Sequential([
      feature_extractor_layer,
      tf.keras.layers.Dense(num_classes, activation='sigmoid'),
      tf.keras.layers.Softmax()
    ])

    return model

def ResNet(training_path, vs, training_epochs, label_dictionary):

    #Loads the model, specifies input shape

    #Change for different model
    classifier_model ="https://tfhub.dev/tensorflow/resnet_50/classification/1"    #ResNet50
    #Change for different model


    IMAGE_SHAPE = (224, 224)
    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
    ])
    #End of Model definition


    #ensure the img inputs match the requirement for the model
    batch_size = 32
    img_height = 224
    img_width = 224

    #Provide jpeg, png, bmp, gif images as well as a Label Dictionary
    labelDictionary = label_dictionary

    #Provide dataset, with data augmentation (provide vs for validation)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      training_path,
      validation_split=vs,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)


    class_names = np.array(train_ds.class_names)
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)


    #Change for different model
    feature_extractor_model = "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1"
    #Change for different model

    num_classes = len(class_names)

    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

    model = tf.keras.Sequential([
      feature_extractor_layer,
      tf.keras.layers.Dense(num_classes, activation='sigmoid'),
      tf.keras.layers.Softmax()
    ])

    model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['acc'])

    class CollectBatchStats(tf.keras.callbacks.Callback):
      def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

      def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()

    batch_stats_callback = CollectBatchStats()
    history = model.fit(train_ds, epochs=training_epochs,
                        callbacks=[batch_stats_callback])

    return model
