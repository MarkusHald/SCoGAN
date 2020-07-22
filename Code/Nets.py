import tensorflow as tf
from tensorflow import keras
from Code import Utils as u
layers = tf.keras.layers


def encoder(args):
    input_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    model = keras.Sequential()

    # normal
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=[input_dim, input_dim, channels]))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(args.noise_dim))
    # compile model
    return model


def res_net_block(input_data, filters, kernel, norm, w_init):
    x = layers.Conv2D(filters, kernel, padding='same',kernel_initializer=w_init)(input_data)
    x = u.get_norm(norm)(x)
    #x = layers.PReLU(args.args.prelu_init)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel, padding='same',kernel_initializer=w_init)(x)
    x = u.get_norm(norm)(x)
    x = layers.Add()([input_data, x])
    return x


def mnist_classifier(args, num_classes):
    img_shape = (32, 32, 3)
    input = tf.keras.layers.Input(shape=img_shape)
    model = tf.keras.layers.Conv2D(32, (3,3))(input)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.Conv2D(64, (3,3))(model)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.MaxPooling2D((2,2))(model)
    model = tf.keras.layers.Dropout(0.25)(model)
    model = tf.keras.layers.Flatten()(model)
    model = tf.keras.layers.Dense(128)(model)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(model)

    return tf.keras.Model(input, output)


def celeba_classifier(args, num_classes):
    img_shape = (128,128,3)
    input = tf.keras.layers.Input(shape=img_shape)

    model = tf.keras.layers.Conv2D(32, (3,3))(input)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.Conv2D(64, (3,3))(model)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.MaxPooling2D((2,2))(model)
    model = tf.keras.layers.Dropout(0.25)(model)
    model = tf.keras.layers.Flatten()(model)
    model = tf.keras.layers.Dense(128)(model)
    model = tf.keras.layers.LeakyReLU()(model)
    model = tf.keras.layers.Dropout(0.5)(model)

    output = tf.keras.layers.Dense(num_classes, activation='sigmoid')(model)
    return tf.keras.Model(input, output)









