import tensorflow as tf
from tensorflow import keras
import numpy as np
layers = tf.keras.layers


# Mnist negative + edge
def cogan_generators_digit(args):
    channels = args.dataset_dim[3]

    output1 = []
    output2 = []

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    features_4x4 = (tf.keras.layers.PReLU())(model)
    output1.append(features_4x4)
    output2.append(features_4x4)

    model = (tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    features_8x8 = (tf.keras.layers.PReLU())(model)
    output1.append(features_8x8)
    output2.append(features_8x8)

    model = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU())(model)
    output1.append(model)
    output2.append(model)

    # Generator 1
    img1 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model)

    # Generator 2
    img2 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model)

    output1.append(img1)
    output2.append(img2)
    return keras.Model(noise, output1), keras.Model(noise, output2)


def cogan_discriminators_digit(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(20, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = tf.keras.layers.MaxPool2D()(x1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(20, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)
    x2 = tf.keras.layers.MaxPool2D()(x2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(50, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))

    output1 = model(x1, training=True)
    output2 = model(x2, training=True)

    return keras.Model(img1, output1), keras.Model(img2, output2)


# Mnist rotate
def cogan_generators_rotate(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    # Shared weights between generators
    model = keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_dim=args.noise_dim, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1024, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.BatchNormalization())

    feature_repr = model(noise)

    # Generator 1
    g1 = tf.keras.layers.Dense(np.prod(img_shape), activation='sigmoid', kernel_regularizer=args.wd, bias_initializer=args.bi)(feature_repr)
    img1 = tf.keras.layers.Reshape(img_shape)(g1)

    # Generator 2
    g2 = tf.keras.layers.Dense(np.prod(img_shape), activation='sigmoid', kernel_regularizer=args.wd, bias_initializer=args.bi)(feature_repr)
    img2 = tf.keras.layers.Reshape(img_shape)(g2)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_rotate(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    model1 = tf.keras.layers.Conv2D(20, (5,5), padding='same', kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    model1 = tf.keras.layers.MaxPool2D()(model1)
    model1 = tf.keras.layers.Conv2D(50, (5,5), padding='same', kernel_regularizer=args.wd, bias_initializer=args.bi)(model1)
    model1 = tf.keras.layers.MaxPool2D()(model1)
    model1 = tf.keras.layers.Dense(500, kernel_regularizer=args.wd, bias_initializer=args.bi)(model1)
    model1 = tf.keras.layers.LeakyReLU()(model1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    model2 = tf.keras.layers.Conv2D(20, (5,5), padding='same', kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)
    model2 = tf.keras.layers.MaxPool2D()(model2)
    model2 = tf.keras.layers.Conv2D(50, (5,5), padding='same', kernel_regularizer=args.wd, bias_initializer=args.bi)(model2)
    model2 = tf.keras.layers.MaxPool2D()(model2)
    model2 = tf.keras.layers.Dense(500, kernel_regularizer=args.wd, bias_initializer=args.bi)(model2)
    model2 = tf.keras.layers.LeakyReLU()(model2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(8,8,500)))
    model.add(tf.keras.layers.Dense(1, kernel_regularizer=args.wd, bias_initializer=args.bi))

    validity1 = model(model1)
    validity2 = model(model2)

    return keras.Model(img1, validity1), keras.Model(img2, validity2)


# Faces
def cogan_generators_faces(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    # Generator 1
    img1 = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    img1 = (tf.keras.layers.BatchNormalization())(img1)
    img1 = (tf.keras.layers.PReLU(args.prelu_init))(img1)
    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)

    # Generator 2
    img2 = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    img2 = (tf.keras.layers.BatchNormalization())(img2)
    img2 = (tf.keras.layers.PReLU(args.prelu_init))(img2)
    img2 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_faces(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(args.prelu_init)(x1)
    x1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(args.prelu_init)(x1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU(args.prelu_init)(x2)
    x2 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU(args.prelu_init)(x2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))

    output1 = model(x1, training=True)
    output2 = model(x2, training=True)

    return keras.Model(img1, output1), keras.Model(img2, output2)


# 256x256 CoGANs
def cogan_generators_256(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(2048*4*4, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(noise)
    model = tf.keras.layers.Reshape((4, 4, 2048))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.LeakyReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.LeakyReLU())(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    # Generator 1
    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model)

    # Generator 2
    img2 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model)

    return keras.Model(noise, img1), keras.Model(noise, img2)


def cogan_discriminators_256(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    # Discriminator 1
    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(args.prelu_init)(x1)
    x1 = tf.keras.layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(args.prelu_init)(x1)

    # Discriminator 2
    img2 = tf.keras.layers.Input(shape=img_shape)
    x2 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU(args.prelu_init)(x2)
    x2 = tf.keras.layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.PReLU(args.prelu_init)(x2)

    # Shared discriminator layers
    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv2D(256, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(1024, (5, 5), padding='same', strides=(2, 2), kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))

    output1 = model(x1)
    output2 = model(x2)

    return keras.Model(img1, output1), keras.Model(img2, output2)