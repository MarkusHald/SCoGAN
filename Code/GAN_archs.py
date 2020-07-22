import tensorflow as tf
from tensorflow import keras
from Code import Utils as u, Nets as nets

layers = tf.keras.layers


# Regular GANs
def toy_gen(n_dim):
    inputs = keras.Input(shape=(n_dim,), name='digits')
    x = layers.Dense(128, activation='tanh', name='dense1')(inputs)
    x = layers.Dense(128, activation='tanh', name='dense2')(x)
    x = layers.Dense(128, activation='tanh', name='dense3')(x)
    outputs = layers.Dense(2, activation='linear', name='preds')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def toy_disc(args):
    inputs = keras.Input(shape=(args.batch_size, 2), name='digits')
    x = layers.Dense(128, activation='tanh', name='dense1')(inputs)
    x = layers.Dense(128, activation='tanh', name='dense2')(x)
    x = layers.Dense(128, activation='tanh', name='dense3')(x)
    outputs = layers.Dense(1, name='preds')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def cifargan_gen(args):
    g_dim = args.g_dim
    z_dim = args.noise_dim
    img_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    img_resize = img_dim//(2*2*2)

    model = keras.Sequential()
    # foundation for 4x4 image
    model.add(layers.Dense(g_dim * img_resize * img_resize, input_dim=z_dim, kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(layers.Reshape((img_resize, img_resize, g_dim)))
    # upsample to 8x8
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    # output layer
    model.add(layers.Conv2D(channels, (6, 6), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    return model


def cifargan_disc(args):
    d_dim = args.d_dim
    input_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    model = keras.Sequential()

    # normal
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=[input_dim, input_dim, channels], kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    # compile model
    return model


def gan32_gen(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)
    model = u.get_norm(args.norm)(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = u.get_norm(args.norm)(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = u.get_norm(args.norm)(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model = u.get_norm(args.norm)(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    # Generator 1
    img = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model)

    return keras.Model(noise, img)


def gan32_disc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    img = tf.keras.layers.Input(shape=img_shape)
    x = tf.keras.layers.Conv2D(20, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img)
    x = tf.keras.layers.MaxPool2D()(x)

    model = tf.keras.layers.Conv2D(50, (5, 5), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x)
    model = tf.keras.layers.MaxPool2D()(model)
    model = tf.keras.layers.Flatten()(model)
    model = tf.keras.layers.Dense(500, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model)
    model = tf.keras.layers.PReLU(args.prelu_init)(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    model = tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model)

    return keras.Model(img, model)


def gan64_gen(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*16*16)(noise)
    model = tf.keras.layers.Reshape((16, 16, 1024))(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (3,3), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.LeakyReLU(alpha=0.2))(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.LeakyReLU(alpha=0.2))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.LeakyReLU(alpha=0.2))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same'))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.LeakyReLU(alpha=0.2))(model)

    # Generator 1
    img1 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same')(model)

    return keras.Model(noise, img1)


def gan64_disc(args):
    d_dim = args.d_dim
    input_dim = args.dataset_dim[1]
    channels = args.dataset_dim[3]
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    model = keras.Sequential()
    # normal
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1))
    # compile model
    return model


def gan128_gen(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4, kernel_initializer=args.w_init, kernel_regularizer=args.wd)(noise)
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

    return keras.Model(noise, img1)


def gan128_disc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    img1 = tf.keras.layers.Input(shape=img_shape)

    x1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(args.prelu_init)(x1)

    x1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(args.prelu_init)(x1)

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

    output1 = model(x1)

    return keras.Model(img1, output1)


def gan256_gen(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(2048*4*4, kernel_regularizer=args.wd)(noise)
    model = tf.keras.layers.Reshape((4, 4, 2048))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(1024, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(512, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    model = (tf.keras.layers.BatchNormalization(momentum=0.8))(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    img1 = (tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', kernel_regularizer=args.wd))(model)
    img1 = (tf.keras.layers.BatchNormalization(momentum=0.8))(img1)
    img1 = (tf.keras.layers.PReLU(args.prelu_init))(img1)
    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_regularizer=args.wd)(img1)

    return keras.Model(noise, img1)


def gan256_disc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    img1 = tf.keras.layers.Input(shape=img_shape)
    x1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd)(img1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(args.prelu_init)(x1)

    x1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.PReLU(args.prelu_init)(x1)

    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(2048, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, kernel_regularizer=args.wd))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.PReLU(args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1, kernel_regularizer=args.wd))

    output1 = model(x1)

    return keras.Model(img1, output1)


# Multi output GANs
def gan32_gen_multi(args):
    channels = args.dataset_dim[3]
    output1 = []

    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4, kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.PReLU(args.prelu_init)(model)

    model1 = (tf.keras.layers.Conv2DTranspose(512, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)

    model1 = (tf.keras.layers.BatchNormalization())(model1)
    features1_8x8 = (tf.keras.layers.PReLU(args.prelu_init))(model1)
    output1.append(features1_8x8)

    model1 = (tf.keras.layers.Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same',kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(features1_8x8)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    features1_16x16 = (tf.keras.layers.PReLU(args.prelu_init))(model1)
    output1.append(features1_16x16)

    model1 = (tf.keras.layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(features1_16x16)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    model1 = (tf.keras.layers.PReLU(args.prelu_init))(model1)
    img1 = tf.keras.layers.Conv2DTranspose(channels, (6,6), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(model1)

    output1.append(img1)

    return keras.Model(noise, output1)


def gan128_gen_multi(args):
    channels = args.dataset_dim[3]
    output1 = []

    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024 * 4 * 4, kernel_initializer=args.w_init, kernel_regularizer=args.wd,
                                  bias_initializer=args.bi)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)
    model = (tf.keras.layers.BatchNormalization())(model)
    model = (tf.keras.layers.PReLU(args.prelu_init))(model)

    model1 = (tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(model)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    feature1_1 = (tf.keras.layers.PReLU(args.prelu_init))(model1)
    output1.append(feature1_1)

    model1 = (tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(feature1_1)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    feature2_1 = (tf.keras.layers.PReLU(args.prelu_init))(model1)
    output1.append(feature2_1)

    model1 = (tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init,kernel_regularizer=args.wd, bias_initializer=args.bi))(feature2_1)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    feature3_1 = (tf.keras.layers.PReLU(args.prelu_init))(model1)
    output1.append(feature3_1)

    model1 = (tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(feature3_1)
    model1 = (tf.keras.layers.BatchNormalization())(model1)
    feature4_1 = (tf.keras.layers.PReLU(args.prelu_init))(model1)
    output1.append(feature4_1)

    img1 = (tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))(feature4_1)
    img1 = (tf.keras.layers.BatchNormalization())(img1)
    img1 = (tf.keras.layers.PReLU(args.prelu_init))(img1)
    img1 = tf.keras.layers.Conv2DTranspose(channels, (3, 3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    output1.append(img1)

    return keras.Model(noise, output1)


# Resnet GANs
def resnet128_gen(args):
    channels = args.dataset_dim[3]

    # Shared weights between generators
    noise = tf.keras.layers.Input(shape=(args.noise_dim,))

    model = tf.keras.layers.Dense(1024*4*4, kernel_initializer=args.w_init)(noise)
    model = tf.keras.layers.Reshape((4, 4, 1024))(model)
    model = u.get_norm(args.norm)(model)
    model = layers.Activation('relu')(model)

    model = (tf.keras.layers.Conv2DTranspose(512, 3, strides=(2, 2), padding='same', kernel_initializer=args.w_init))(model)
    model = u.get_norm(args.norm)(model)
    model = layers.Activation('relu')(model)

    model = (tf.keras.layers.Conv2DTranspose(256, 3, strides=(2, 2), padding='same', kernel_initializer=args.w_init))(model)
    model = u.get_norm(args.norm)(model)
    model = layers.Activation('relu')(model)

    for i in range(6):
        model = nets.res_net_block(model, 256, 3, args.norm, args.w_init)

    model = (tf.keras.layers.Conv2DTranspose(128, 3, strides=(2, 2), padding='same', kernel_initializer=args.w_init))(model)
    model = u.get_norm(args.norm)(model)
    model = layers.Activation('relu')(model)

    model = (tf.keras.layers.Conv2DTranspose(64, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init))(model)
    model = u.get_norm(args.norm)(model)
    model = layers.Activation('relu')(model)

    img1 = (tf.keras.layers.Conv2DTranspose(32, (3,3), strides=(2, 2), padding='same', kernel_initializer=args.w_init))(model)
    img1 = u.get_norm(args.norm)(img1)
    img1 = layers.Activation('relu')(img1)

    img1 = tf.keras.layers.Conv2DTranspose(channels, (3,3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer=args.w_init)(img1)

    return keras.Model(noise, img1)


def resnet128_disc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    img1 = tf.keras.layers.Input(shape=img_shape)

    x1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = u.get_norm(args.norm)(x1)
    x1 = tf.keras.layers.PReLU(args.args.prelu_init)(x1)

    x1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(img1)
    x1 = u.get_norm(args.norm)(x1)
    x1 = tf.keras.layers.PReLU(args.args.prelu_init)(x1)

    x1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x1)
    x1 = u.get_norm(args.norm)(x1)
    x1 = tf.keras.layers.PReLU(args.args.prelu_init)(x1)

    x1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi)(x1)
    x1 = u.get_norm(args.norm)(x1)
    x1 = tf.keras.layers.PReLU(args.args.prelu_init)(x1)

    model = keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(u.get_norm(args.norm))
    model.add(tf.keras.layers.PReLU(args.args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(u.get_norm(args.norm))
    model.add(tf.keras.layers.PReLU(args.args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(u.get_norm(args.norm))
    model.add(tf.keras.layers.PReLU(args.args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(u.get_norm(args.norm))
    model.add(tf.keras.layers.PReLU(args.args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same', kernel_initializer=args.w_init, kernel_regularizer=args.wd, bias_initializer=args.bi))
    model.add(u.get_norm(args.norm))
    model.add(tf.keras.layers.PReLU(args.args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048, kernel_initializer=args.w_init, kernel_regularizer=args.wd))
    model.add(u.get_norm(args.norm))
    model.add(tf.keras.layers.PReLU(args.args.prelu_init))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1, kernel_initializer=args.w_init, kernel_regularizer=args.wd))

    output1 = model(x1)

    return keras.Model(img1, output1)


def patch_gan_disc(args):
    img_shape = (args.dataset_dim[1], args.dataset_dim[2], args.dataset_dim[3])

    img1 = tf.keras.layers.Input(shape=img_shape)

    x = tf.keras.layers.Conv2D(32, 4, strides=(2, 2), padding='same', kernel_initializer=args.w_init)(img1)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(64, 4, strides=(2, 2), padding='same', kernel_initializer=args.w_init)(x)
    x = u.get_norm(args.norm)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(128, 4, strides=(2, 2), padding='same', kernel_initializer=args.w_init)(x)
    x = u.get_norm(args.norm)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(256, 4, strides=(2, 2), padding='same', kernel_initializer=args.w_init)(x)
    x = u.get_norm(args.norm)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    out = tf.keras.layers.Conv2D(1, (4, 4), padding='same', kernel_initializer=args.w_init)(x)

    return keras.Model(img1, out)