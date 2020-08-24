import tensorflow as tf
from Code import Utils as u
import os.path
import matplotlib.pyplot as plt
import numpy as np

'''
path_a2o = 'C:/Users/marku/Desktop/gan_training_output/perceptual/sw_0.00000000001_cw_0.001/20k/41565'

args.dataset='apple2orange'
dat, shape = dt.select_dataset_gan(args)
it = iter(dat)
b = next(it)[0]
# o 5, 13, 15
# a 3, 16, 19
img = b[5]

g_a2o_1 = tf.keras.models.load_model(path_a2o+'/generator1')
g_a2o_2 = tf.keras.models.load_model(path_a2o+'/generator2')

plt.imshow(img* 0.5 + 0.5)
plt.savefig(os.path.join(args.dir, "ST/a2o/1.png"))
latent_a2o = u.find_latent_code(img, g_a2o_1, args, True, 1000)
image_a2o_stylized = g_a2o_2(latent_a2o)[-1]
plt.imshow(image_a2o_stylized[0]* 0.5 + 0.5)
plt.savefig(os.path.join(args.dir, "ST/a2o/1_transferred.png"))
image_a2o_stylized = g_a2o_1(latent_a2o)[-1]
plt.imshow(image_a2o_stylized[0]* 0.5 + 0.5)
plt.savefig(os.path.join(args.dir, "ST/a2o/1_transferred_og.png"))



path_mnist = 'C:/Users/marku/Desktop/gan_training_output/weight_penalty/39737'

args.dataset='mnist'
dat, shape = dt.select_dataset_gan(args)
it = iter(dat)
b = next(it)[0]

image_mnist = b[8]

g_mnist_1 = tf.keras.models.load_model(path_mnist + '/generator1')
g_mnist_2 = tf.keras.models.load_model(path_mnist + '/generator2')

plt.imshow(image_mnist[:,:,0]* 0.5 + 0.5, cmap='gray')
plt.savefig(os.path.join(args.dir, "ST/mnist2edge/3.png"))
latent_mnist = u.find_latent_code(image_mnist, g_mnist_1, args, False, 500)
image_mnist_stylized = g_mnist_2(latent_mnist)
plt.imshow(image_mnist_stylized[0][:,:,0]* 0.5 + 0.5, cmap='gray')
plt.savefig(os.path.join(args.dir, "ST/mnist2edge/3_transferred.png"))
'''

class Style_Transferer(object):

    def __init__(self, save_path, gen_path1, gen_path2="empty", feature_reg=False, verbose=False, iterations=1000):
        self.save_path = save_path
        self.gen_path1 = gen_path1
        self.gen_path2 = gen_path2
        self.feature_reg = feature_reg
        self.verbose = verbose
        self.iterations = iterations
        self.layer_names = ['block1_conv1','block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
        self.g1 = tf.keras.models.load_model(self.gen_path1)
        self.input_shape = self.g1.inputs[0].shape[1]
        if self.gen_path2 != "empty":
            self.g2 = tf.keras.models.load_model(self.gen_path2)
        self.vgg = self.feature_layers()

    def style_transfer(self, content_images):
        latents = np.empty([len(content_images),100])

        for idx,img in enumerate(content_images):
            plt.imshow(img * 0.5 + 0.5)
            plt.savefig(os.path.join(self.save_path, str(idx)+".png"))
            latent = self.find_latent_code(img)
            self.g1 = tf.keras.models.load_model(self.gen_path1)
            latents[idx] = tf.identity(latent)

        stylized_imgs = self.g1(latents)[-1]
        for idx,stylized_img in enumerate(stylized_imgs):
            plt.imshow(tf.squeeze(stylized_img) * 0.5 + 0.5)
            plt.savefig(os.path.join(self.save_path, str(idx)+"_transferred.png"))

        if self.gen_path2 != "empty":
            stylized_imgs = self.g1(latents)[-1]
            for idx, stylized_img in enumerate(stylized_imgs):
                plt.imshow(tf.squeeze(stylized_img) * 0.5 + 0.5)
                plt.savefig(os.path.join(self.save_path, str(idx) + "_same_domain.png"))

    def find_latent_code(self, content_image):
        noise = tf.random.uniform(shape=(1, self.input_shape), minval=-1., maxval=1.)
        x = tf.Variable(noise, trainable=True)
        opt = tf.optimizers.Adam(learning_rate=0.001)

        #content_image = tf.expand_dims(content_image,0)
        #content_image = (0.5 * content_image + 0.5) * 255
        #content_image = tf.keras.applications.vgg19.preprocess_input(content_image)
        #content_image = self.vgg(content_image)

        for i in range(self.iterations):
            with tf.GradientTape() as t:
                # no need to watch a variable:
                # trainable variables are always watched

                img_guess = self.generate(x)

                #loss = self.calc_loss(content_image, self.vgg(img_guess))
                loss = self.calc_loss(content_image, img_guess)

                if self.verbose and i % 250 == 0:
                    print(str(i)+"/"+str(self.iterations))

            # Is the tape that computes the gradients!
            trainable_variables = [x]
            gradients = t.gradient(loss, trainable_variables)
            # The optimize applies the update, using the variables
            # and the optimizer update rule
            opt.apply_gradients(zip(gradients, trainable_variables))
        return x

    def generate(self, input):
        if self.feature_reg:
            return self.g1(input)[-1]
        else:
            return self.g1(input)

    def feature_layers(self):
        vgg = tf.keras.applications.VGG19(include_top=False)
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in self.layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def calc_loss(self, content_img, guess_img):
        # diff = tf.math.squared_difference(self.content_image, img_guess)
        diff = tf.math.abs(content_img - guess_img)
        return tf.math.reduce_mean(diff)


image_celeb = plt.imread('C:/Users/marku/Desktop/img_align_celeba/000008.jpg')
image_celeb2 = plt.imread('C:/Users/marku/Desktop/img_align_celeba/000022.jpg')
image_celeb3 = plt.imread('C:/Users/marku/Desktop/img_align_celeba/000332.jpg')
image_celeb4 = plt.imread('C:/Users/marku/Desktop/img_align_celeba/000099.jpg')
imgs = [image_celeb, image_celeb2, image_celeb3, image_celeb4]
imgs = [image_celeb3, image_celeb4]

tf.random.set_seed(2020)

imgs = map(lambda x: (x - 127.5) / 127.5, imgs)
imgs = map(lambda x: tf.convert_to_tensor(x), imgs)
imgs = map(lambda x: tf.image.central_crop(x, 0.8), imgs)
imgs = map(lambda x: tf.image.resize(x, [128, 128]), imgs)
imgs = map(lambda x: tf.cast(x, tf.float32), imgs)
imgs = list(imgs)


ST = Style_Transferer('C:/Users/marku/Desktop/gan_training_output/ST/testing',
                      'C:/Users/marku/Desktop/gan_training_output/perceptual/sw_0.00000000001_cw_0.001/20k/celeba/41735/generator1',
                      iterations=5000,
                      verbose=True,
                      feature_reg=True)
ST.style_transfer(imgs)
