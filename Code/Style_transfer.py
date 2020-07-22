import tensorflow as tf
from Code import Utils as u
import os.path
import matplotlib.pyplot as plt


'''

# Style transfer

path_celeb = 'C:/Users/marku/Desktop/gan_training_output/perceptual/sw_0.00000000001_cw_0.001/20k/celeba/41735'

g_celeb_1 = tf.keras.models.load_model(path_celeb + '/generator1')
g_celeb_2 = tf.keras.models.load_model(path_celeb + '/generator2')

image_celeb = plt.imread('C:/Users/marku/Desktop/img_align_celeba/000008.jpg')
image_celeb = plt.imread('C:/Users/marku/Desktop/img_align_celeba/000022.jpg')
image_celeb = plt.imread('C:/Users/marku/Desktop/img_align_celeba/000332.jpg')

image_celeb = (image_celeb - 127.5) / 127.5
image_celeb = tf.convert_to_tensor(image_celeb)
image_celeb = tf.image.central_crop(image_celeb, 0.7)  # [132, 132, 3])
image_celeb = tf.image.resize(image_celeb, [128, 128])
image_celeb = tf.cast(image_celeb, tf.float32)

plt.imshow(image_celeb* 0.5 + 0.5)
plt.savefig(os.path.join(args.dir, "ST/celeba/3.png"))
latent_celeb = u.find_latent_code(image_celeb, g_celeb_1, args, True, 5000)
image_celeb_stylized = g_celeb_2(latent_celeb)[-1]
plt.imshow(image_celeb_stylized[0]* 0.5 + 0.5)
plt.savefig(os.path.join(args.dir, "ST/celeba/3_transferred.png"))


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


def style_transfer(content_images, args, gen_path1, gen_path2="empty", verbose=False):
    g1 = tf.keras.models.load_model(gen_path1)

    for idx,img in enumerate(content_images):
        plt.imshow(img * 0.5 + 0.5)
        plt.savefig(os.path.join(args.dir, str(idx)+".png"))

        latent_mnist = find_latent_code(img, g1, args, True, 3000, verbose)

        image_mnist_stylized = generate(g1, latent_mnist, args.feature_reg)
        #image_mnist_stylized = g1(latent_mnist)[-1]

        plt.imshow(tf.squeeze(image_mnist_stylized) * 0.5 + 0.5)
        plt.savefig(os.path.join(args.dir, str(idx)+"_transferred.png"))

        if gen_path2 != "empty":
            g2 = tf.keras.models.load_model(gen_path2)
            image_mnist_stylized = generate(g2, latent_mnist, args.feature_reg)
            plt.imshow(image_mnist_stylized * 0.5 + 0.5)
            plt.savefig(os.path.join(args.dir, str(idx)+"_same_domain.png"))


def generate(generator, input, multi=False):
    if multi:
        return generator(input)[-1]
    else:
        return generator(input)


def find_latent_code(content_image, generator, args, feature_loss, iterations=1000, verbose=False):
    x = tf.Variable(u.gen_noise(args, style_transfer=True), trainable=True)
    opt = tf.optimizers.Adam(learning_rate=0.001)

    for i in range(iterations):
        with tf.GradientTape() as t:
            # no need to watch a variable:
            # trainable variables are always watched
            if feature_loss:
                img_guess = generator(x)[-1]
            else:
                img_guess = generator(x)

            diff = tf.math.abs(content_image - img_guess)
            # diff = tf.math.squared_difference(self.content_image, img_guess)
            loss = tf.math.reduce_mean(diff)
            if verbose and i % 250 == 0:
                print(str(i)+"/"+str(iterations))

        # Is the tape that computes the gradients!
        trainable_variables = [x]
        gradients = t.gradient(loss, trainable_variables)
        # The optimize applies the update, using the variables
        # and the optimizer update rule
        opt.apply_gradients(zip(gradients, trainable_variables))
    return x