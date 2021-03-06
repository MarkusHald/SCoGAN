import tensorflow as tf
import numpy as np
import time
from Code import Utils as u, Data as dt, GAN_trainer as gan_t, CoGAN_trainer as cogan_t, SCoGAN_trainer as scogan_t
import argparse
import os.path

parser = argparse.ArgumentParser()

# Settings
parser.add_argument('--dataset',        type=str,           default='toy',      help=' toy | mnist | cifar10 | lsun | frey | svhn')
parser.add_argument('--loss',           type=str,           default='ce',       help=' wgan | ce')
parser.add_argument('--disc_penalty',   type=str,           default='none',       help='none | wgan-gp')
parser.add_argument('--gen_penalty',    type=str,           default='none',       help='weight | feature')
parser.add_argument('--batch_size',     type=int,           default=128)
parser.add_argument('--epochs',         type=int,           default=5000)
parser.add_argument('--disc_iters',     type=int,           default=1)
parser.add_argument('--clip',           type=float,         default=0.01,       help='upper bound for clipping')
parser.add_argument('--penalty_weight_d',      type=int,           default=10)
parser.add_argument('--penalty_weight_g',      type=int,           default=10)
parser.add_argument('--lr_d',           type=float,         default=0.0002)
parser.add_argument('--lr_g',           type=float,         default=0.0002)
parser.add_argument('--b1',             type=float,         default=0.5)
parser.add_argument('--b2',             type=float,         default=0.999)
parser.add_argument('--optim_d',        type=str,           default='adam',     help='adam | sgd | rms')
parser.add_argument('--optim_g',        type=str,           default='adam',     help='adam | rms')
parser.add_argument('--num_samples_to_gen', type=int,       default=8)
parser.add_argument('--images_while_training', type=int,    default=1,          help='Every x epoch to print images while training')
parser.add_argument('--dir',            type=str,           default='C:/Users/palmi/Desktop/samples',     help='Directory to save images, models, weights etc')
parser.add_argument('--g_dim',          type=int,           default=256,        help='generator layer dimensions')
parser.add_argument('--d_dim',          type=int,           default=64,         help='discriminator layer dimensions')
parser.add_argument('--gan_type',       type=str,           default='cogan',    help='64 | 128 | cifargan | cogan | classifier')
parser.add_argument('--noise_dim',      type=int,           default=100,        help='size of the latent vector')
parser.add_argument('--limit_dataset',  type=bool,          default=False,      help='limit dataset to one class')
parser.add_argument('--scale_data',     type=int,           default=0,          help='Scale images in dataset to MxM')
parser.add_argument('--label_smooth',   type=bool,          default=False,      help='Smooth the labels of the disc from 1 to 0 occasionally')
parser.add_argument('--input_noise',    type=bool,          default=False,      help='Add gaussian noise to the discriminator inputs')
parser.add_argument('--purpose',        type=str,		    default='',		    help='purpose of this experiment')
parser.add_argument('--grayscale',      type=bool,		    default=False)
parser.add_argument('--weight_decay',   type=float,         default=0.0005)
parser.add_argument('--bias_init',      type=float,         default=0.1)
parser.add_argument('--prelu_init',     type=float,         default=0.25)
parser.add_argument('--norm',           type=str,           default='batch',     help='batch | layer | instance')
parser.add_argument('--noise_type',     type=str,           default='uniform',   help='normal | uniform')
parser.add_argument('--weight_init',    type=str,           default='normal',   help='normal (0.02 mean)| xavier | he')


# CoGAN
parser.add_argument('--g_arch',         type=str,           default='digit_noshare',       help='digit | rotate | 256 | face | digit_noshare | face_noshare')
parser.add_argument('--d_arch',         type=str,           default='digit_noshare',       help='digit | rotate | 256 | face | digit_noshare | face_noshare')
parser.add_argument('--cogan_data',     type=str,           default='mnist2edge',  help='mnist2edge | mnist2rotate | mnist2svhn | mnist2negative | celeb_a | apple2orange | horse2zebra | vangogh2photo')
parser.add_argument('--semantic_loss',  type=bool,          default=False, help='Determines whether semantic loss is used')
parser.add_argument('--semantic_weight',type=int,           default=10, help='Weight of the semantic loss term')
parser.add_argument('--classifier_path',type=str,           default=None, help='Path to the classifier used for semantic loss')
parser.add_argument('--use_cycle',      type=bool,          default=False, help='Turn on the cycle consistency loss')
parser.add_argument('--cycle_weight',   type=int,           default=10, help='Weight for the cycle gan loss')
parser.add_argument('--shared_layers',  type=int,            default=2, help='Number of layers to calculate feature/weight regularizer from')
parser.add_argument('--feature_loss',   type=bool,          default=False, help="Use vgg to extract features used for regularizing")
parser.add_argument('--fl_high_weight', type=int,           default=1, help='Weight for high level feature similarity')
parser.add_argument('--fl_low_weight',  type=int,           default=1, help='Weight for low level feature similarity')
parser.add_argument('--perceptual_loss',type=bool,         default=False, help='For using perceptual loss')
parser.add_argument('--style_weight',   type=float,           default=-1, help='If -1 use proportional to content weight, else use set value')
parser.add_argument('--content_weight', type=float,         default=1, help='Weight for content loss')

args = parser.parse_args()

# Debugging

args.gan_type = "scogan"
args.loss = 'ce'
args.dir = 'C:/Users/marku/Desktop/gan_training_output/testing'
args.g_arch = '32'
args.d_arch = '32'
args.batch_size = 8
args.cogan_data = 'mnist2edge'
args.perceptual_loss = True
args.style_weight = 0.000001
#args.dataset = 'celeba'
#args.disc_penalty = 'wgan-gp'
#args.gen_penalty = 'feature'
#args.scale_data = 64
#args.epochs = 2
#args.disc_iters = 1
#args.images_while_training = 10
#args.limit_dataset = True

if args.style_weight == -1:
    args.style_weight = args.content_weight / 8e-4


args.wd = tf.keras.regularizers.l2(args.weight_decay)
args.bi = tf.keras.initializers.Constant(args.bias_init)
args.w_init = u.select_weight_init(args.weight_init)
args.prelu_init = tf.keras.initializers.Constant(args.prelu_init)

# We will reuse this seed overtime for visualization
args.seed = u.gen_noise(args, gen_noise_seed=True)

# Set random seeds for reproducability
tf.random.set_seed(2020)
np.random.seed(2020)

#st.style_transfer([image_celeb], args, 'C:/Users/marku/Desktop/gan_training_output/perceptual/sw_0.00000000001_cw_0.001/20k/celeba/41735/generator1', verbose=True)
#u.latent_walk('C:/users/marku/Desktop/gan_training_output/relax_weight_sharing/26508/generator1','C:/Users/marku/Desktop/gan_training_output/relax_weight_sharing/26508/generator2',100,3)

u.select_optimisers(args)

# Choose gan type
if args.gan_type == 'cogan':

    # Choose data
    start = time.time()
    domain1, domain2, shape = dt.select_dataset_multigan(args)
    args.dataset_dim = shape
    data_load_time = time.time() - start

    # Select architectures
    generator1, generator2, discriminator1, discriminator2 = u.select_cogan_architecture(args)

    # Write config
    u.write_config(args)

    # Start training
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        print('Using GPU')
        ganTrainer = cogan_t.CoGANTrainer(generator1, generator2, discriminator1, discriminator2, domain1, domain2)
        full_training_time = ganTrainer.train(args)
    else:
        print('Using CPU')
        ganTrainer = cogan_t.CoGANTrainer(generator1, generator2, discriminator1, discriminator2, domain1, domain2)
        full_training_time = ganTrainer.train(args)

    generator1._name = 'gen1'
    discriminator1._name = 'disc1'
    generator2._name = 'gen2'
    discriminator2._name = 'disc2'

    with open(os.path.join(args.dir, 'config.txt'), 'a') as file:
        file.write('\nFull training time: ' + str(full_training_time) + '\nData load time: ' + str(data_load_time) + '\n')
        generator1.summary(print_fn=lambda x: file.write(x + '\n'))
        discriminator1.summary(print_fn=lambda x: file.write(x + '\n'))
        generator2.summary(print_fn=lambda x: file.write(x + '\n'))
        discriminator2.summary(print_fn=lambda x: file.write(x + '\n'))

    generator1.save(args.dir + '/generator1')
    discriminator1.save(args.dir + '/discriminator1')
    generator2.save(args.dir + '/generator2')
    discriminator2.save(args.dir + '/discriminator2')
    if args.use_cycle:
        ganTrainer.encoder.save(args.dir + '/encoder')

elif args.gan_type == 'scogan':
    # Choose data
    start = time.time()
    domain1, domain2, shape = dt.select_dataset_multigan(args)
    args.dataset_dim = shape
    data_load_time = time.time() - start

    # Select architectures
    generator1, generator2, discriminator1, discriminator2 = u.select_scogan_architecture(args)

    # Write config
    u.write_config(args)

    # Start training
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        print('Using GPU')
        ganTrainer = scogan_t.SCoGANTrainer(generator1, generator2, discriminator1, discriminator2, domain1, domain2)
        full_training_time = ganTrainer.train(args)
    else:
        print('Using CPU')
        ganTrainer = scogan_t.SCoGANTrainer(generator1, generator2, discriminator1, discriminator2, domain1, domain2)
        full_training_time = ganTrainer.train(args)

    generator1._name = 'gen1'
    discriminator1._name = 'disc1'
    generator2._name = 'gen2'
    discriminator2._name = 'disc2'

    with open(os.path.join(args.dir, 'config.txt'), 'a') as file:
        file.write('\nFull training time: ' + str(full_training_time) + '\nData load time: ' + str(data_load_time) + '\n')
        generator1.summary(print_fn=lambda x: file.write(x + '\n'))
        discriminator1.summary(print_fn=lambda x: file.write(x + '\n'))
        generator2.summary(print_fn=lambda x: file.write(x + '\n'))
        discriminator2.summary(print_fn=lambda x: file.write(x + '\n'))

    generator1.save(args.dir + '/generator1')
    discriminator1.save(args.dir + '/discriminator1')
    generator2.save(args.dir + '/generator2')
    discriminator2.save(args.dir + '/discriminator2')
    if args.use_cycle:
        ganTrainer.encoder.save(args.dir + '/encoder')

elif args.gan_type =='sampler':
    np.load('c:/users/palmi/desktop/Experiments/39056/losses/')

    g1 = tf.keras.models.load_model('c:/users/palmi/desktop/Experiments/39056/generator1')
    g2 = tf.keras.models.load_model('c:/users/palmi/desktop/Experiments/39056/generator2')

    ct_trainer = cogan_t.CoGANTrainer(g1,g2,None,None,None,None)

    ct_trainer.sample_images(epoch=6000, seed=args.seed, dir='c:/users/palmi/desktop/experiments/39056/',channels=3)

elif args.gan_type == 'classifier':
    num_classes = 13

    #x1, x2, shape = data.select_dataset_cogan(args)
    #newDataset = x1.concatenate(x2)
    #newTestSet = t1.concatenate(t2).batch(10000)
    #newDataset = newDataset.shuffle(120000).repeat().batch(batch_size=args.batch_size)
    #it2 = iter(newDataset)
    #it_test = iter(newTestSet)

    #celeba a
    x1, x2 = Data.load_celeba_data_classifier(args.batch_size)
    newDataset = x1[0]
    newTestSet = x2[0]
    labels_train = x1[1]
    labels_eval = x2[1]

    #it1 = iter(x2)
    #batch = next(it1)
    #labels = batch[1]
    #batch = batch[0]
    #batch = 0.5 * batch + 0.5

    #newTestSet = next(it_test)
    #model = tf.keras.models.load_model('classifier2')
    #results = model.evaluate(batch, labels)

    #model = nets.mnist_classifier(args, num_classes)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    model = Nets.celeba_classifier(args, num_classes)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    model.fit(newDataset, labels_train, batch_size=16, epochs=args.epochs, verbose=1, validation_data=(newTestSet, labels_eval))
    score = model.evaluate(newTestSet[0], newTestSet[1], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('classifier')

elif args.gan_type == 'svhn_prune':
    model = tf.keras.models.load_model('classifier2')
    x1,x2,shape = Data.select_dataset_cogan(args)
    img_index = []

    it1 = iter(x2)
    batch = next(it1)
    labels = batch[1]
    batch = batch[0]
    results = model.predict(batch)

    for i in range(0,len(results)):
        max_value = np.max(results[i])
        if max_value < 0.4:
            img_index.append(i)
    remaining = len(img_index)
    """
    batch = 0.5 * batch + 0.5
    fig, axs = plt.subplots(4, 4)
    cnt = 0
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(batch[img_index[cnt], :, :, :])
            axs[i, j].axis('off')
            cnt += 1
    fig.show()
    """
    img_index = frozenset(img_index)
    batch = [i for j, i in enumerate(batch) if j not in img_index]
    newBatch = []
    for index, ele in enumerate(batch):
        newBatch.append(ele.numpy())
    newBatch = np.asarray(newBatch)
    newlen = len(newBatch)
    np.save('c:/users/palmi/desktop/40_svhn.npy',newBatch)

else:
    start = time.time()
    train_dat, shape = dt.select_dataset_gan(args)
    if shape is None:
        args.dataset_dim = train_dat.element_spec[0].shape
    else:
        args.dataset_dim = shape
    data_load_time = time.time() - start
    if args.input_noise:
        args.variance = 0.1

    # Select architectures
    generator, discriminator = u.select_gan_architecture(args)

    # Write config
    u.write_config(args)

    # Start training
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        with tf.device('/GPU:0'):
            print('Using GPU')
            ganTrainer = gan_t.GANTrainer(generator, discriminator, train_dat)
            full_training_time = ganTrainer.train(args)
    else:
        print('Using CPU')
        ganTrainer = gan_t.GANTrainer(generator, discriminator, train_dat)
        full_training_time = ganTrainer.train(args)

    generator._name='gen'
    discriminator._name='disc'

    with open(os.path.join(args.dir, 'config.txt'), 'a') as file:
        file.write('\nFull training time: '+str(full_training_time)+'\nData load time: '+str(data_load_time))
        generator.summary(print_fn=lambda x: file.write(x + '\n'))
        discriminator.summary(print_fn=lambda x: file.write(x + '\n'))

    generator.save(args.dir+'/generator')
    discriminator.save(args.dir+'/discriminator')





