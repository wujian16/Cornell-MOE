import numpy
from multiprocessing import Process, Queue

# import keras
# from keras import backend as K
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# import tensorflow as tf
# from keras.datasets import cifar10
#
# from oct2py import octave
# octave.addpath(octave.genpath("gpml-matlab-v4.0-2016-10-19"))

def run_in_separate_process(method, args):
    def queue_wrapper(q, params):
        r = method(*params)
        q.put(r)
    q = Queue()
    p = Process(target=queue_wrapper, args=(q, args))
    p.start()
    return_val = q.get()
    p.join()
    if type(return_val) is Exception:
        raise return_val
    return return_val

class Branin(object):
    def __init__(self):
        self._dim = 2
        self._search_domain = numpy.array([[0, 15], [-5, 15]])
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.397887
        self._num_fidelity = 0
        self._num_observations = 0

    def evaluate_true(self, x):
        """ This function is usually evaluated on the square x_1 \in [-5, 10], x_2 \in [0, 15]. Global minimum
        is at x = [-pi, 12.275], [pi, 2.275] and [9.42478, 2.475] with minima f(x*) = 0.397887.

            :param x[2]: 2-dim numpy array
        """
        a = 1
        b = 5.1 / (4 * pow(numpy.pi, 2.0))
        c = 5 / numpy.pi
        r = 6
        s = 10
        t = 1 / (8 * numpy.pi)
        return numpy.array([(a * pow(x[1] - b * pow(x[0], 2.0) + c * x[0] - r, 2.0) + s * (1 - t) * numpy.cos(x[0]) + s),
                (2*a*(x[1] - b * pow(x[0], 2.0) + c * x[0] - r) * (-2* b * x[0] + c) + s * (1 - t) * (-numpy.sin(x[0]))),
                (2*a*(x[1] - b * pow(x[0], 2.0) + c * x[0] - r))])

    def evaluate(self, x):
        return self.evaluate_true(x)

class Rosenbrock(object):
    def __init__(self):
        self._dim = 3
        self._search_domain = numpy.repeat([[-2., 2.]], 3, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.0
        self._num_fidelity = 0
        self._num_observations = 0

    def evaluate_true(self, x):
        """ Global minimum is 0 at (1, 1, 1)

            :param x[3]: 3-dimension numpy array
        """
        value = 0.0
        for i in range(self._dim-1):
            value += pow(1. - x[i], 2.0) + 100. * pow(x[i+1] - pow(x[i], 2.0), 2.0)
        results = [value]
        for i in range(self._dim-1):
            results += [2.*(x[i]-1) - 400.*x[i]*(x[i+1]-pow(x[i], 2.0))]
        results += [200. * (x[self._dim-1]-pow(x[self._dim-2], 2.0))]
        return numpy.array(results)

    def evaluate(self, x):
        return self.evaluate_true(x)

class Hartmann3(object):
    def __init__(self):
        self._dim = 3
        self._search_domain = numpy.repeat([[0., 1.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = -3.86278
        self._num_fidelity = 0
        self._num_observations = 0

    def evaluate_true(self, x):
        """ domain is x_i \in (0, 1) for i = 1, ..., 3
            Global minimum is -3.86278 at (0.114614, 0.555649, 0.852547)

            :param x[3]: 3-dimension numpy array with domain stated above
        """
        alpha = numpy.array([1.0, 1.2, 3.0, 3.2])
        A = numpy.array([[3., 10., 30.], [0.1, 10., 35.], [3., 10., 30.], [0.1, 10., 35.]])
        P = 1e-4 * numpy.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
        value = 0.0
        for i in range(4):
            inner_value = 0.0
            for j in range(self._dim):
                inner_value -= A[i, j] * pow(x[j] - P[i, j], 2.0)
            value -= alpha[i] * numpy.exp(inner_value)
        return numpy.array([value])

    def evaluate(self, x):
        t = self.evaluate_true(x)
        return t

# class CIFAR10(object):
#     def __init__(self):
#         self._dim = 5
#         self._search_domain = numpy.array([[-6, 0], [32, 512], [5, 9], [5, 9], [5, 9]])
#         self._num_init_pts = 1
#         self._sample_var = 0.0
#         self._min_value = 0.0
#         self._num_fidelity = 0
#         self._num_observations = 0
#
#     def train(self, x):
#         try:
#             # Data loading and preprocessing
#             # The data, shuffled and split between train and test sets:
#             num_classes = 10
#             # The data, shuffled and split between train and test sets:
#             (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#
#             # Convert class vectors to binary class matrices.
#             y_train = keras.utils.to_categorical(y_train, num_classes)
#             y_test = keras.utils.to_categorical(y_test, num_classes)
#
#             lr, batch_size, unit1, unit2, unit3 = x
#             lr = pow(10, lr)
#             unit1 = int(pow(2, unit1))
#             unit2 = int(pow(2, unit2))
#             unit3 = int(pow(2, unit3))
#             batch_size = int(batch_size)
#
#             K.clear_session()
#             sess = tf.Session()
#             K.set_session(sess)
#             graphr = K.get_session().graph
#             with graphr.as_default():
#                 num_classes = 10
#                 epochs = 50
#                 data_augmentation = True
#
#                 model = Sequential()
#                 # cov bloack
#                 model.add(Conv2D(unit1, (3, 3), padding='same',
#                                  input_shape=x_train.shape[1:]))
#                 model.add(Activation('relu'))
#                 model.add(Conv2D(unit1, (3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))
#
#                 model.add(Conv2D(unit2, (3, 3), padding='same'))
#                 model.add(Activation('relu'))
#                 model.add(Conv2D(unit2, (3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))
#
#                 model.add(Conv2D(unit3, (3, 3), padding='same'))
#                 model.add(Activation('relu'))
#                 model.add(Conv2D(unit3, (3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))
#
#                 model.add(Flatten())
#                 model.add(Dense(num_classes))
#                 model.add(Activation('softmax'))
#
#                 # initiate RMSprop optimizer
#                 opt = keras.optimizers.Adam(lr=lr)
#
#                 # Let's train the model using RMSprop
#                 model.compile(loss='categorical_crossentropy',
#                               optimizer=opt,
#                               metrics=['accuracy'])
#
#                 x_train = x_train.astype('float32')
#                 x_test = x_test.astype('float32')
#                 x_train /= 255
#                 x_test /= 255
#
#                 if not data_augmentation:
#                     print('Not using data augmentation.')
#                     model.fit(x_train, y_train,
#                               batch_size=batch_size,
#                               epochs=epochs,
#                               validation_data=(x_test, y_test),
#                               shuffle=True)
#                 else:
#                     print('Using real-time data augmentation.')
#                     # This will do preprocessing and realtime data augmentation:
#                     datagen = ImageDataGenerator(
#                             featurewise_center=False,  # set input mean to 0 over the dataset
#                             samplewise_center=False,  # set each sample mean to 0
#                             featurewise_std_normalization=False,  # divide inputs by std of the dataset
#                             samplewise_std_normalization=False,  # divide each input by its std
#                             zca_whitening=False,  # apply ZCA whitening
#                             rotation_range=0.,  # randomly rotate images in the range (degrees, 0 to 180)
#                             width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#                             height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#                             horizontal_flip=True,  # randomly flip images
#                             vertical_flip=False)  # randomly flip images
#
#                     # Compute quantities required for feature-wise normalization
#                     # (std, mean, and principal components if ZCA whitening is applied).
#                     datagen.fit(x_train)
#
#                     # Fit the model on the batches generated by datagen.flow().
#                     model.fit_generator(datagen.flow(x_train, y_train,
#                                                      batch_size=batch_size),
#                                         steps_per_epoch=x_train.shape[0] // batch_size,
#                                         epochs=epochs,
#                                         validation_data=None)
#
#
#                 # Evaluate model with test data set and share sample prediction results
#                 evaluation = model.evaluate_generator(datagen.flow(x_test, y_test,
#                                                                    batch_size=batch_size),
#                                                       steps=x_test.shape[0] // batch_size)
#
#                 print('Model Accuracy = %.2f' % (evaluation[1]))
#                 return 1-evaluation[1]
#         except Exception as e:
#             return e
#
#     def evaluate_true(self, x):
#         loss = run_in_separate_process(self.train, [x])
#         if type(loss) is Exception:
#             raise loss
#         else:
#             return numpy.array([loss])
#
#     def evaluate(self, x):
#         return self.evaluate_true(x)
#
# class KISSGP(object):
#     def __init__(self):
#         self._dim = 3
#         self._search_domain = numpy.array([[-1, 3], [-1, 3], [-1, 3]])
#         self._num_init_pts = 1
#         self._sample_var = 0.0
#         self._min_value = 0.0
#         self._num_fidelity = 0
#         self._num_observations = 3
#
#     def evaluate_true(self, x):
#         value = numpy.array(octave.KISSGP(numpy.exp(x))).flatten()
#         return value
#
#     def evaluate(self, x):
#         return self.evaluate_true(x)