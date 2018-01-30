import os
import cv2
import pandas
import numpy
import pickle

from collections import Counter

from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.utils import to_categorical, plot_model

from keras.layers import Conv2D, Dense, MaxPool2D, Input, \
    Flatten, Dropout, BatchNormalization, Activation, LSTM, Reshape

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from keras.preprocessing.image import ImageDataGenerator

EPOCHS = 400
BATCH_SIZE = 512
TRAINING_TESTING_RATIO = 0.7

PATIENCE_EPOCHS = 200

TRAINING_THIS_TIME = True

class Logger():
    def __init__(self):
        self.loss_measure = "loss"
        self.acc_measure = "acc"

        from time import gmtime, strftime
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        self.target_path = "log/" + timestamp

        if not os.path.exists(self.target_path):
            os.mkdir(self.target_path)

        import signal
        signal.signal(signal.SIGINT, self.signalHandler)

    def signalHandler(self, signal, frame):
        print "system signal (interrupt or kill) received"
        import shutil
        shutil.rmtree(self.target_path)
        import sys
        sys.exit(1)

    def getLoggerName(self):
        return self.target_path + "/logger.csv"

    def getWeightName(self):
        return self.target_path + "/weights.last.hdf5"

    def loadWeight(self, model, filename=None):
        if filename:
            model.load_weights(filename)
        else:
            model.load_weights(self.target_path + "/weights.last.hdf5")

    def saveHistory(self, history):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        fig = pyplot.figure(figsize=(8, 10))
        pyplot.subplot(211)
        pyplot.plot(history.history[self.loss_measure])
        pyplot.plot(history.history["val_"+self.loss_measure])
        pyplot.title("Training & testing loss curve")
        pyplot.xlabel("epoch")
        pyplot.ylabel("loss")
        pyplot.legend(["train", "test"], loc='upper right')

        pyplot.subplot(212)
        pyplot.plot(history.history[self.acc_measure])
        pyplot.plot(history.history["val_"+self.acc_measure])
        pyplot.title("Training & testing acc curve")
        pyplot.xlabel("epoch")
        pyplot.ylabel("acc")
        pyplot.legend(["train", "test"], loc='upper left')

        fig.savefig(self.target_path + "/history.png" )

    def saveModelArch(self, model):
        from keras.utils import plot_model
        plot_model(model, to_file=self.target_path + "/model.png", show_shapes=True)

    def saveResults(self, true, pred):
        from sklearn.metrics import confusion_matrix, accuracy_score
        conf_matrix = confusion_matrix(true, pred)
        score = accuracy_score(true, pred)
        classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        with open(self.target_path+"/mega_info.txt", "a+") as fout:
            fout.writelines("Training: epoch {}, batch size {}\n".format(EPOCHS, BATCH_SIZE))
            fout.writelines("Accuracy score: {}\n".format(score))
            fout.writelines("Confusion matrix:\n")
            fout.write('{:^10}'.format(' '))
            for each in classes:
                fout.write('{:^10}'.format(each))
            fout.write('\n')
            for i in range(len(classes)):
                fout.write('{:^10}'.format(classes[i]))
                for item in conf_matrix[i]:
                    fout.write('{:^10}'.format(item))
                fout.write('\n')
            fout.write('\n')


def loadData():
    print "loading data..."
    raw_data = pandas.read_csv("data/fer2013.csv")

    raw_data['image'] = raw_data['pixels'].apply(
        lambda x: numpy.fromstring(x, dtype=numpy.uint8, sep=' ').reshape((48, 48, 1))
    )

    cols = ['emotion', 'image']

    training = raw_data[raw_data[u'Usage'] == u"Training"].reset_index(drop=True)
    public_test = raw_data[raw_data[u'Usage'] == u"PublicTest"].reset_index(drop=True)
    private_test = raw_data[raw_data[u'Usage'] == u"PrivateTest"].reset_index(drop=True)

    training = training[cols]
    public_test = public_test[cols]
    private_test = private_test[cols]

    return training, public_test, private_test

def constructModel():
    print "constructing model..."
    model = Sequential()
    input_shape = (48, 48, 1)

    # 48*48*1
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))

    # 24*24*16
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))

    # flatten: 12*12*32
    model.add(Flatten())

    # 1*256
    model.add(Dense(128))
    model.add(Activation('relu'))

    # 1*256
    model.add(Dense(7, activation='softmax'))

    return model

def dataFormatting(data):
    print "formatting..."
    temp_x = []
    temp_y = []

    for index, row in data.iterrows():
        temp_img = numpy.empty_like(row[u'image'], dtype=numpy.float32)
        cv2.normalize(row[u'image'], temp_img, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        temp_x.append(temp_img)
        temp_y.append(row[u'emotion'])

    # plotDistrib(temp_y)

    x = numpy.array(temp_x)
    y = to_categorical(numpy.array(temp_y), 7)

    return x, y

def randomSplit(x, y):
    data_size = x.shape[0]
    ratio = TRAINING_TESTING_RATIO
    numpy.random.seed(seed=1)
    print "random splitting, training ratio: %.2f with fixed seed"%ratio

    permu = numpy.random.permutation(data_size)
    select_bool = permu <= (ratio*data_size)

    return x[select_bool], y[select_bool], x[~select_bool], y[~select_bool]

def sampling(x, y):
    index_array = numpy.arange(len(x))
    y_cat = numpy.argmax(y, axis=1)
    uniq, indices, freq = numpy.unique(y_cat, return_inverse=True, return_counts=True)
    freq_array = freq[indices]
    proport = 1.0/freq_array
    prob = proport/numpy.sum(proport)
    while 1:
        selected_index = numpy.random.choice(index_array, BATCH_SIZE, p=prob)
        yield (x[selected_index], y[selected_index])

'''
def plotDistrib(labels, info=""):
    freq = Counter(labels)
    keys = numpy.round(freq.keys(), decimals=2)
    vals = numpy.array(freq.values())
    sort_index = numpy.argsort(keys)
    keys_sorted = keys[sort_index]
    vals_sorted = vals[sort_index]
    cmap = pyplot.cm.Reds
    color = cmap(numpy.linspace(0.1, 1., len(keys_sorted)))

    _, (fig1, fig2) = pyplot.subplots(1, 2)

    fig1.bar(keys_sorted, vals_sorted, width=0.2, color=color)
    text_str = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    for i in range(len(keys_sorted)):
        fig1.text(keys_sorted[i], vals_sorted[i] + 10, "%s: %d"%(text_str[i], vals_sorted[i]))
    fig1.set_title(info + " (%d in total)"%(len(labels)))
    fig1.set_xlabel("Labels")
    fig1.set_xticks(keys_sorted)
    fig1.set_ylabel("Frequency")

    fig2.pie(vals_sorted, labels=text_str, autopct='%1.2f%%', colors=color)
    fig2.axis('equal')
    fig2.set_title(info + " (%d in total)"%(len(labels)))
    pyplot.draw()
    pyplot.show()

'''

if __name__ == "__main__":
    training, public_test, private_test = loadData()

    x, y = dataFormatting(training)
    x, y, x_val, y_val = randomSplit(x, y)
    x_test, y_test = dataFormatting(public_test)

    total_num = len(x)

    model = constructModel()

    model.compile(loss='categorical_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])

    print "fitting..."
    # checkpoint

    logging = Logger()

    checkpoint = ModelCheckpoint(filepath=logging.getWeightName(),
                                 monitor='val_acc')

    early_stop = EarlyStopping(monitor='val_acc',
                               patience=PATIENCE_EPOCHS,
                               mode='max')

    logger = CSVLogger(filename= logging.getLoggerName(),
                       separator=',',
                       append=False)

    callbacks_list = [checkpoint, early_stop, logger]

    # datagen = ImageDataGenerator(horizontal_flip=True)

    # x_y = datagen.flow(x=x, y=y, batch_size=BATCH_SIZE)

    if TRAINING_THIS_TIME:
        history = model.fit_generator(sampling(x, y),
                                      steps_per_epoch=total_num/BATCH_SIZE,
                                      epochs=EPOCHS,
                                      callbacks=callbacks_list,
                                      validation_data=sampling(x_val, y_val),
                                      validation_steps=total_num/BATCH_SIZE)

    # history = model.fit(x=x,
    #                     y=y,
    #                     epochs=EPOCHS,
    #                     batch_size=BATCH_SIZE,
    #                     callbacks=callbacks_list,
    #                     validation_data=(x_val, y_val))

        logging.saveHistory(history)
        logging.saveModelArch(model)

    model = constructModel()
    logging.loadWeight(model)

    val_pred = model.predict(x_val)

    true_label = numpy.argmax(y_val, axis=1)
    pred_label = numpy.argmax(val_pred, axis=1)

    logging.saveResults(true_label, pred_label)

    test_pred = model.predict(x_test)

    true_label = numpy.argmax(y_test, axis=1)
    pred_label = numpy.argmax(test_pred, axis=1)

    logging.saveResults(true_label, pred_label)

