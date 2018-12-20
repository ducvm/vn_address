#!/usr/bin/env python
# coding: utf-8

# In[12]:


import argparse
import os

import keras
import numpy as np
import tensorflow as tf
from keras import applications
from keras import backend as K
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.layers import (Activation, BatchNormalization, Bidirectional, Dense,
                          Dropout, Input, Lambda, Reshape)
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import  Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adadelta, Adam
from sklearn.model_selection import KFold

from DataLoader import (CHAR_DICT, MAX_LEN, SIZE, TextImageGenerator, VizCallback,
                    ctc_lambda_func)


# In[13]:


def get_model(input_shape):
    #inputs = Input(name='the_inputs', shape=input_shape, dtype='float32')
    #base_model = applications.VGG16(weights='imagenet', include_top=False)
    #inner = base_model(inputs)
    #inner = Reshape(target_shape=(
    #    int(inner.shape[1]), -1), name='reshape')(inner)
    #inner = Dense(512, activation='relu',
    #              kernel_initializer='he_normal', name='dense1')(inner)
    #inner = Dropout(0.25)(inner)
    
    # EDIT =========
    inputs = Input(name='the_inputs', shape=input_shape, dtype='float32')
    # First Layer: Conv (5x5) + Pool (2x2) - Output size: 400 x 32 x 64
    cnn = Conv2D(filters = 64,
                 kernel_size = (5, 5),
                 strides=(1, 1),
                 padding = 'same',
                 activation ='relu',
                 input_shape = input_shape)(inputs)
    cnn = MaxPooling2D(pool_size=(2,2), padding='same')(cnn)

    # Second Layer: Conv (5x5) - Output size: 400 x 32 x 128
    cnn = Conv2D(filters = 128,
                   kernel_size = (5, 5),
                   strides=(1, 1),
                   padding = 'same',
                   activation ='relu')(cnn)

    # Third Layer: Conv (3x3) + Pool (2x2) + BN- Output size: 200 x 16 x 128
    cnn = Conv2D(filters = 128,
                   kernel_size = (3, 3),
                   strides=(1, 1),
                   padding = 'same',
                   activation ='relu')(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), padding='same')(cnn)
    cnn = BatchNormalization()(cnn)

    # Fourth Layer: Conv (3x3) - Output size: 200 x 16 x 256
    cnn = Conv2D(filters=256,
                   kernel_size=(3, 3),
                   padding='same',
                   strides=(1, 1),
                   activation='relu')(cnn)

    # Five Layer: conv(3x3)- Output size: 200 x 16 x 256
    cnn = Conv2D(filters=256,
                   kernel_size=(3, 3),
                   padding='same',
                   strides=(1, 1),
                   activation='relu')(cnn)

    # Sixth Layer: Conv (3x3) Output size: 200 x 16 x 512
    cnn = Conv2D(filters=512,
                   kernel_size=(3,3),
                   padding='same',
                   strides=(1,1),
                   activation='relu')(cnn)
    cnn = BatchNormalization()(cnn)

    # Seventh Layer: Conv (3x3) + Pool (2x2) - Output size: 100 x 8 x 512
    cnn = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   padding='same',
                   strides=(1,1),
                   activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), padding='same')(cnn)
    
    # cuts down input size going into RNN:
    inner = Reshape(target_shape=(int(cnn.shape[1]), -1), name='reshape')(cnn)
    inner = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)
    inner = Dropout(0.25)(inner)
    # ==============
    
    lstm = Bidirectional(LSTM(512, return_sequences=True, name='lstm1'))(inner)

    y_pred = Dense(CHAR_DICT, activation='softmax',
                   kernel_initializer='he_normal', name='dense2')(lstm)

    labels = Input(name='the_labels', shape=[MAX_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])

    y_func = K.function([inputs], [y_pred])

    Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out).summary()
    return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out), y_func


# In[14]:


def train_kfold(idx, kfold, datapath, labelpath, epochs, batch_size, lr, finetune=False):
    sess = tf.Session()
    K.set_session(sess)

    model, y_func = get_model((*SIZE, 1))
    ada = Adam(lr=lr)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

    # load data
    train_idx, valid_idx = kfold[idx]
    train_generator = TextImageGenerator(datapath, labelpath, *SIZE, batch_size, 8, train_idx, True, MAX_LEN)
    train_generator.build_data()
    valid_generator = TextImageGenerator(
        datapath, labelpath, *SIZE, batch_size, 8, valid_idx, False, MAX_LEN)
    valid_generator.build_data()

    # callbacks
    weight_path = '../model/best_%d.h5' % idx
    ckp = ModelCheckpoint(weight_path, monitor='val_loss',
                          verbose=1, save_best_only=True, save_weights_only=True)
    vis = VizCallback(sess, y_func, valid_generator, len(valid_idx))
    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    if finetune:
        print('load pretrain model')
        model.load_weights(weight_path)

    model.fit_generator(generator=train_generator.next_batch(),
                        steps_per_epoch=int(len(train_idx) / batch_size),
                        #steps_per_epoch=1,
                        epochs=epochs,
                        callbacks=[ckp, vis, earlystop, tensorboard],
                        validation_data=valid_generator.next_batch(),
                        validation_steps=int(len(valid_idx) / batch_size))
                        #validation_steps=1)

# In[15]:


def train(datapath, labelpath, epochs, batch_size, lr, finetune=False):
    nsplits = 5

    nfiles = np.arange(len(os.listdir(datapath))-1)

    kfold = list(KFold(nsplits, random_state=2018).split(nfiles))
    for idx in range(nsplits):
        train_kfold(idx, kfold, datapath, labelpath,
                    epochs, batch_size, lr, finetune if nsplits==0 else False)


# In[18]:


train("../data/",
      "/../data/lables.json",
      50,
      25,
      0.01)


# In[ ]:




