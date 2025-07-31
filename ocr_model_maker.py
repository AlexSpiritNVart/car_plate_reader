# === МОДЕЛЬ: ocr_model.py ===

from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape, Permute, Conv2D, AveragePooling2D, Flatten, concatenate, BatchNormalization, TimeDistributed, Dropout
from keras.regularizers import l2

def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3,3), padding='same', kernel_initializer='he_normal')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter

def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1,1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter

def dense_cnn(input, nclass):
    nb_filter = 64
    dropout_rate = 0.2
    weight_decay = 1e-4

    x = Conv2D(nb_filter, (5,5), strides=(2,2), padding='same',
               use_bias=False, kernel_initializer='he_normal')(input)

    x, nb_filter = dense_block(x, 6, nb_filter, 8, dropout_rate, weight_decay)
    x, nb_filter = transition_block(x, 128, dropout_rate, weight_decay)
    x, nb_filter = dense_block(x, 6, nb_filter, 8, dropout_rate, weight_decay)
    x, nb_filter = transition_block(x, 128, dropout_rate, weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Permute((2,1,3))(x)
    x = TimeDistributed(Flatten())(x)

    y_pred = Dense(nclass, activation='softmax', name='out')(x)
    return y_pred

def build_ocr_model(input_shape=(48, 160, 1), nclass=25):
    inp = Input(shape=input_shape, name='the_input')
    y_pred = dense_cnn(inp, nclass)
    model = Model(inputs=inp, outputs=y_pred)
    return model

