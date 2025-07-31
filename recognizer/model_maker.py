from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Permute, Conv2D, ZeroPadding2D, AveragePooling2D, Flatten, concatenate, BatchNormalization, TimeDistributed
from keras.regularizers import l2

def conv_block(input, growth_rate, dropout_rate=None, weight_decay=5e-1):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3,3), kernel_initializer='he_normal', padding='same')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter

def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-1, first_pool=True):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1,1), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    if pooltype == 2:
        if first_pool:
            x = AveragePooling2D((2,1), strides=(2,1))(x)
        else:
            x = AveragePooling2D((1,1), strides=(1,1))(x)
    elif pooltype == 1:
        x = ZeroPadding2D(padding=(0,1))(x)
        x = AveragePooling2D((2,2), strides=(2,1))(x)
    elif pooltype == 3:
        x = AveragePooling2D((2,2), strides=(2,1))(x)
    return x, nb_filter

def dense_cnn(input, nclass):
    _dropout_rate = 0.99
    _weight_decay = 5e-1

    _nb_filter = 64
    x = Conv2D(_nb_filter, (5,5), strides=(2,2), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(_weight_decay))(input)

    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    x, _nb_filter = transition_block(x, 128, _dropout_rate, pooltype=2, weight_decay=_weight_decay, first_pool=True)
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    x, _nb_filter = transition_block(x, 128, _dropout_rate, pooltype=2, weight_decay=_weight_decay, first_pool=False)
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Permute((2,1,3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)
    return y_pred

def build_ocr_model(input_shape=(32, 306, 1), nclass=25):
    inp = Input(shape=input_shape, name='the_input')
    y_pred = dense_cnn(inp, nclass)
    model = Model(inputs=inp, outputs=y_pred)
    return model
