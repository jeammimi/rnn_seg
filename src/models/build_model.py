from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import Concatenate
import theano.tensor as T
import theano
from keras.backend.common import _EPSILON
import numpy as np


def build_model(n_states=10, n_cat=27, n_layers=3, inputsize=5, hidden=50, simple=False, sub=False, segmentation=True, merge_mode="concat"):

    RNN = LSTM
    if simple:
        RNN = SimpleRNN

    inputs = Input(shape=(None, inputsize))

    l1 = Bidirectional(RNN(hidden, return_sequences=True), merge_mode='concat')(inputs)
    to_concat = [l1]

    for j in range(1, n_layers):
        # print(globals())
        locals()["l%i" % (j + 1)] = Bidirectional(RNN(hidden,
                                                      return_sequences=True, activation='tanh'), merge_mode=merge_mode)(Concatenate()([locals()["l%i" % j], inputs]))
        locals()["l%i" % (j + 1)] = TimeDistributed(Dense(hidden,
                                                          activation="linear"))(locals()["l%i" % (j + 1)])
        to_concat.append(locals()["l%i" % (j + 1)])
    to_concat += [inputs]

    output = TimeDistributed(Dense(n_states, activation="softmax"),
                             name="output")(Concatenate()(to_concat))

    cat = Bidirectional(LSTM(n_cat, return_sequences=False), merge_mode=merge_mode)(output)
    if segmentation:
        category = Dense(n_cat, activation="softmax", name="category")(cat)

    def perm_loss(y_true, y_pred):
        def loss(m, y_true, y_pred, perm):

            # return  perm[T.cast(m,"int32")]
            y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            return T.mean(T.sum(y_true[::, ::, perm[m]] * T.log(y_pred), axis=-1), axis=-1)

        # perm = np.array([[0,1],[1,0]],dtype=np.int)
        if sub:
            perm = np.array([[0, 1, 2, 3, 4, 5, 6] + range(7, 10),
                             [0, 1, 2, 4, 5, 3, 6] + range(7, 10),
                             [0, 1, 2, 5, 4, 3, 6] + range(7, 10),
                             [0, 1, 2, 3, 5, 4, 6] + range(7, 10),
                             [0, 1, 2, 4, 3, 5, 6] + range(7, 10),
                             [0, 1, 2, 5, 3, 4, 6] + range(7, 10)], dtype=np.int)
        else:
            perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                             [0, 1, 2, 4, 5, 3, 6],
                             [0, 1, 2, 5, 4, 3, 6],
                             [0, 1, 2, 3, 5, 4, 6],
                             [0, 1, 2, 4, 3, 5, 6],
                             [0, 1, 2, 5, 3, 4, 6]], dtype=np.int)

        seq = T.arange(len(perm))
        result, _ = theano.scan(fn=loss, outputs_info=None,
                                sequences=seq, non_sequences=[y_true, y_pred, perm])
        return -T.mean(T.max(result, axis=0))  # T.max(result.dimshuffle(1,2,0),axis=-1)

    model = Model(inputs=inputs, outputs=[output, category])

    if segmentation:
        model.compile(optimizer='adadelta', loss={
                      'output': perm_loss, "category": "categorical_crossentropy"})
    else:
        model.compile(optimizer='adadelta', loss={"category": "categorical_crossentropy"})

    return model
