
# coding: utf-8

# In[1]:

from keras.objectives import categorical_crossentropy
import theano.tensor as T
import theano
import numpy as np
# categorical_crossentropy??
# Loss:


perm = [[0, 1, 2], [1, 2, 0], [2, 1, 0], [0, 2, 1], [1, 0, 2], [2, 0, 1]]
perm = [[-3, -2, -1] + iperm for iperm in perm]
perm = np.array(perm, dtype=np.int)
perm += 3
# print perm

test_true = [[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]

eps = 1e-7
test_pred = [[[1 - eps, +eps], [1 - eps, eps], [1 - eps, eps]], [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]],
             [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]]]


def perm_loss(y_true, y_pred):
    def loss(m,  y_true, y_pred, perm):

        # return  perm[T.cast(m,"int32")]
        return T.mean(T.sum(y_true[::, ::, perm[m]] * T.log(y_pred), axis=-1), axis=-1)

    #perm = np.array([[0,1],[1,0]],dtype=np.int)
    perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                     [0, 1, 2, 4, 5, 3, 6],
                     [0, 1, 2, 5, 4, 3, 6],
                     [0, 1, 2, 3, 5, 4, 6],
                     [0, 1, 2, 4, 3, 5, 6],
                     [0, 1, 2, 5, 3, 4, 6]], dtype=np.int)

    """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                     [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
    seq = T.arange(len(perm))
    result, _ = theano.scan(fn=loss, outputs_info=None,
                            sequences=seq, non_sequences=[y_true, y_pred, perm])
    return -T.mean(T.max(result, axis=0))  # T.max(result.dimshuffle(1,2,0),axis=-1)

#r =perm_loss(test_true,test_pred).eval()
# print r
# print r.shape


# In[4]:

import keras
from keras.layers.recurrent import Recurrent
from keras import backend as K
from keras import activations, initializations
if int(keras.__version__.split(".")[0]) >= 1.0:
    # print "v1"
    from keras import activations, initializations, regularizers
    from keras.engine import Layer, InputSpec
    from keras.layers.recurrent import time_distributed_dense

    class BiLSTMv1(Recurrent):
        '''Long-Short Term Memory unit - Hochreiter 1997.

        For a step-by-step description of the algorithm, see
        [this tutorial](http://deeplearning.net/tutorial/lstm.html).

        # Arguments
            output_dim: dimension of the internal projections and the final output.
            init: weight initialization function.
                Can be the name of an existing function (str),
                or a Theano function (see: [initializations](../initializations.md)).
            inner_init: initialization function of the inner cells.
            forget_bias_init: initialization function for the bias of the forget gate.
                [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
                recommend initializing with ones.
            activation: activation function.
                Can be the name of an existing function (str),
                or a Theano function (see: [activations](../activations.md)).
            inner_activation: activation function for the inner cells.
            W_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the input weights matrices.
            U_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
            b_regularizer: instance of [WeightRegularizer](../regularizers.md),
                applied to the bias.
            dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
            dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

        # References
            - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
            - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
            - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        '''

        def __init__(self, output_dim,
                     init='glorot_uniform', inner_init='orthogonal',
                     forget_bias_init='one', activation='tanh',
                     inner_activation='hard_sigmoid',
                     W_regularizer=None, U_regularizer=None, b_regularizer=None,
                     dropout_W=0., dropout_U=0., close=False, **kwargs):
            self.output_dim = output_dim
            self.init = initializations.get(init)
            self.inner_init = initializations.get(inner_init)
            self.forget_bias_init = initializations.get(forget_bias_init)
            self.activation = activations.get(activation)
            self.inner_activation = activations.get(inner_activation)
            self.W_regularizer = regularizers.get(W_regularizer)
            self.U_regularizer = regularizers.get(U_regularizer)
            self.b_regularizer = regularizers.get(b_regularizer)
            self.dropout_W, self.dropout_U = dropout_W, dropout_U
            self.close = close

            if self.dropout_W or self.dropout_U:
                self.uses_learning_phase = True
            super(BiLSTMv1, self).__init__(**kwargs)

        def build(self, input_shape):
            self.input_spec = [InputSpec(shape=input_shape)]
            input_dim = input_shape[2]
            self.input_dim = input_dim

            if self.stateful:
                self.reset_states()
            else:
                # initial states: 2 all-zero tensors of shape (output_dim)
                self.states = [None, None]

            self.W_i = self.init((input_dim, self.output_dim),
                                 name='{}_W_i'.format(self.name))
            self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_i'.format(self.name))
            self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

            self.W_f = self.init((input_dim, self.output_dim),
                                 name='{}_W_f'.format(self.name))
            self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_f'.format(self.name))
            self.b_f = self.forget_bias_init((self.output_dim,),
                                             name='{}_b_f'.format(self.name))

            self.W_c = self.init((input_dim, self.output_dim),
                                 name='{}_W_c'.format(self.name))
            self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_c'.format(self.name))
            self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

            self.W_o = self.init((input_dim, self.output_dim),
                                 name='{}_W_o'.format(self.name))
            self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_o'.format(self.name))
            self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

            if self.close:
                self.W_h = self.init((self.output_dim, self.output_dim),
                                     name='{}_W_h'.format(self.name))
                self.b_h = K.zeros((self.output_dim,),
                                   name='{}_b_h'.format(self.name))

            self.regularizers = []
            if self.W_regularizer:
                if not self.close:
                    self.W_regularizer.set_param(K.concatenate([self.W_i,
                                                                self.W_f,
                                                                self.W_c,
                                                                self.W_o]))
                if self.close:
                    self.W_regularizer.set_param(K.concatenate([self.W_i,
                                                                self.W_f,
                                                                self.W_c,
                                                                self.W_o,
                                                                self.W_h]))

                self.regularizers.append(self.W_regularizer)
            if self.U_regularizer:
                self.U_regularizer.set_param(K.concatenate([self.U_i,
                                                            self.U_f,
                                                            self.U_c,
                                                            self.U_o]))
                self.regularizers.append(self.U_regularizer)
            if self.b_regularizer:
                if not self.close:
                    self.b_regularizer.set_param(K.concatenate([self.b_i,
                                                                self.b_f,
                                                                self.b_c,
                                                                self.b_o]))
                else:
                    self.b_regularizer.set_param(K.concatenate([self.b_i,
                                                                self.b_f,
                                                                self.b_c,
                                                                self.b_o,
                                                                self.b_h]))
                self.regularizers.append(self.b_regularizer)

            self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o]
            if self.close:
                self.trainable_weights += [self.W_h, self.b_h]

            if self.initial_weights is not None:
                self.set_weights(self.initial_weights)
                del self.initial_weights

        def reset_states(self):
            assert self.stateful, 'Layer must be stateful.'
            input_shape = self.input_spec[0].shape
            if not input_shape[0]:
                raise Exception('If a RNN is stateful, a complete ' +
                                'input_shape must be provided (including batch size).')
            if hasattr(self, 'states'):
                K.set_value(self.states[0],
                            np.zeros((input_shape[0], self.output_dim)))
                K.set_value(self.states[1],
                            np.zeros((input_shape[0], self.output_dim)))
            else:
                self.states = [K.zeros((input_shape[0], self.output_dim)),
                               K.zeros((input_shape[0], self.output_dim))]

        def preprocess_input(self, x, train=False):
            if self.consume_less == 'cpu':
                if train and (0 < self.dropout_W < 1):
                    dropout = self.dropout_W
                else:
                    dropout = 0
                input_shape = self.input_spec[0].shape
                input_dim = input_shape[2]
                timesteps = input_shape[1]

                x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                             input_dim, self.output_dim, timesteps)
                x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                             input_dim, self.output_dim, timesteps)
                x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                             input_dim, self.output_dim, timesteps)
                x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                             input_dim, self.output_dim, timesteps)
                return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
            else:
                return x

        def call(self, x, mask=None):

            self.go_backwards = False
            R1 = Recurrent.call(self, x, mask=mask)

            self.go_backwards = True
            R2 = Recurrent.call(self, x, mask=mask)

            if self.return_sequences:
                if K._BACKEND == 'tensorflow':
                    R2 = tf.reverse(R2, [False, True, False])
                else:
                    R2 = R2[::, ::-1, ::]
            if self.close:
                return K.dot(R1 + R2, self.W_h) + self.b_h
            else:
                return R1 / 2 + R2 / 2

        def cell(self, x):
            pass

        def step(self, x, states):
            h_tm1 = states[0]
            c_tm1 = states[1]
            B_U = states[2]
            B_W = states[3]

            if self.consume_less == 'cpu':
                x_i = x[:, :self.output_dim]
                x_f = x[:, self.output_dim: 2 * self.output_dim]
                x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
                x_o = x[:, 3 * self.output_dim:]
            else:
                x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x * B_W[3], self.W_o) + self.b_o

            i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
            f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
            o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

            h = o * self.activation(c)
            return h, [h, c]

        def get_constants(self, x):
            constants = []
            if 0 < self.dropout_U < 1:
                ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                ones = K.concatenate([ones] * self.output_dim, 1)
                B_U = [K.dropout(ones, self.dropout_U) for _ in range(4)]
                constants.append(B_U)
            else:
                constants.append([K.cast_to_floatx(1.) for _ in range(4)])

            if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
                input_shape = self.input_spec[0].shape
                input_dim = input_shape[-1]
                ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                ones = K.concatenate([ones] * input_dim, 1)
                B_W = [K.dropout(ones, self.dropout_W) for _ in range(4)]
                constants.append(B_W)
            else:
                constants.append([K.cast_to_floatx(1.) for _ in range(4)])
            return constants

        def get_config(self):
            config = {"output_dim": self.output_dim,
                      "init": self.init.__name__,
                      "inner_init": self.inner_init.__name__,
                      "forget_bias_init": self.forget_bias_init.__name__,
                      "activation": self.activation.__name__,
                      "inner_activation": self.inner_activation.__name__,
                      "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                      "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                      "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                      "dropout_W": self.dropout_W,
                      "dropout_U": self.dropout_U}
            base_config = super(BiLSTMv1, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


# In[ ]:

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Merge, Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
#from keras.objectives import categorical_crossentropy


def reverse(X):
    return X[::, ::, ::-1]


def output_shape(input_shape):
    # here input_shape includes the samples dimension
    return input_shape  # shap


def sub_mean(X):
    xdms = X.shape
    return X.reshape(xdms[0])


def old_version(ndim=2):

    #middle = 50
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200, 5))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10
    graph.add_node(Convolution1D(nb_filter=5, filter_length=4, input_shape=(None, 5),
                                 border_mode="same"), input='input1', name="conv1")

    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1', name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1', name="input1b")

    # graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    # 66,4

    graph.add_node(LSTM(output_dim=20, activation='sigmoid', input_shape=(200, 10),
                        inner_activation='hard_sigmoid', return_sequences=True),
                   name="allmost", inputs=["input1", "input1b"], concat_axis=-1, merge_mode="concat")
    graph.add_node(Lambda(reverse, output_shape), inputs=["input1", "input1b"], concat_axis=-1, merge_mode="concat",
                   name="reversed0")

    graph.add_node(LSTM(output_dim=20, activation='sigmoid',
                        inner_activation='hard_sigmoid', return_sequences=True),
                   name="allmost1", input="reversed0")

    graph.add_node(Lambda(reverse, output_shape), input="allmost1", name="reversed")

    # Here get the subcategory
    graph.add_node(TimeDistributedDense(7, activation="softmax"), inputs=["allmost", "reversed"],
                   name="output0", merge_mode="concat", concat_axis=-1)

    ##########################################
    # First ehd here
    # graph.add_output(name="output",input="output0")
    #graph.compile('adadelta', {'output':'categorical_crossentropy' })
    ################################################

    # Here get the number of category
    graph.add_node(LSTM(output_dim=27, activation='softmax',
                        inner_activation='hard_sigmoid', return_sequences=False),
                   name="category0", input="output0")

    graph.add_node(Reshape((1, 27)), input="category0", name="category00")

    graph.add_output(name="output", input="output0")
    # graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category", input="category00")

    graph.compile('adadelta', {'output': 'categorical_crossentropy',
                               'category': 'categorical_crossentropy'})

    return graph


# In[2]:

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Merge, Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
#from keras.objectives import categorical_crossentropy


def reverse(X):
    return X[::, ::, ::-1]


def output_shape(input_shape):
    # here input_shape includes the samples dimension
    return input_shape  # shap


def sub_mean(X):
    xdms = X.shape
    return X.reshape(xdms[0])


def old_but_ok(ndim=2):
    #middle = 50
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200, 5))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10

    graph.add_node(Convolution1D(nb_filter=10, filter_length=4, input_shape=(None, 5),
                                 border_mode="same"), input='input1', name="conv1")

    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1', name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1', name="input1b")

    # graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    # 66,4

    # First with 20 of activation

    inside = 50

    graph.add_node(LSTM(output_dim=inside, activation='sigmoid', input_shape=(200, 15),
                        inner_activation='hard_sigmoid', return_sequences=True),
                   name="1allmost", inputs=["input1", "input1b"], concat_axis=-1, merge_mode="concat")

    graph.add_node(Lambda(reverse, output_shape), inputs=["input1", "input1b"], concat_axis=-1, merge_mode="concat",
                   name="reversed0")

    graph.add_node(LSTM(output_dim=inside, activation='sigmoid',
                        inner_activation='hard_sigmoid', return_sequences=True),
                   name="allmost1", input="reversed0")

    graph.add_node(Lambda(reverse, output_shape), input="allmost1", name="reversed")

    # END first

    graph.add_node(LSTM(output_dim=inside, activation='sigmoid', input_shape=(200, 2 * inside + 15),
                        inner_activation='hard_sigmoid', return_sequences=True), name="allmost_l2",
                   inputs=["input1", "input1b", "1allmost", "reversed"], merge_mode="concat", concat_axis=-1)

    graph.add_node(Lambda(reverse, output_shape), inputs=["input1", "input1b", "1allmost", "reversed"], merge_mode="concat",
                   concat_axis=-1,
                   name="reversed0_l2")

    graph.add_node(LSTM(output_dim=inside, activation='sigmoid', input_shape=(200, 2 * inside + 15),
                        inner_activation='hard_sigmoid', return_sequences=True),
                   name="allmost1_l2", input="reversed0_l2")

    graph.add_node(Lambda(reverse, output_shape), input="allmost1_l2", name="reversed_l2")

    # END second

    graph.add_node(Dropout(0.4), inputs=["1allmost", "reversed", "allmost_l2", "reversed_l2"],
                   merge_mode="concat", concat_axis=-1, name="output0_drop")

    # Here get the subcategory

    graph.add_node(TimeDistributedDense(7, activation="softmax"), input="output0_drop",
                   name="output0")

    ##########################################
    # First ehd here
    # graph.add_output(name="output",input="output0")
    #graph.compile('adadelta', {'output':'categorical_crossentropy' })
    ################################################

    # Here get the number of category
    graph.add_node(LSTM(output_dim=12,
                        inner_activation='hard_sigmoid', return_sequences=False),
                   name="category0_r", input="output0")

    graph.add_node(LSTM(output_dim=12,
                        inner_activation='hard_sigmoid', return_sequences=False, go_backwards=True),
                   name="category0_l", input="output0")

    graph.add_node(Dense(12, activation="softmax"), inputs=["category0_l", "category0_r"], concat_axis=1, merge_mode="concat",
                   name="category0")

    graph.add_node(Reshape((1, 12)), input="category0", name="category00")

    # graph.load_weights("step_check")
    #############################################
    # Original end there
    # graph.load_weights("step_check")

    # graph.add_output(name="category",input="category0")
    #graph.compile('adadelta', {'output':'categorical_crossentropy'})

    #############################################

    # graph.add_node(TimeDistributedDense(1,activation="linear"),input='output0',name="output1")

    # graph.load_weights("step_check_bigger")

    graph.add_output(name="output", input="output0")
    # graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category", input="category00")

    graph.compile('adadelta', {'output': perm_loss,
                               'category': 'categorical_crossentropy'})

    graph.load_weights("old_weights/specialist_4_diff_size_50")

    return graph


# In[1]:

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Merge, Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
#from keras.objectives import categorical_crossentropy


def return_two_layer():

    def reverse(X):
        return X[::, ::, ::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])

    #middle = 50
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200, 5))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10

    graph.add_node(Convolution1D(nb_filter=10, filter_length=4, input_shape=(None, 5),
                                 border_mode="same"), input='input1', name="conv1")

    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1', name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1', name="input1b")

    # graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    # 66,4

    # First with 20 of activation

    inside = 50

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh', input_shape=(200, 15),
                          inner_activation='hard_sigmoid', return_sequences=True),
                   name="l1", inputs=["input1", "input1b"], concat_axis=-1, merge_mode="concat")

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh', input_shape=(200, inside + 15),
                          inner_activation='hard_sigmoid', return_sequences=True), name="l2",
                   inputs=["input1", "input1b", "l1"], merge_mode="concat", concat_axis=-1)

    graph.add_node(Dropout(0.4), inputs=["l1", "l2"],
                   merge_mode="concat", concat_axis=-1, name="output0_drop")
    # Here get the subcategory

    graph.add_node(TimeDistributedDense(7, activation="softmax"), input="output0_drop",
                   name="output0")

    ##########################################
    # First ehd here
    # graph.add_output(name="output",input="output0")
    #graph.compile('adadelta', {'output':'categorical_crossentropy' })
    ################################################

    # Here get the number of category
    graph.add_node(BiLSTM(output_dim=12,
                          inner_activation='hard_sigmoid', return_sequences=False),
                   name="category0bi", input="output0")

    graph.add_node(Dense(12, activation="softmax"), input="category0bi", name="category0")

    graph.add_node(Reshape((1, 12)), input="category0", name="category00")

    # graph.load_weights("step_check")
    #############################################
    # Original end there
    # graph.load_weights("step_check")

    # graph.add_output(name="category",input="category0")
    #graph.compile('adadelta', {'output':'categorical_crossentropy'})

    #############################################

    # graph.add_node(TimeDistributedDense(1,activation="linear"),input='output0',name="output1")

    # graph.load_weights("step_check_bigger")

    graph.add_output(name="output", input="output0")
    # graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category", input="category00")

    graph.compile('adadelta', {'output': perm_loss,
                               'category': 'categorical_crossentropy'})

    graph.load_weights("saved_weights/two_bilayer_without_sub")
    return graph
    # graph.load_weights("training_general_scale10")
    #############################################
    # Second end there


#############################################


#history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
# predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
# graph.save_weights("step_check",overwrite=True)


# In[6]:

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Merge, Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
#from keras.objectives import categorical_crossentropy


def return_three_layer():

    def reverse(X):
        return X[::, ::, ::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])

    #middle = 50
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200, 5))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10

    graph.add_node(Convolution1D(nb_filter=10, filter_length=4, input_shape=(None, 5),
                                 border_mode="same"), input='input1', name="conv1")

    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1', name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1', name="input1b")

    # graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    # 66,4

    # First with 20 of activation

    inside = 50

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh', input_shape=(200, 15),
                          inner_activation='hard_sigmoid', return_sequences=True),
                   name="l1", inputs=["input1", "input1b"], concat_axis=-1, merge_mode="concat")

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh', input_shape=(200, inside + 15),
                          inner_activation='hard_sigmoid', return_sequences=True), name="l2",
                   inputs=["input1", "input1b", "l1"], merge_mode="concat", concat_axis=-1)

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh', input_shape=(200, inside + 15),
                          inner_activation='hard_sigmoid', return_sequences=True), name="l3",
                   inputs=["input1", "input1b", "l2"], merge_mode="concat", concat_axis=-1)

    graph.add_node(Dropout(0.4), inputs=["l1", "l2", "l3"],
                   merge_mode="concat", concat_axis=-1, name="output0_drop")
    # Here get the subcategory

    graph.add_node(TimeDistributedDense(10, activation="softmax"), input="output0_drop",
                   name="output0")

    graph.add_node(BiLSTM(output_dim=27,
                          inner_activation='hard_sigmoid', return_sequences=False),
                   name="category0bi", input="output0")

    graph.add_node(Dense(27, activation="softmax"), input="category0bi", name="category0")

    graph.add_node(Reshape((1, 27)), input="category0", name="category00")

    graph.add_output(name="output", input="output0")
    # graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category", input="category0")

    graph.compile('adadelta', {'output': perm_loss,
                               'category': 'categorical_crossentropy'})

    graph.load_weights("three_layer_specialist")
    return graph

# graph.load_weights("training_general_scale10")
#############################################
# Second end there


#############################################


#history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
# predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
# graph.save_weights("step_check",overwrite=True)


# In[2]:

import theano
# print theano.__version__ , theano.__file__
import keras
# print keras.__version__, keras.__file__
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Merge, Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

import theano.tensor as T
import theano
from keras.backend.common import _EPSILON
#from keras.objectives import categorical_crossentropy


def return_three_bis(ndim=2, inside=50):

    # categorical_crossentropy??
    # Loss:

    perm = [[0, 1, 2], [1, 2, 0], [2, 1, 0], [0, 2, 1], [1, 0, 2], [2, 0, 1]]
    perm = [[-3, -2, -1] + iperm for iperm in perm]
    perm = np.array(perm, dtype=np.int)
    perm += 3

    test_true = [[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]

    eps = 1e-7
    test_pred = [[[1 - eps, +eps], [1 - eps, eps], [1 - eps, eps]], [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]],
                 [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]]]

    def perm_loss(y_true, y_pred):
        def loss(m,  y_true, y_pred, perm):

            # return  perm[T.cast(m,"int32")]
            y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            return T.mean(T.sum(y_true[::, ::, perm[m]] * T.log(y_pred), axis=-1), axis=-1)

        #perm = np.array([[0,1],[1,0]],dtype=np.int)
        perm = np.array([[0, 1, 2, 3, 4, 5, 6] + range(7, 10),
                         [0, 1, 2, 4, 5, 3, 6] + range(7, 10),
                         [0, 1, 2, 5, 4, 3, 6] + range(7, 10),
                         [0, 1, 2, 3, 5, 4, 6] + range(7, 10),
                         [0, 1, 2, 4, 3, 5, 6] + range(7, 10),
                         [0, 1, 2, 5, 3, 4, 6] + range(7, 10)], dtype=np.int)

        """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
        seq = T.arange(len(perm))
        result, _ = theano.scan(fn=loss, outputs_info=None,
                                sequences=seq, non_sequences=[y_true, y_pred, perm])
        return -T.mean(T.max(result, axis=0))  # T.max(result.dimshuffle(1,2,0),axis=-1)

    def reverse(X):
        return X[::, ::, ::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])

    #middle = 50
    add = 0
    if ndim == 3:
        add = 1
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200, 5 + add))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10

    graph.add_node(Convolution1D(nb_filter=10, filter_length=4, input_shape=(None, 5 + add),
                                 border_mode="same"), input='input1', name="conv1")

    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1', name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1', name="input1b")

    # graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    # 66,4

    # First with 20 of activation

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh', input_shape=(200, 15 + add),
                          inner_activation='hard_sigmoid', return_sequences=True),
                   name="l1", inputs=["input1", "input1b"], concat_axis=-1, merge_mode="concat")

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh', input_shape=(200, inside + 15 + add),
                          inner_activation='hard_sigmoid', return_sequences=True,), name="l2",
                   inputs=["input1", "input1b", "l1"], merge_mode="concat", concat_axis=-1)

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh', input_shape=(200, inside + 15 + add),
                          inner_activation='hard_sigmoid', return_sequences=True), name="l3",
                   inputs=["input1", "input1b", "l2"], merge_mode="concat", concat_axis=-1)

    graph.add_node(Dropout(0.4), inputs=["l1", "l2", "l3"],
                   merge_mode="concat", concat_axis=-1, name="output0_drop")
    # Here get the subcategory

    graph.add_node(TimeDistributedDense(10, activation="softmax"), input="output0_drop",
                   name="output0")

    graph.add_node(TimeDistributedDense(4, activation="softmax"), input="output0",
                   name="output0b")

    graph.add_node(BiLSTM(output_dim=27,
                          inner_activation='hard_sigmoid', return_sequences=False),
                   name="category0bi", input="output0")

    graph.add_node(Dense(27, activation="softmax"), input="category0bi", name="category0")

    graph.add_node(Reshape((1, 27)), input="category0", name="category00")

    graph.add_output(name="output", input="output0")
    graph.add_output(name="outputtype", input="output0b")

    # graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category", input="category0")

    graph.compile('adadelta', {'output': perm_loss,
                               'category': 'categorical_crossentropy',
                               'outputtype': 'categorical_crossentropy'})

    # graph.load_weights("training_general_scale10")
    #############################################
    # Second end there

    if ndim == 2 and inside == 50:
        graph.load_weights("saved_weights/three_bilayer_sub_bis")

    if ndim == 3 and inside == 50:
        graph.load_weights("saved_weights/three_bilayer_sub_bis_3D_isotrope")

    #############################################

    return graph

    #history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
    # predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
    # graph.save_weights("step_check",overwrite=True)


# In[9]:

def return_three_bis_three_level(ndim=2, inside=50):

    # categorical_crossentropy??
    # Loss:

    perm = [[0, 1, 2], [1, 2, 0], [2, 1, 0], [0, 2, 1], [1, 0, 2], [2, 0, 1]]
    perm = [[-3, -2, -1] + iperm for iperm in perm]
    perm = np.array(perm, dtype=np.int)
    perm += 3

    test_true = [[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]

    eps = 1e-7
    test_pred = [[[1 - eps, +eps], [1 - eps, eps], [1 - eps, eps]], [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]],
                 [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]]]

    def perm_loss(y_true, y_pred):
        def loss(m,  y_true, y_pred, perm):

            # return  perm[T.cast(m,"int32")]
            y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            return T.mean(T.sum(y_true[::, ::, perm[m]] * T.log(y_pred), axis=-1), axis=-1)

        #perm = np.array([[0,1],[1,0]],dtype=np.int)
        perm = np.array([[0, 1, 2, 3, 4, 5, 6] + range(7, 10),
                         [0, 1, 2, 4, 5, 3, 6] + range(7, 10),
                         [0, 1, 2, 5, 4, 3, 6] + range(7, 10),
                         [0, 1, 2, 3, 5, 4, 6] + range(7, 10),
                         [0, 1, 2, 4, 3, 5, 6] + range(7, 10),
                         [0, 1, 2, 5, 3, 4, 6] + range(7, 10)], dtype=np.int)

        """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
        seq = T.arange(len(perm))
        result, _ = theano.scan(fn=loss, outputs_info=None,
                                sequences=seq, non_sequences=[y_true, y_pred, perm])
        return -T.mean(T.max(result, axis=0))  # T.max(result.dimshuffle(1,2,0),axis=-1)

    def reverse(X):
        return X[::, ::, ::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])

    #middle = 50
    add = 0
    if ndim == 3:
        add = 1
    graph = Graph()
    graph.add_input(name='input1', input_shape=(200, 3 * (5 + add)))
    #graph.add_input(name='input2', input_shape=(None,2))

    #nbr_filter = 10

    graph.add_node(Convolution1D(nb_filter=10, filter_length=4, input_shape=(None, 3 * (5 + add)),
                                 border_mode="same"), input='input1', name="conv1")

    graph.add_node(MaxPooling1D(pool_length=2),
                   input='conv1', name="max1")

    graph.add_node(UpSampling1D(length=2),
                   input='max1', name="input1b")

    # graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    # 66,4

    # First with 20 of activation

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh', input_shape=(200, 10 + 3 * (5 + add)),
                          inner_activation='hard_sigmoid', return_sequences=True),
                   name="l1", inputs=["input1", "input1b"], concat_axis=-1, merge_mode="concat")

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh', input_shape=(200, inside + 10 + 3 * (5 + add)),
                          inner_activation='hard_sigmoid', return_sequences=True,), name="l2",
                   inputs=["input1", "input1b", "l1"], merge_mode="concat", concat_axis=-1)

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh', input_shape=(200, inside + 10 + 3 * (5 + add)),
                          inner_activation='hard_sigmoid', return_sequences=True), name="l3",
                   inputs=["input1", "input1b", "l2"], merge_mode="concat", concat_axis=-1)

    graph.add_node(Dropout(0.4), inputs=["l1", "l2", "l3"],
                   merge_mode="concat", concat_axis=-1, name="output0_drop")
    # Here get the subcategory

    graph.add_node(TimeDistributedDense(10, activation="softmax"), input="output0_drop",
                   name="output0")

    graph.add_node(TimeDistributedDense(4, activation="softmax"), input="output0",
                   name="output0b")

    graph.add_node(BiLSTM(output_dim=27,
                          inner_activation='hard_sigmoid', return_sequences=False),
                   name="category0bi", input="output0")

    graph.add_node(Dense(27, activation="softmax"), input="category0bi", name="category0")

    graph.add_node(Reshape((1, 27)), input="category0", name="category00")

    graph.add_output(name="output", input="output0")
    graph.add_output(name="outputtype", input="output0b")

    # graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category", input="category0")

    graph.compile('adadelta', {'output': perm_loss,
                               'category': 'categorical_crossentropy',
                               'outputtype': 'categorical_crossentropy'})

    # graph.load_weights("training_general_scale10")
    #############################################
    # Second end there

    #############################################

    return graph


# In[12]:

import theano
# print theano.__version__ , theano.__file__
import keras
# print keras.__version__, keras.__file__
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Merge, Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU


import theano.tensor as T
import theano
from keras.backend.common import _EPSILON
#from keras.objectives import categorical_crossentropy


def return_three_bis_simpler(ndim=2, permute=True, extend=0):

    # categorical_crossentropy??
    # Loss:

    perm = [[0, 1, 2], [1, 2, 0], [2, 1, 0], [0, 2, 1], [1, 0, 2], [2, 0, 1]]
    perm = [[-3, -2, -1] + iperm for iperm in perm]
    perm = np.array(perm, dtype=np.int)
    perm += 3

    test_true = [[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]

    eps = 1e-7
    test_pred = [[[1 - eps, +eps], [1 - eps, eps], [1 - eps, eps]], [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]],
                 [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]]]

    def perm_loss(y_true, y_pred):
        def loss(m,  y_true, y_pred, perm):

            # return  perm[T.cast(m,"int32")]
            y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            return T.mean(T.sum(y_true[::, ::, perm[m]] * T.log(y_pred), axis=-1), axis=-1)

        #perm = np.array([[0,1],[1,0]],dtype=np.int)
        perm = np.array([[0, 1, 2, 3, 4, 5, 6] + range(7, 10),
                         [0, 1, 2, 4, 5, 3, 6] + range(7, 10),
                         [0, 1, 2, 5, 4, 3, 6] + range(7, 10),
                         [0, 1, 2, 3, 5, 4, 6] + range(7, 10),
                         [0, 1, 2, 4, 3, 5, 6] + range(7, 10),
                         [0, 1, 2, 5, 3, 4, 6] + range(7, 10)], dtype=np.int)

        """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
        seq = T.arange(len(perm))
        result, _ = theano.scan(fn=loss, outputs_info=None,
                                sequences=seq, non_sequences=[y_true, y_pred, perm])
        return -T.mean(T.max(result, axis=0))  # T.max(result.dimshuffle(1,2,0),axis=-1)

    def reverse(X):
        return X[::, ::, ::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])

    #middle = 50
    add = 0
    if ndim == 3:
        add = 1

    graph = Graph()
    graph.add_input(name='input1', input_shape=(None, 5 + add))

    # graph.add_node(Convolution1D(nb_filter=4,filter_length=3,input_shape=(None,2),
    #                             border_mode="same"),input='input1',name="output0")

    # 66,4

    # First with 20 of activation

    inside = 50

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',
                          inner_activation='hard_sigmoid', return_sequences=True),
                   name="l1", input="input1")

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',
                          inner_activation='hard_sigmoid', return_sequences=True,), name="l2",
                   inputs=["input1", "l1"], merge_mode="concat", concat_axis=-1)

    graph.add_node(BiLSTM(output_dim=inside, activation='tanh',
                          inner_activation='hard_sigmoid', return_sequences=True), name="l3",
                   inputs=["input1", "l2"], merge_mode="concat", concat_axis=-1)

    graph.add_node(Dropout(0.4), inputs=["l1", "l2", "l3"],
                   merge_mode="concat", concat_axis=-1, name="output0_drop")
    # Here get the subcategory

    graph.add_node(TimeDistributedDense(10 + extend, activation="softmax"), input="output0_drop",
                   name="output0")

    if permute:
        graph.add_node(TimeDistributedDense(4, activation="softmax"), input="output0",
                       name="output0b")

    graph.add_node(BiLSTM(output_dim=27,
                          inner_activation='hard_sigmoid', return_sequences=False),
                   name="category0bi", input="output0")

    graph.add_node(Dense(27, activation="softmax"), input="category0bi", name="category0")

    graph.add_node(Reshape((1, 27)), input="category0", name="category00")

    graph.add_output(name="output", input="output0")
    if permute:
        graph.add_output(name="outputtype", input="output0b")

    # graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category", input="category0")

    if permute:
        graph.compile('adadelta', {'output': perm_loss,
                                   'category': 'categorical_crossentropy',
                                   'outputtype': 'categorical_crossentropy'})
    else:
        graph.compile('adadelta', {'output': "categorical_crossentropy",
                                   'category': 'categorical_crossentropy'})

    # graph.load_weights("training_general_scale10")
    #############################################
    # Second end there

    #############################################

    return graph

    #history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
    # predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
    # graph.save_weights("step_check",overwrite=True)


# In[9]:

import theano
# print theano.__version__ , theano.__file__
import keras
# print keras.__version__, keras.__file__
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Merge, Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

import theano.tensor as T
import theano
from keras.backend.common import _EPSILON
#from keras.objectives import categorical_crossentropy


def return_three_paper(ndim=2, inside=50, permutation=True, inputsize=5, simple=False):

    # categorical_crossentropy??
    # Loss:

    perm = [[0, 1, 2], [1, 2, 0], [2, 1, 0], [0, 2, 1], [1, 0, 2], [2, 0, 1]]
    perm = [[-3, -2, -1] + iperm for iperm in perm]
    perm = np.array(perm, dtype=np.int)
    perm += 3

    test_true = [[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]

    eps = 1e-7
    test_pred = [[[1 - eps, +eps], [1 - eps, eps], [1 - eps, eps]], [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]],
                 [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]]]

    def perm_loss(y_true, y_pred):
        def loss(m,  y_true, y_pred, perm):

            # return  perm[T.cast(m,"int32")]
            y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            return T.mean(T.sum(y_true[::, ::, perm[m]] * T.log(y_pred), axis=-1), axis=-1)

        #perm = np.array([[0,1],[1,0]],dtype=np.int)
        perm = np.array([[0, 1, 2, 3, 4, 5, 6] + range(7, 10),
                         [0, 1, 2, 4, 5, 3, 6] + range(7, 10),
                         [0, 1, 2, 5, 4, 3, 6] + range(7, 10),
                         [0, 1, 2, 3, 5, 4, 6] + range(7, 10),
                         [0, 1, 2, 4, 3, 5, 6] + range(7, 10),
                         [0, 1, 2, 5, 3, 4, 6] + range(7, 10)], dtype=np.int)

        """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
        seq = T.arange(len(perm))
        result, _ = theano.scan(fn=loss, outputs_info=None,
                                sequences=seq, non_sequences=[y_true, y_pred, perm])
        return -T.mean(T.max(result, axis=0))  # T.max(result.dimshuffle(1,2,0),axis=-1)

    def reverse(X):
        return X[::, ::, ::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])

    #middle = 50
    add = 0

    if ndim == 3:
        add = 1

    if simple:
        Bi = BiSimpleRNN

    else:
        Bi = BiLSTM
    graph = Graph()
    graph.add_input(name='input1', input_shape=(None, inputsize))

    graph.add_node(Bi(output_dim=inside, activation='tanh', return_sequences=True, close=True, input_shape=(200, inputsize),),
                   name="l1", input="input1")

    graph.add_node(Bi(output_dim=inside, input_shape=(200, inputsize),
                      return_sequences=True, close=True, activation='tanh'), name="l2",
                   inputs=["input1", "l1"], merge_mode="concat", concat_axis=-1)

    graph.add_node(Bi(output_dim=inside, activation='tanh', input_shape=(200, inputsize),
                      return_sequences=True, close=True), name="l3",
                   inputs=["input1", "l2"], merge_mode="concat", concat_axis=-1)

    graph.add_node(Dropout(0.4), inputs=["l1", "l2", "l3"],
                   merge_mode="concat", concat_axis=-1, name="output0_drop")
    # Here get the subcategory

    graph.add_node(TimeDistributedDense(10, activation="softmax"), input="output0_drop",
                   name="output0")

    graph.add_node(Bi(output_dim=27, activation='tanh', return_sequences=False, close=True),
                   name="category0bi", input="output0")

    graph.add_node(Dense(27, activation="softmax"), input="category0bi", name="category0")

    graph.add_output(name="output", input="output0")

    # graph.add_output(name="rOutput",input="output1")
    graph.add_output(name="category", input="category0")

    if permutation:
        graph.compile('adadelta', {'output': perm_loss,
                                   'category': 'categorical_crossentropy'})
    else:
        graph.compile('adadelta', {'output': 'categorical_crossentropy',
                                   'category': 'categorical_crossentropy'})

    # graph.load_weights("training_general_scale10")
    #############################################
    # Second end there

    #############################################

    return graph

    #history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
    # predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
    # graph.save_weights("step_check",overwrite=True)


# In[9]:

import theano
# print theano.__version__ , theano.__file__
import keras
# print keras.__version__, keras.__file__
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Merge, Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
"""
if  int(keras.__version__.split(".")[0]) >= 1.0 :
    from Bilayer import BiLSTMv1 as BiLSTM
    from Bilayer import BiSimpleRNNv1 as BiSimpleRNN

else:
    from Bilayer import BiLSTM, BiSimpleRNN
    """
import theano.tensor as T
import theano
from keras.backend.common import _EPSILON
#from keras.objectives import categorical_crossentropy


def return_layer_paper(ndim=2, inside=50, permutation=True, inputsize=5, simple=False,
                       n_layers=4, category=True, output=True):

    # categorical_crossentropy??
    # Loss:

    perm = [[0, 1, 2], [1, 2, 0], [2, 1, 0], [0, 2, 1], [1, 0, 2], [2, 0, 1]]
    perm = [[-3, -2, -1] + iperm for iperm in perm]
    perm = np.array(perm, dtype=np.int)
    perm += 3

    test_true = [[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]

    eps = 1e-7
    test_pred = [[[1 - eps, +eps], [1 - eps, eps], [1 - eps, eps]], [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]],
                 [[eps, 1 - eps], [eps, 1 - eps], [eps, 1 - eps]]]

    def perm_loss(y_true, y_pred):
        def loss(m,  y_true, y_pred, perm):

            # return  perm[T.cast(m,"int32")]
            y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            return T.mean(T.sum(y_true[::, ::, perm[m]] * T.log(y_pred), axis=-1), axis=-1)

        #perm = np.array([[0,1],[1,0]],dtype=np.int)
        perm = np.array([[0, 1, 2, 3, 4, 5, 6] + range(7, 10),
                         [0, 1, 2, 4, 5, 3, 6] + range(7, 10),
                         [0, 1, 2, 5, 4, 3, 6] + range(7, 10),
                         [0, 1, 2, 3, 5, 4, 6] + range(7, 10),
                         [0, 1, 2, 4, 3, 5, 6] + range(7, 10),
                         [0, 1, 2, 5, 3, 4, 6] + range(7, 10)], dtype=np.int)

        """perm = np.array([[0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6]],dtype=np.int)"""
        seq = T.arange(len(perm))
        result, _ = theano.scan(fn=loss, outputs_info=None,
                                sequences=seq, non_sequences=[y_true, y_pred, perm])
        return -T.mean(T.max(result, axis=0))  # T.max(result.dimshuffle(1,2,0),axis=-1)

    def reverse(X):
        return X[::, ::, ::-1]

    def output_shape(input_shape):
        # here input_shape includes the samples dimension
        return input_shape  # shap

    def identity(X):
        return X

    def sub_mean(X):
        xdms = X.shape
        return X.reshape(xdms[0])

    #middle = 50
    add = 0

    if ndim == 3:
        add = 1

    if simple:
        Bi = BiSimpleRNN

    else:
        Bi = BiLSTMv1
    graph = Graph()
    graph.add_input(name='input1', input_shape=(None, inputsize))

    graph.add_node(Bi(output_dim=inside, activation='tanh', return_sequences=True, close=True, input_shape=(200, inputsize),),
                   name="l1", input="input1")

    for layer in range(2, n_layers + 1):

        graph.add_node(Bi(output_dim=inside, input_shape=(200, inputsize),
                          return_sequences=True, close=True, activation='tanh'), name="l%i" % layer,
                       inputs=["input1", "l%i" % (layer - 1)], merge_mode="concat", concat_axis=-1)

    graph.add_node(Dropout(0.4), inputs=["l%i" % layer for layer in range(1, n_layers + 1)],
                   merge_mode="concat", concat_axis=-1, name="output0_drop")
    # Here get the subcategory

    graph.add_node(TimeDistributedDense(10, activation="softmax"), input="output0_drop",
                   name="output0")

    res = {}

    if category:
        graph.add_node(Bi(output_dim=27, activation='tanh', return_sequences=False, close=True),
                       name="category0bi", input="output0")
        graph.add_node(Dense(27, activation="softmax"), input="category0bi", name="category0")
        graph.add_output(name="category", input="category0")
        res['category'] = 'categorical_crossentropy'

    if output:
        graph.add_output(name="output", input="output0")

        if permutation:
            res['output'] = perm_loss
        else:
            res['output'] = 'categorical_crossentropy'

    graph.compile('adadelta', res)

    return graph

    #history = graph.fit({'input1':X_train[::,1], 'input2':X2_train[::0], 'output':y_train}, nb_epoch=10)
    # predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}
    # graph.save_weights("step_check",overwrite=True)


# In[10]:

#graph = return_layer_paper(n_layers=3)


# In[12]:

# graph.summary()


# In[ ]:
