import os
import random, csv
from statistics import mean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers, Sequential
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
import math
import h5py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

THEANO_FLAGS = device = 'gpu0'
floatX = 'float32'

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score
from tensorflow.keras import backend as K
from tensorflow.keras import Model
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Masking, BatchNormalization, Flatten, \
    Softmax, Bidirectional, Activation, Permute,LayerNormalization,Conv1D,MaxPooling1D
import pickle
import matplotlib.pyplot as plt

path_ = os.path.dirname(__file__)
# canshu
batchsz = 64
max_review_len = 200
units = 64
epochs = 300

with open(os.path.join(path_, 'mysavedata/ATCG/Train_data.pkl'), 'rb+') as f2:
    seqs_train1 = pickle.load(f2)
with open(os.path.join(path_, 'mysavedata/ATCG/Train_data2.pkl'), 'rb+') as f2:
    seqs_train2 = pickle.load(f2)
seqs_train = np.hstack((seqs_train1, seqs_train2))

with open(os.path.join(path_, 'mysavedata/ATCG/Train_score.pkl'), 'rb+') as f:
    scores_train1 = pickle.load(f)
scores_train1 = [np.float32(x) for x in scores_train1]
scores_train1 = np.array(scores_train1)
with open(os.path.join(path_, 'mysavedata/ATCG/Train_score2.pkl'), 'rb+') as f:
    scores_train2 = pickle.load(f)
scores_train2 = [np.float32(x) for x in scores_train2]
scores_train2 = np.array(scores_train2)
scores_train = np.hstack((scores_train1, scores_train2))

with open(os.path.join(path_, 'mysavedata/ATCGtest/Test_data.pkl'), 'rb+') as f4:
    seqs_test1 = pickle.load(f4)
with open(os.path.join(path_, 'mysavedata/ATCGtest/Test_data2.pkl'), 'rb+') as f4:
    seqs_test2 = pickle.load(f4)
seqs_test = np.hstack((seqs_test1, seqs_test2))

with open(os.path.join(path_, 'mysavedata/ATCGtest/Test_score.pkl'), 'rb+') as f3:
    scores_test1 = pickle.load(f3)
scores_test1 = [np.float32(x) for x in scores_test1]
scores_test1 = np.array(scores_test1)
with open(os.path.join(path_, 'mysavedata/ATCGtest/Test_score2.pkl'), 'rb+') as f3:
    scores_test2 = pickle.load(f3)
scores_test2 = [np.float32(x) for x in scores_test2]
scores_test2 = np.array(scores_test2)
scores_test = np.hstack((scores_test1, scores_test2))


with open(os.path.join(path_, 'mysavedata/ATCGtest/Test_seq.pkl'), 'rb+') as f5:
    testseq1 = pickle.load(f5)
with open(os.path.join(path_, 'mysavedata/ATCGtest/Test_seq2.pkl'), 'rb+') as f5:
    testseq2 = pickle.load(f5)
testseq = np.hstack((testseq1, testseq2))

with open(os.path.join('./result/CNN_attention/', 'testseq_lenet5.csv'), 'w', newline='') as f4:  # 实验结果保存的地方
    writer = csv.writer(f4)
    for z in testseq:
        writer.writerow([z])
#
with open(os.path.join('./result/CNN_attention/', 'truth_lenet5.csv'), 'w', newline='') as f6:  # 实验结果保存的地方
    writer = csv.writer(f6)
    for q in scores_test:
        writer.writerow([q])

seqs_train = keras.preprocessing.sequence.pad_sequences(seqs_train, maxlen=max_review_len, dtype='float32',
                                                        padding='post')  # 对训练的数据做预处理、# post:在后填充; pre:在前填充
# seqs_val = keras.preprocessing.sequence.pad_sequences(seqs_val, maxlen=max_review_len, dtype='float32', padding='post')
seqs_test = keras.preprocessing.sequence.pad_sequences(seqs_test, maxlen=max_review_len, dtype='float32',
                                                       padding='post')
seqs_train = tf.convert_to_tensor(seqs_train, dtype=tf.float32)
scores_train = tf.convert_to_tensor(scores_train, dtype=tf.float32)

seqs_test = tf.convert_to_tensor(seqs_test, dtype=tf.float32)
# # 构建数据集，打散，批量，并丢掉最后一个不够batchsz的batch
db_train = tf.data.Dataset.from_tensor_slices((seqs_train, scores_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((seqs_test))
db_test = db_test.batch(batchsz, drop_remainder=True)

def attention(inputs,d_model):
    Q = Dense(d_model)(inputs)
    K1 = Dense(d_model)(inputs)
    V = Dense(d_model)(inputs)
    d_k = d_model
    K_T = Permute((2, 1))(K1)
    scores = tf.matmul(Q, K_T / math.sqrt(d_k))
    alp = tf.nn.softmax(scores)
    context = tf.matmul(alp, V)
    output1 = K.sum(context, axis=1)
    return output1, alp


inputs = Input(shape=(200, 4))
inputs1 = Masking(0, input_shape=(200, 4))(inputs)
# lstm_out = Bidirectional(LSTM(units, use_bias=True, recurrent_activation='sigmoid', kernel_initializer='glorot_uniform',
#                               return_sequences=True), name='bilstm')(inputs1)
conv1 = Conv1D(units, 3, strides=1, padding='valid')(inputs1)
conv1 = tf.nn.relu(conv1)
output1,conv1 = attention(conv1,units)
conv1 = MaxPooling1D(2)(conv1)
conv2 = Conv1D(units, 5, strides=1, padding='valid')(conv1)
conv2 = tf.nn.relu(conv2)
output2,conv2 = attention(conv2,units)
conv2 = MaxPooling1D(2)(conv2)
flatten1 = Flatten()(conv2)
output = Dense(units)(flatten1)
output = tf.nn.relu(output)
output = Dense(1)(output)
model1 = tf.keras.Model(inputs=inputs, outputs=output)


def r2(y_true, y_pred):
    rsquare = 1 - K.sum((y_true - y_pred) ** 2) / K.sum((y_true - K.mean(y_true)) ** 2)
    return rsquare


def get_huber_loss_fn(**huber_loss_kwargs):
    def custom_huber_loss(y_true, y_pred):
        return tf.losses.huber(y_true, y_pred, **huber_loss_kwargs)

    return custom_huber_loss

def loss_(weight):
    def loss_function(y_true, y_pred):
        #  weight= y_true+y_pred
        # weight_loss = tf.abs(y_true)
        y_true = tf.cast(y_true, dtype=K.floatx())
        abs_true = tf.abs(y_true)
        r2 = 1 - (1 - K.sum((y_true - y_pred) ** 2) / K.sum((y_true - K.mean(y_true)) ** 2))
        huber_loss = tf.losses.huber(y_true, y_pred)*abs_true
        #  mae = tf.losses.mae(y_true,y_pred)
        mse = tf.losses.mse(y_true, y_pred)
        # huber = tf.losses.
        mss_loss = mse * abs_true
        loss = weight * r2 + (1 - weight) * huber_loss
        return loss

    return loss_function


qall = []
qaall = []
r2all = []
# for i in range(n_epoch):
# checkpoint_filepath = './models/model_lenet5.h5'
checkpoint_filepath = './model/cnn_attention/model_epoch.298_valloss.8.3272_valR2.0.8079.hdf5'
checkpoint_filepath_many = os.path.join('./model/cnn_attention/',
                                   'model_epoch.{epoch:03d}_valloss.{val_loss:.4f}_valR2.{val_r2:.4f}.hdf5')
# checkpoint_filepath = './models_2/model_epoch.285_valloss.4.6585_valR2.0.9148.hdf5'
# checkpoint_filepath_many = os.path.join('./models_2/',
#                                    'model_epoch.{epoch:03d}_valloss.{val_loss:.4f}_valR2.{val_r2:.4f}.hdf5')
if os.path.exists(checkpoint_filepath):
    print("------------load Model!-----------")
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath_many,
    #     save_weights_only=False,
    #     save_best_only=True
    # )
    model1 = load_model(checkpoint_filepath, custom_objects={'loss_function': loss_(0.3), 'r2': r2})
    # model1.compile(optimizer=optimizers.Adam(learning_rate=0.001),
    #                #  loss=get_huber_loss_fn(delta=0.1),
    #                loss=loss_(0.3),
    #                metrics=[r2]
    #                )
    # q = model1.fit(seqs_train, scores_train, epochs=epochs, callbacks=[model_checkpoint_callback], verbose=2,
    #                batch_size=batchsz, validation_split=0.15)
    cl = model1.predict(seqs_test, batch_size=1)
    R_2 = r2_score(scores_test[:len(cl)], cl)
    print("R_2：")
    print(R_2)

else:
    print("------------Train Model!-----------")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_many,
        save_weights_only=False,
        save_best_only=True
    )

    model1.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                   #  loss=get_huber_loss_fn(delta=0.1),
                   loss=loss_(0.3),
                   metrics=[r2]
                   )  # Adam
    q = model1.fit(seqs_train, scores_train, epochs=epochs, callbacks=[model_checkpoint_callback], verbose=2,
                   batch_size=batchsz, validation_split=0.15)
    # cl = model1.predict(seqs_test, batch_size=1)
    # qall = q.history['loss']
    # qaall = q.history['val_loss']
    # model1.save('./my_model_lenet5.h5')

    # num = 1
    # epochs2 = range(1, len(qaall) + 1)
    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # ax.plot(epochs2, q.history['loss'], 'b', label='Training loss')
    # ax.plot(epochs2, q.history['val_loss'], 'r', label='Validation val_loss')
    # ax2.plot(epochs2, q.history['val_r2'], 'g', label='r2')

    # ax.set_xlabel('epochs')
    # ax.legend()
    # ax.set_ylabel('loss')
    # ax.legend()
    # ax2.set_ylabel('val_r2')
    # ax2.legend()
    # # plt.plot(epochs,r2,'g',label='r2_score')
    # plt.show()
    # plt.savefig('./model_loss_lenet5.jpg')

    # R_2 = r2_score(scores_test[:len(cl)], cl)
    # print("R_2：")
    # print(R_2)

with open(os.path.join('./result/CNN_attention/', 'pred_score_lenet5.csv'), 'w', newline='') as f5:
    writer = csv.writer(f5)
    for a in cl:
        writer.writerow(a)
print(cl)