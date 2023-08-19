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
# from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Masking, BatchNormalization, Flatten, \
    Softmax, Bidirectional, Activation, Permute, Conv1D, MaxPooling1D,AveragePooling1D
import pickle
import matplotlib.pyplot as plt

path_ = os.path.dirname(__file__)
# canshu
batchsz = 64
max_review_len = 200
units = 64
epochs = 300
from tensorflow.keras.models import Sequential

with open(os.path.join('../', 'traindata/only_264706_traindata.pkl'), 'rb+') as f2:
    seqs_train = pickle.load(f2)

with open(os.path.join('../', 'traindata/only_264706_trainscore.pkl'), 'rb+') as f:
    scores_train1 = pickle.load(f)
scores_train1 = [np.float32(x) for x in scores_train1]
scores_train = np.array(scores_train1)

with open(os.path.join('../', 'allsoftware/new_allsoftware_testdata.pkl'), 'rb+') as f4:
    seqs_test = pickle.load(f4)


with open(os.path.join('../', 'allsoftware/new_allsoftware_testscore.pkl'), 'rb+') as f3:
    scores_test1 = pickle.load(f3)
scores_test1 = [np.float32(x) for x in scores_test1]
scores_test = np.array(scores_test1)


with open(os.path.join('../', 'allsoftware/new_allsoftware_testseq.pkl'), 'rb+') as f5:
    testseq = pickle.load(f5)


with open(os.path.join('./', 'testseq__resnet2.csv'), 'w', newline='') as f4:  # 实验结果保存的地方
    writer = csv.writer(f4)
    for z in testseq:
        writer.writerow([z])
#
with open(os.path.join('./', 'truth__resnet2.csv'), 'w', newline='') as f6:  # 实验结果保存的地方
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
# seqs_val = tf.convert_to_tensor(seqs_val, dtype=tf.float32)
# scores_val = tf.convert_to_tensor(scores_val, dtype=tf.float32)
seqs_test = tf.convert_to_tensor(seqs_test, dtype=tf.float32)
# # 构建数据集，打散，批量，并丢掉最后一个不够batchsz的batch
db_train = tf.data.Dataset.from_tensor_slices((seqs_train, scores_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
# db_val = tf.data.Dataset.from_tensor_slices((seqs_val, scores_val))
# db_val = db_val.batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((seqs_test))
db_test = db_test.batch(batchsz, drop_remainder=True)


def attention(inputs):
    Q = Dense(units * 2)(inputs)
    # Q = tf.nn.leaky_relu(Q)
    K1 = Dense(units * 2)(inputs)
    V = Dense(units * 2)(inputs)
    # d_model = K1.size(-1)
    d_k = units * 2
    K_T = Permute((2, 1))(K1)
    scores = tf.matmul(Q, K_T / math.sqrt(d_k))
    alp = tf.nn.softmax(scores)
    context = tf.matmul(alp, V)
    output1 = K.sum(context, axis=1)
    return output1, alp


inputs = Input(shape=(200, 4))
inputs1 = Masking(0, input_shape=(200, 4))(inputs)
conv1 = Conv1D(64, 7, strides=2, padding='valid')(inputs1)
pool1 = MaxPooling1D(3, strides=2, padding='same')(conv1)
conv2 = Conv1D(64, 3, strides=1, padding='same')(pool1)
conv2 = tf.nn.relu(BatchNormalization()(conv2))
conv2 = Conv1D(64, 3, strides=1, padding='same')(conv2)
connect1 = tf.add(pool1,conv2)
conv3 = Conv1D(64, 3, strides=1, padding='same')(connect1)
conv3 = tf.nn.relu(BatchNormalization()(conv3))
conv3 = Conv1D(64, 3, strides=1, padding='same')(conv3)
connect2 = tf.add(connect1, conv3)
conv4 = Conv1D(128, 3, strides=2, padding='same')(connect2)
conv4 = tf.nn.relu(BatchNormalization()(conv4))
conv4 = Conv1D(128, 3, strides=1, padding='same')(conv4)
connect3 = Conv1D(128, 1, strides=2, padding='same')(connect2)
connect3 = tf.nn.relu(tf.add(conv4, connect3))
conv5 = Conv1D(128, 3, strides=1, padding='same')(connect3)
conv5 = tf.nn.relu(BatchNormalization()(conv5))
conv5 = Conv1D(128, 3, strides=1, padding='same')(conv5)
connect4 = tf.add(connect3, conv5)
conv6 = Conv1D(256, 3, strides=2, padding='same')(connect4)
conv6 = tf.nn.relu(BatchNormalization()(conv6))
conv6 = Conv1D(256, 3, strides=1, padding='same')(conv6)
connect5 = Conv1D(256, 1, strides=2, padding='same')(connect4)
connect5 = tf.nn.relu(tf.add(conv6, connect5))
conv7 = Conv1D(256, 3, strides=1, padding='same')(connect5)
conv7 = tf.nn.relu(BatchNormalization()(conv7))
conv7 = Conv1D(256, 3, strides=1, padding='same')(conv7)
connect6 = tf.add(connect5, conv7)
conv8 = Conv1D(512, 3, strides=2, padding='same')(connect6)
conv8 = tf.nn.relu(BatchNormalization()(conv8))
conv8 = Conv1D(512, 3, strides=1, padding='same')(conv8)
connect7 = Conv1D(512, 1, strides=2, padding='same')(connect6)
connect7 = tf.nn.relu(tf.add(conv8, connect7))
conv9 = Conv1D(512, 3, strides=1, padding='same')(connect7)
conv9 = tf.nn.relu(BatchNormalization()(conv9))
conv9 = Conv1D(512, 3, strides=1, padding='same')(conv9)
connect8 = tf.add(connect7, conv9)
pool2 = AveragePooling1D(7)(connect8)

output = Dense(1)(pool2)
output = K.squeeze(output,axis=2)
model1 = tf.keras.Model(inputs=inputs, outputs=output)


# n_epoch=10
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
# checkpoint_filepath = './models/model_resnet2.h5'
checkpoint_filepath = './model_epoch.023_valloss.9.3704_valR2.0.7710.hdf5'
checkpoint_filepath_many = os.path.join('./model_Resnet18/',
                                   'model_epoch.{epoch:03d}_valloss.{val_loss:.4f}_valR2.{val_r2:.4f}.hdf5')
if os.path.exists(checkpoint_filepath):
    print("------------load Model!-----------")
    # model1=load_model(checkpoint_filepath)
    # model1 = mylstm(units)
    # model1.call(inputs=seqs_train)
    model1 = load_model(checkpoint_filepath, custom_objects={'loss_function': loss_(0.3), 'r2': r2})
    cl = model1.predict(seqs_test, batch_size=1)
    R_2 = r2_score(scores_test[:len(cl)], cl)
    print("R_2：")
    print(R_2)
# print(cl)
# qall = q.history['loss']
# qaall = q.history['val_loss']
else:
    print("------------Train Model!-----------")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_many,
        save_weights_only=False,
        save_best_only=True
    )
    # earstop=tf.keras.callbacks.EarlyStopping(monitor='val_r2',mode='max',patience=20)
    #  model1 = mylstm(units)
    # reduce_lr=tf.keras.callbacks.LearningRateScheduler(scheduler)
    model1.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                   #  loss=get_huber_loss_fn(delta=0.1),
                   loss=loss_(0.3),
                   metrics=[r2]
                   )  # Adam
    q = model1.fit(seqs_train, scores_train, epochs=epochs, callbacks=[model_checkpoint_callback], verbose=2,
                   batch_size=batchsz, validation_split=0.15)
    cl = model1.predict(seqs_test, batch_size=1)
    qall = q.history['loss']
    qaall = q.history['val_loss']
    model1.save('./my_model_resnet2.h5')

    num = 1
    epochs2 = range(1, len(qaall) + 1)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(epochs2, q.history['loss'], 'b', label='Training loss')
    ax.plot(epochs2, q.history['val_loss'], 'r', label='Validation val_loss')
    ax2.plot(epochs2, q.history['val_r2'], 'g', label='r2')

    ax.set_xlabel('epochs')
    ax.legend()
    ax.set_ylabel('loss')
    ax.legend()
    ax2.set_ylabel('val_r2')
    ax2.legend()
    # plt.plot(epochs,r2,'g',label='r2_score')
    plt.show()
    plt.savefig('./model_loss_resnet2.jpg')

    R_2 = r2_score(scores_test[:len(cl)], cl)
    print("R_2：")
    print(R_2)

with open(os.path.join('./', 'pred_score__resnet2.csv'), 'w', newline='') as f5:
    writer = csv.writer(f5)
    for a in cl:
        writer.writerow(a)
print(cl)