import tensorflow as tf
from tensorflow.keras import backend as K
import spektral
import numpy as np
import math
from sklearn.feature_selection import mutual_info_regression


class Weight(tf.keras.callbacks.Callback):
    def __init__(self, w0, w1, w2, w3):
        self.w0, self.w1, self.w2, self.w3 = w0, w1, w2, w3
        self.K = 4
        self.Loss = [list() for i in range(self.K)]
        super(Weight, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.Loss[0].append(logs.get("label_loss")) # CE
        self.Loss[1].append(logs.get("lc_loss"))    # LC
        self.Loss[2].append(logs.get("coral_loss")) # CORAL
        self.Loss[3].append(logs.get("mi_loss"))    # MI

    def on_epoch_begin(self, epoch, logs=None):
        T = 1.
        if epoch >= 2:  # DWA, starting from the 3rd epoch
            wKt = list()
            for k in range(self.K):
                Lkt1, Lkt2 = self.Loss[k][epoch-1], self.Loss[k][epoch-2]
                wKt.append(Lkt1 / Lkt2)

            lambdaKt = list()
            wKt = [wkt/T - max(wKt)/T for wkt in wKt]
            for wkt in wKt:
                lambdaKt.append(self.K * math.exp(wkt) / sum(math.exp(wkt) for wkt in wKt))

            K.set_value(self.w0, lambdaKt[0])
            K.set_value(self.w1, lambdaKt[1])
            K.set_value(self.w2, lambdaKt[2])
            K.set_value(self.w3, lambdaKt[3])


def squareplus(x):
    b = 4
    return 0.5* (tf.math.sqrt(tf.math.pow(x,2) +b) +x)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, AdjMat):
        self.AdjMat = AdjMat # (num_nodes, num_nodes)
        super(Encoder, self).__init__(trainable=True)

    def build(self, input_shape):
        features = input_shape[2]   # num_features (i.e., feature dimension on each node)
        self.ARMA = spektral.layers.ARMAConv(features, order=3, iterations=7, share_weights=True,
                                             gcn_activation=squareplus, dropout_rate=0.5, activation=squareplus,
                                             use_bias=True, kernel_initializer='he_uniform')
        super(Encoder, self).build(input_shape)

    def call(self, eeg):
        return self.ARMA([eeg, self.AdjMat])    # (batch_size, num_nodes, num_features)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, AdjMat):
        self.AdjMat = AdjMat
        super(Decoder, self).__init__(trainable=True)

    def build(self, input_shape):
        features = input_shape[2]

        attn_heads = 1
        self.kernel = self.add_weight(name="kernel", shape=[features, attn_heads, features])
        self.attn_kernel_self = self.add_weight(name="attn_kernel_self", shape=[features, attn_heads, 1])
        self.attn_kernel_neighs = self.add_weight(name="attn_kernel_neighs", shape=[features, attn_heads, 1])

        super(Decoder, self).build(input_shape)

    def call(self, encode):
        encode = tf.einsum("...NI , IHO -> ...NHO", encode, self.kernel)
        attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", encode, self.attn_kernel_self)
        attn_for_neighs = tf.einsum("...NHI , IHO -> ...NHO", encode, self.attn_kernel_neighs)
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)
        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.math.reduce_mean(attn_coef, axis=2, keepdims=False)

        attn_coef = tf.math.tanh(attn_coef /50.)
        attn_coef = tf.nn.relu(attn_coef)

        mask = tf.where(self.AdjMat==0, 0., 1.)
        mask = tf.tile(tf.expand_dims(mask,0), (tf.shape(encode)[0],1,1))
        attn_coef = tf.math.multiply_no_nan(attn_coef, mask)
        return attn_coef # (batch_size, num_nodes, num_nodes)


def compute_coral(Ds, Dt):
    ns, nt = tf.cast(tf.shape(Ds)[0], 'float32'), tf.cast(tf.shape(Dt)[0], 'float32')
    d = tf.cast(tf.shape(Ds)[1], 'float32')

    DsTDs = tf.linalg.matmul(Ds, Ds, transpose_a=True)
    onesTDs = tf.linalg.matmul(tf.ones([ns,1]), Ds, transpose_a=True)
    Cs = 1./(ns-1.) * \
        (DsTDs - 1./ns*tf.linalg.matmul(onesTDs, onesTDs, transpose_a=True))

    DtTDt = tf.linalg.matmul(Dt, Dt, transpose_a=True)
    onesTDt = tf.linalg.matmul(tf.ones([nt,1]), Dt, transpose_a=True)
    Ct = 1./(nt-1.) * \
        (DtTDt - 1./nt*tf.linalg.matmul(onesTDt, onesTDt, transpose_a=True))

    coral = 1. / (4.*tf.math.square(d)) * \
        tf.math.square(tf.norm(Cs-Ct, ord='euclidean'))
    return coral

def compute_coral_c(Ds, Dt, ls, lt, c):
    coral_c = tf.TensorArray(dtype='float32', size=c, clear_after_read=False)
    for i in tf.range(c):
        Ds_i = tf.gather(Ds, indices=tf.squeeze(tf.where(ls==i)))
        Dt_i = tf.gather(Dt, indices=tf.squeeze(tf.where(lt==i)))

        ns_i = tf.size(tf.where(ls==i), out_type=tf.dtypes.float32)
        nt_i = tf.size(tf.where(lt==i), out_type=tf.dtypes.float32)
        if ns_i < 2 or nt_i < 2:
            raise ValueError('The i-th subject does not have enough samples belonging to the c-th attention class!')
        else:
            coral_i = compute_coral(Ds_i, Dt_i)

        coral_c = coral_c.write(i, coral_i)

    coral_c = coral_c.stack()
    coral_c = tf.math.reduce_mean(coral_c)
    return coral_c

class CORAL(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name='Correlation_alignment')

    def call(self, y_true, y_pred):
        domains = tf.shape(y_true)[1]   # number of subjects
        c = tf.constant(2) # number of attention classes (left / right)

        Coral_r = tf.TensorArray(dtype='float32', size=domains, clear_after_read=False)
        for i in tf.range(domains):
            Coral_c = tf.TensorArray(dtype='float32', size=domains, clear_after_read=False)
            for j in tf.range(domains):
                if j > i:
                    Ds = y_pred[:,i,:]  # (batch_size, num_features)
                    Dt = y_pred[:,j,:]

                    ls = tf.cast(y_true[:,i], 'int32')
                    lt = tf.cast(y_true[:,j], 'int32')

                    coral = compute_coral_c(Ds, Dt, ls, lt, c)
                else:
                    coral = tf.constant(0.)

                Coral_c = Coral_c.write(j, coral)

            Coral_c = Coral_c.stack()
            Coral_r = Coral_r.write(i, Coral_c)

        Coral_r = Coral_r.stack()

        domains = tf.cast(domains, 'float32')
        Coral = tf.math.reduce_sum(Coral_r) / (domains*(domains-1.)/2.)

        return Coral


def mutual_information(x, y):
    X = np.expand_dims(x, axis=-1)
    mi = mutual_info_regression(X, y, discrete_features=False, n_neighbors=3)
    return mi.astype('float32')

class Mutual_Information(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name='Mutual_information')

    def call(self, y_true, y_pred): # The ground-truth label, y_true, is not used
        batch = tf.shape(y_pred)[0]
        domains = tf.shape(y_pred)[1]
        MI = tf.TensorArray(dtype='float32', size=batch, dynamic_size=False, clear_after_read=False)

        for i in tf.range(batch):
            mi = tf.TensorArray(dtype='float32', size=domains, dynamic_size=False, clear_after_read=False)

            for j in tf.range(domains):
                encode_public = y_pred[i, j, 0, :]
                encode_private = y_pred[i, j, 1, :]

                mutual_info = tf.numpy_function(mutual_information, [encode_public, encode_private], tf.float32)
                mutual_info = tf.squeeze(mutual_info)

                mi = mi.write(j, mutual_info)

            mi = mi.stack()
            MI = MI.write(i, mi)

        MI = MI.stack()
        MI = tf.math.reduce_mean(MI, axis=(-1,-2))

        return MI

def percent(A):
    """
    Keep the top 20% of the entries of the matrix A, and set the rest to zeros
    """
    channels = 64
    ratio = .2
    A_sort = np.sort(np.ravel(A))[::-1]
    a = int(np.floor(channels * channels * ratio))
    A[A < A_sort[a]] = 0.
    return A

def AAD(directions, channels, features, domains, AdjMat):
    '''
    Args:
        directions (int): 2. The number of candidate attention directions (left or right).
        channels (int): 64. The number of EEG channels.
        features (int): 32. The dimension of the differential entropy feature on each EEG channel. It depends on how many frequency bands you have set up.
        domains (int): 15. The number of subjects. For example, there are 15 subjects used for training in the KUL dataset.
        AdjMat (np.array): The precomputed adjacency matrix with the shape of (num_subjects, num_channels, num_channels).

    Returns:
        tf.keras.Model.
            model input: EEG with the shape of (num_samples_per_subject, num_subjects, num_EEG_channels, dimension of differential entropy)
            model output: 1) Predicted label with the shape of (num_samples_per_subject, num_subjects, directions)
                          2) Reconstructed adjacency matrix with the shape of (num_samples_per_subject, num_subjects, num_channels, num_channels)
                          3) Encoded EEG by the public encoder, with the shape of (num_samples_per_subject, num_subjects, dimension of differential entropy)
                          4) Concatenation of the encoded EEG by the public encoder and that by the pravite encoder, with the shape of (num_samples_per_subject, num_subjects, 2, dimension of differential entropy)
    '''
    # Input
    eeg_in = tf.keras.Input(shape=(domains, channels, features))   # (batch_size, num_subjects, num_channels, num_features)
    eeg = tf.unstack(eeg_in, num=domains, axis=1)

    LN = tf.keras.layers.LayerNormalization(axis=(1,2))
    eeg = [LN(eeg[i]) for i in range(domains)]

    AdjMat_public = tf.math.reduce_mean(AdjMat, axis=0)
    AdjMat_private = tf.unstack(AdjMat, num=domains, axis=0)

    AdjMat_public = tf.numpy_function(percent, [AdjMat_public], tf.float32)
    AdjMat_private = [tf.numpy_function(percent, [AdjMat_private[i]], tf.float32) for i in range(domains)]

    AdjMat_public = tf.numpy_function(spektral.utils.normalized_adjacency, [AdjMat_public, True], tf.float32)
    AdjMat_private = [tf.numpy_function(spektral.utils.normalized_adjacency, [AdjMat_private[i], True], tf.float32) for i in range(domains)]

    # Encoder
    Encoder_public = Encoder(AdjMat_public)
    encode_public = [Encoder_public(eeg[i]) for i in range(domains)]

    Encoder_private = [Encoder(AdjMat_private[i]) for i in range(domains)]
    encode_private = [Encoder_private[i](eeg[i]) for i in range(domains)]

    Add = tf.keras.layers.Add()
    encode = [Add([encode_public[i], encode_private[i]]) for i in range(domains)]   # num_subjects × (batch_size, num_channels, num_features)

    # Decoder
    Decoder_all = [Decoder(AdjMat_private[i]) for i in range(domains)]
    AdjMat_rec = [Decoder_all[i](encode[i]) for i in range(domains)]   # num_subjects × (batch_size, num_channels, num_channels)

    Stack_AdjMat = tf.keras.layers.Lambda(lambda AdjMat_rec: tf.stack([AdjMat_rec[i] for i in range(domains)], axis=1), name='lc')
    AdjMat_rec = Stack_AdjMat(AdjMat_rec)   # (batch_size, num_subjects, num_channels, num_channels)

    # Pooling
    Pooling = spektral.layers.GlobalAvgPool()

    encode_public = [Pooling(encode_public[i]) for i in range(domains)]  # num_subjects × (batch_size, num_features)
    encode_private = [Pooling(encode_private[i]) for i in range(domains)]

    Stack_public = tf.keras.layers.Lambda(lambda encode_public: tf.stack([encode_public[i] for i in range(domains)], axis=1), name='coral')
    Stack_private = tf.keras.layers.Lambda(lambda encode_private: tf.stack([encode_private[i] for i in range(domains)], axis=1))

    encode_public = Stack_public(encode_public) # (batch_size, num_subjects, num_features)
    encode_private = Stack_private(encode_private)

    encode_public_private = tf.keras.layers.concatenate(
        [tf.expand_dims(encode_public, 2), tf.expand_dims(encode_private, 2)],
        axis=2, name='mi')  # (batch_size, num_subjects, 2, feature_dimension)

    # Classification
    Dense = tf.keras.layers.Dense(round((features+directions)/2), activation=squareplus, kernel_initializer='he_uniform')
    BN = tf.keras.layers.BatchNormalization()
    Softmax = tf.keras.layers.Dense(directions, activation='softmax')

    label = tf.keras.models.Sequential([Dense, BN, Softmax], name='label')(encode_public)    # (batch_size, num_subjects, directions)

    # Build a model
    model = tf.keras.Model(inputs=eeg_in,
                           outputs=[label, AdjMat_rec, encode_public, encode_public_private])
    model.summary(show_trainable=False)
    global w0, w1, w2, w3
    w0, w1, w2, w3 = K.variable(1.), K.variable(0.), K.variable(0.), K.variable(0.)
    model.compile(loss=['sparse_categorical_crossentropy', 'log_cosh', CORAL(), Mutual_Information()],
                  loss_weights=[w0, w1, w2, w3],
                  optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(5e-3, 500), weight_decay=5e-3,
                                                                   use_ema=True, amsgrad=True, jit_compile=False),
                  metrics=[['sparse_categorical_accuracy'], [], [], []],
                  run_eagerly=True)
    return model

# How to use it
segments = 10000  # The number of EEG segments per subject
AdjMat = np.random.normal(size=(15,64,64))  # Adjacency matrix for each subject

AdjMat_rec = np.zeros_like(AdjMat)
    for j in range(AdjMat_rec.shape[0]):
        AdjMat_rec[j,:,:] = percent(AdjMat[j,:,:])  # retain 20% of connections in each adjacency matrix

model = AAD(2, 64, 32, 15, AdjMat)    # KUL dataset: 2 directions (left/right), 64 EEG channels, 32 frequency bands, 15 source subjects (the remaining one is the target subject)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='label_sparse_categorical_accuracy', min_delta=1e-16, patience=10,
                                                  verbose=2, mode='auto', restore_best_weights=True)

model.fit(EEG_train,
          [LOC_train, np.repeat(np.expand_dims(AdjMat_rec,0), segments, axis=0), LOC_train, np.zeros((segments,))],    # np.zeros((segments,)) is not used by MI loss
          batch_size=64, epochs=80, verbose=1, initial_epoch=0, shuffle=True,
          callbacks=[Weight(w0, w1, w2, w3), early_stopping])
"""
EEG_train: differential entropy features for training. The shape is (num_samples_per_subject, num_subjects, num_channels, num_freqnency_bands), e.g., (10000, 15, 64, 32).

LOC_train: ground_truth label, 1 for left and 2 for right. The shape is (num_samples_per_subject, num_subjects), e.g., (10000, 15).
"""
