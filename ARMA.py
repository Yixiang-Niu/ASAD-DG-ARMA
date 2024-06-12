import tensorflow as tf
from tensorflow.keras import backend as K
import spektral
import numpy as np
import math
from sklearn.feature_selection import mutual_info_regression

class Weight(tf.keras.callbacks.Callback):    # DWA
    def __init__(self, w0, w1, w2, w3): # 必须分开，不能整合成一个张量😔
        self.w0, self.w1, self.w2, self.w3 = w0, w1, w2, w3
        self.K = 4
        self.Loss = [list() for i in range(self.K)]
        super(Weight, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.Loss[0].append(logs.get("label_loss"))
        self.Loss[1].append(logs.get("lc_loss"))
        self.Loss[2].append(logs.get("coral_loss"))
        self.Loss[3].append(logs.get("mi_loss"))

    def on_epoch_begin(self, epoch, logs=None):
        T = 1.
        if epoch >= 2:
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
        print(f'\n第{epoch+1}个epoch的Weight为{K.get_value(self.w0):.3f} {K.get_value(self.w1):.3f} {K.get_value(self.w2):.3f} {K.get_value(self.w3):.3f}')

def squareplus(x):
    b = 4
    return 0.5* (tf.math.sqrt(tf.math.pow(x,2) +b) +x)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, AdjMat):
        self.AdjMat = AdjMat # (channels, channels)
        super(Encoder, self).__init__(trainable=True)

    def build(self, input_shape):
        features = input_shape[2]
        self.ARMA = spektral.layers.ARMAConv(features, order=3, iterations=7, share_weights=True,
                                             gcn_activation=squareplus, dropout_rate=0.48, activation=squareplus,
                                             use_bias=True, kernel_initializer='he_uniform')
        super(Encoder, self).build(input_shape)

    def call(self, eeg):    # [(batch, channels, features)]
        return self.ARMA([eeg, self.AdjMat])    # ([batch, channels, features])

class Decoder(tf.keras.layers.Layer):   # 参考https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional/gat_conv.py
    def __init__(self, AdjMat):
        self.AdjMat = AdjMat # 用（正则化）邻接矩阵给attn_coef打mask
        super(Decoder, self).__init__(trainable=True)

    def build(self, input_shape):   # ([batch, channels, features])
        # batch = input_shape[0]  # 不指定batch_size则不能索引input_shape[0]！！！
        # channels = input_shape[1]
        features = input_shape[2] # 输入输出节点特征维度不变

        attn_heads = 1
        self.kernel = self.add_weight(name="kernel", shape=[features, attn_heads, features])
        self.attn_kernel_self = self.add_weight(name="attn_kernel_self", shape=[features, attn_heads, 1])
        self.attn_kernel_neighs = self.add_weight(name="attn_kernel_neighs", shape=[features, attn_heads, 1])

        # self.diag0 = tf.zeros([batch, channels])
        super(Decoder, self).build(input_shape)

    def call(self, encode): # einsum太神奇了
        encode = tf.einsum("...NI , IHO -> ...NHO", encode, self.kernel)
        attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", encode, self.attn_kernel_self)
        attn_for_neighs = tf.einsum("...NHI , IHO -> ...NHO", encode, self.attn_kernel_neighs)
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)    # 没看懂😅
        attn_coef = attn_for_self + attn_for_neighs # 与原文等价，详见https://github.com/dmlc/dgl/blob/c4aa74baf80551515c8f58244137deedea920172/python/dgl/nn/tensorflow/conv/gatconv.py
        attn_coef = tf.math.reduce_mean(attn_coef, axis=2, keepdims=False)  # 注意力头平均；([batch, channels, channels])

        attn_coef = tf.math.tanh(attn_coef /48.)    # 缩小48倍😜！
        attn_coef = tf.where(attn_coef<0., 0., attn_coef)
        # attn_coef = tf.nn.relu(attn_coef)   # 经验证，与上式等价

        mask = tf.where(self.AdjMat==0, 0., 1.)
        mask = tf.tile(tf.expand_dims(mask,0), (tf.shape(encode)[0],1,1))
        attn_coef = tf.math.multiply_no_nan(attn_coef, mask)
        return attn_coef # ([batch, channels, channels])
# -----------------------------------------------------------------------------
# class AdjMat2RL(tf.keras.layers.Layer):
#     def __init__(self):
#         super(AdjMat2RL, self).__init__(trainable=False)

#     def build(self, input_shape):   # ([batch, channels, channels])
#         # self.batch = input_shape[0]
#         # self.channels = input_shape[1]

#         super(AdjMat2RL, self).build(input_shape)

#     def call(self, AdjMat):
#         # RL = tf.zeros([self.batch, self.channels, self.channels])
#         RL = tf.zeros(tf.shape(AdjMat))
#         # for i in tf.range(self.batch):
#         for i in tf.range(tf.shape(AdjMat)[0]):
#             AdjMat_i = AdjMat[i,:,:]

#             # NL_i = tf.numpy_function(spektral.utils.normalized_laplacian, [AdjMat_i, True], tf.float32)    # symmetric: boolean, compute symmetric normalization
#             # RL_i = tf.numpy_function(spektral.utils.rescale_laplacian, [NL_i], tf.float32)
#             RL_i = tf.numpy_function(spektral.utils.normalized_adjacency, [AdjMat_i, True], tf.float32) # 实际应该用对称（不对称可以吗？）正则化邻接矩阵，而不是上面两行拉普拉斯？

#             RL_i = tf.expand_dims(RL_i, axis=0)
#             RL = tf.tensor_scatter_nd_update(RL, [[i]], RL_i)   # 学会这种更新张量的方式
#         # ................................................
#         # NL = tf.map_fn(lambda Mat: tf.numpy_function(spektral.utils.normalized_laplacian, [Mat], tf.float16), elems=AdjMat, fn_output_signature=tf.float16) # 为何有错？
#         # RL = tf.map_fn(lambda Mat: tf.numpy_function(spektral.utils.rescale_laplacian, [Mat], tf.float16), elems=NL, fn_output_signature=tf.float16)
#         return RL # ([batch, channels, channels])

def compute_coral(Ds, Dt):
    ns, nt = tf.cast(tf.shape(Ds)[0], 'float32'), tf.cast(tf.shape(Dt)[0], 'float32')
    d = tf.cast(tf.shape(Ds)[1], 'float32') # 或tf.cast(tf.shape(Dt)[1], 'float32')

    DsTDs = tf.linalg.matmul(Ds, Ds, transpose_a=True)
    onesTDs = tf.linalg.matmul(tf.ones([ns,1]), Ds, transpose_a=True)   # tf.ones支持float维度
    Cs = 1./(ns-1.) * \
        (DsTDs - 1./ns*tf.linalg.matmul(onesTDs, onesTDs, transpose_a=True)) # ([features, features])

    DtTDt = tf.linalg.matmul(Dt, Dt, transpose_a=True)
    onesTDt = tf.linalg.matmul(tf.ones([nt,1]), Dt, transpose_a=True)
    Ct = 1./(nt-1.) * \
        (DtTDt - 1./nt*tf.linalg.matmul(onesTDt, onesTDt, transpose_a=True))

    coral = 1. / (4.*tf.math.square(d)) * \
        tf.math.square(tf.norm(Cs-Ct, ord='euclidean'))
    return coral

def compute_coral_c(Ds, Dt, ls, lt, c):   # D: float32(batch, d); l: int32 (batch); c: int32!!!
    coral_c = tf.TensorArray(dtype='float32', size=c, clear_after_read=False)
    for i in tf.range(c):
        Ds_i = tf.gather(Ds, indices=tf.squeeze(tf.where(ls==i)))   # tf.where返回多一个无用维度
        Dt_i = tf.gather(Dt, indices=tf.squeeze(tf.where(lt==i)))

        ns_i = tf.size(tf.where(ls==i), out_type=tf.dtypes.float32)
        nt_i = tf.size(tf.where(lt==i), out_type=tf.dtypes.float32)
        if ns_i < 2 or nt_i < 2:
            # raise ValueError('某个域的某个类别样本数太少')   # 只有分类后由于batch中样本的随机性可能导致某类样本缺失
            coral_i = 0.
        else:
            coral_i = compute_coral(Ds_i, Dt_i)

        coral_c = coral_c.write(i, coral_i)

    coral_c = coral_c.stack()
    coral_c = tf.math.reduce_mean(coral_c)  # 各类求平均！😊
    return coral_c

class CORAL_C(tf.keras.losses.Loss):    # 按类别相互对齐（不同类分别对齐后求平均）
    def __init__(self):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name='Correlation_alignment_class')   # 损失函数返回长度为batch size的向量，model.fit时自动求平均；或者提前自己求平均；注意这里如何向父类传递参数

    def call(self, y_true, y_pred): # int8([batch, domains])；([batch, domains, features])
        domains = tf.shape(y_true)[1]
        c = tf.constant(2) # 假定就两类！

        Coral_r = tf.TensorArray(dtype='float32', size=domains, clear_after_read=False) # size必为int32
        for i in tf.range(domains):
            Coral_c = tf.TensorArray(dtype='float32', size=domains, clear_after_read=False)
            for j in tf.range(domains):
                if j > i:   # 比过的不再比，且不跟自己比
                    Ds = y_pred[:,i,:]  # ([batch, features])
                    Dt = y_pred[:,j,:]

                    ls = tf.cast(y_true[:,i], 'int32')  # 可以自动精度转换吗？
                    lt = tf.cast(y_true[:,j], 'int32')

                    coral = compute_coral_c(Ds, Dt, ls, lt, c)
                else:
                    coral = tf.constant(0.)

                Coral_c = Coral_c.write(j, coral)

            Coral_c = Coral_c.stack()
            Coral_r = Coral_r.write(i, Coral_c)

        Coral_r = Coral_r.stack()   # 右上三角无主对角线([domains, domains])

        domains = tf.cast(domains, 'float32')
        Coral = tf.math.reduce_sum(Coral_r) / (domains*(domains-1.)/2.)   # 被试（域）对儿平均！😊

        return Coral
# -------------------------------------- 老CORAL（相互对齐不管类别） ---------------------------------------
# class CORAL(tf.keras.losses.Loss):  # 经验证，与他人基本相等
#     def __init__(self):
#         super().__init__(reduction=tf.keras.losses.Reduction.NONE, name='Correlation_alignment')   # 损失函数返回长度为batch size的向量，model.fit时自动求平均；或者提前自己求平均；注意这里如何向父类传递参数

#     def call(self, y_true, y_pred): # y_true不用，随便赋一个；([batch, domains, features])
#         batch = tf.cast(tf.shape(y_pred)[0], 'float32')
#         domains = tf.shape(y_pred)[1]   # 自动tf.int32
#         features = tf.cast(tf.shape(y_pred)[2], 'float32')

#         if domains > 1: # 至少有两个域；tf布尔可以这样
#             Coral_r = tf.TensorArray(dtype='float32', size=domains, dynamic_size=False, clear_after_read=False) # size必为int32
#             for i in tf.range(domains):
#                 Coral_c = tf.TensorArray(dtype='float32', size=domains, dynamic_size=False, clear_after_read=False)
#                 for j in tf.range(domains):
#                     if j > i:   # 比过的不再比，且不跟自己比
#                         Ds = y_pred[:,i,:]  # ([batch, features])
#                         Dt = y_pred[:,j,:]

#                         DsTDs = tf.linalg.matmul(Ds, Ds, transpose_a=True)
#                         onesTDs = tf.linalg.matmul(tf.ones([batch,1]), Ds, transpose_a=True)
#                         Cs = 1./(batch-1.) * \
#                             (DsTDs - 1./batch*tf.linalg.matmul(onesTDs, onesTDs, transpose_a=True)) # ([features, features])

#                         DtTDt = tf.linalg.matmul(Dt, Dt, transpose_a=True)
#                         onesTDt = tf.linalg.matmul(tf.ones([batch,1]), Dt, transpose_a=True)
#                         Ct = 1./(batch-1.) * \
#                             (DtTDt - 1./batch*tf.linalg.matmul(onesTDt, onesTDt, transpose_a=True))

#                         coral = 1./(4.*tf.math.square(features)) * \
#                                 tf.math.square(tf.norm(Cs-Ct, ord='euclidean'))   # Default is 'euclidean' which is equivalent to Frobenius norm if tensor is a matrix and equivalent to 2-norm for vectors；等价于tf.reduce_sum(tf.multiply(Cs-Ct, Cs-Ct))
#                     else:
#                         coral = tf.constant(0.)

#                     Coral_c = Coral_c.write(j, coral)

#                 Coral_c = Coral_c.stack()
#                 Coral_r = Coral_r.write(i, Coral_c)

#             Coral_r = Coral_r.stack()   # 右上三角无主对角线([domains, domains])

#             domains = tf.cast(domains, 'float32')
#             Coral = tf.math.reduce_sum(Coral_r) / (domains*(domains-1.)/2.)   # 平均；([])

#         else:
#             Coral = tf.constant(0.) # ([])

#         return Coral

def mutual_information(x, y):
    X = np.expand_dims(x, axis=-1)
    mi = mutual_info_regression(X, y, discrete_features=False, n_neighbors=3)
    return mi.astype('float32') # 默认float64

class Mutual_Information(tf.keras.losses.Loss):
    def __init__(self):    # 经验证，SUM_OVER_BATCH_SIZE将一个batch取均值！
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name='Mutual_information')

    def call(self, y_true, y_pred): # ([batch, domains, 2, features])
        batch = tf.shape(y_pred)[0]
        domains = tf.shape(y_pred)[1]
        MI = tf.TensorArray(dtype='float32', size=batch, dynamic_size=False, clear_after_read=False)    # size必为int32！

        for i in tf.range(batch):
            mi = tf.TensorArray(dtype='float32', size=domains, dynamic_size=False, clear_after_read=False)

            for j in tf.range(domains):
                encode_public = y_pred[i, j, 0, :]
                encode_private = y_pred[i, j, 1, :]

                mutual_info = tf.numpy_function(mutual_information, [encode_public, encode_private], tf.float32)   # self.表示类内调用
                mutual_info = tf.squeeze(mutual_info)   # ([])

                mi = mi.write(j, mutual_info)

            mi = mi.stack()
            MI = MI.write(i, mi)

        MI = MI.stack()  # # ([batch, domains])
        MI = tf.math.reduce_mean(MI, axis=(-1,-2))  # 平均batch和被试（域）！😊

        return MI # 维度和精度均不变

def percent(A):
    channels = 64
    ratio = .2
    A_sort = np.sort(np.ravel(A))[::-1]
    a = int(np.floor(channels * channels * ratio))  # ????????????????????????????????????????????????????????????????????????????
    A[A < A_sort[a]] = 0.
    return A

def AAD(sources, channels, features, domains, AdjMat):  # AdjMat不作为model的Input
    # ------------------------------ 输入 ------------------------------
    eeg_in = tf.keras.Input(shape=(domains, channels, features))   # ([batch, domains, channels, features])
    eeg = tf.unstack(eeg_in, num=domains, axis=1)  # 返回列表[domains]，每个元素为([batch, channels, features])

    LN = tf.keras.layers.LayerNormalization(axis=(1,2)) # 共享👍
    eeg = [LN(eeg[i]) for i in range(domains)]

    AdjMat_public = tf.math.reduce_mean(AdjMat, axis=0) # 共享👍
    AdjMat_private = tf.unstack(AdjMat, num=domains, axis=0)    # (domains个[channels, channels])

    AdjMat_public = tf.numpy_function(percent, [AdjMat_public], tf.float32)
    AdjMat_private = [tf.numpy_function(percent, [AdjMat_private[i]], tf.float32) for i in range(domains)]

    AdjMat_public = tf.numpy_function(spektral.utils.normalized_adjacency, [AdjMat_public, True], tf.float32)
    AdjMat_private = [tf.numpy_function(spektral.utils.normalized_adjacency, [AdjMat_private[i], True], tf.float32) for i in range(domains)]
    # ----------------------------- ■ 特征解耦 ------------------------------
    Encoder_public = Encoder(AdjMat_public) # 共享👍
    encode_public = [Encoder_public(eeg[i]) for i in range(domains)]

    Encoder_private = [Encoder(AdjMat_private[i]) for i in range(domains)]
    encode_private = [Encoder_private[i](eeg[i]) for i in range(domains)]

    Add = tf.keras.layers.Add()
    encode = [Add([encode_public[i], encode_private[i]]) for i in range(domains)]   # ([batch, channels, features])
    # ................................................
    Decoder_all = [Decoder(AdjMat_private[i]) for i in range(domains)]
    AdjMat_rec = [Decoder_all[i](encode[i]) for i in range(domains)]   # 重构私有邻接矩阵；([batch, channels, channels])
    # ------------------------------------------------------------------
    Pooling = spektral.layers.GlobalAvgPool()   # 各通道平均

    encode_public = [Pooling(encode_public[i]) for i in range(domains)]  # ([batch, features])
    encode_private = [Pooling(encode_private[i]) for i in range(domains)]
    # ------------------------ ★ 域泛化：CORAL ------------------------
    Stack_public = tf.keras.layers.Lambda(lambda encode_public: tf.stack([encode_public[i] for i in range(domains)], axis=1), name='coral')
    Stack_private = tf.keras.layers.Lambda(lambda encode_private: tf.stack([encode_private[i] for i in range(domains)], axis=1))
    Stack_AdjMat = tf.keras.layers.Lambda(lambda AdjMat_rec: tf.stack([AdjMat_rec[i] for i in range(domains)], axis=1), name='lc')

    encode_public = Stack_public(encode_public) # ([batch, domains, features])
    encode_private = Stack_private(encode_private)
    AdjMat_rec = Stack_AdjMat(AdjMat_rec)

    encode_public_private = tf.keras.layers.concatenate(
        [tf.expand_dims(encode_public, 2), tf.expand_dims(encode_private, 2)],
        axis=2, name='mi')  # ([batch, domains, 2, features])
    # -------------------------- ■ FC ---------------------------
    Dense = tf.keras.layers.Dense(round((features+sources)/2), activation=squareplus, kernel_initializer='he_uniform') # 共享标签分类器
    BN = tf.keras.layers.BatchNormalization()
    Softmax = tf.keras.layers.Dense(sources, activation='softmax')    # 给输出Dense层命名

    label = tf.keras.models.Sequential([Dense, BN, Softmax], name='label')(encode_public)    # ([batch, domain, sources])
    # ---------------------------- 构建模型 -----------------------------
    model = tf.keras.Model(inputs=eeg_in,
                           outputs=[label, AdjMat_rec, encode_public, encode_public_private])
    model.summary(show_trainable=False)
    global w0, w1, w2, w3
    w0, w1, w2, w3 = K.variable(1.), K.variable(0.), K.variable(0.), K.variable(0.)
    model.compile(loss=['sparse_categorical_crossentropy', 'log_cosh', CORAL(), Mutual_Information()],   # 所有loss都是batch和被试（域）上平均！😊loss精度自动跟随输入（int8与float16输出为float16）
                  loss_weights=[w0, w1, w2, w3],    # 在自定义Callback中调节其权重；无需重新compile；列表
                  optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(4.8e-3, 433), weight_decay=4.8e-3,
                                                                   use_ema=True, amsgrad=True, jit_compile=False), # Currently tf.py_function is not compatible with XLA. Calling tf.py_function inside tf.function(jit_compile=True) will raise an error
                  metrics=[['sparse_categorical_accuracy'], [], [], []],    # metrics必须放在列表里！
                  run_eagerly=False) # 用于在model中（如自定义loss里）获取具体的值；但eager极其慢！
    return model

model = AAD(sources, channels, features, domains, AdjMat)

AdjMat_rec = np.zeros_like(AdjMat)  # 不要正则化
for j in range(AdjMat_rec.shape[0]):
    AdjMat_rec[j,:,:] = percent(AdjMat[j,:,:])

model.fit(EEG_train,
          [LOC_train, np.repeat(np.expand_dims(AdjMat_rec,0), max(segments), axis=0), LOC_train, np.zeros((max(segments),1), dtype='int8')],    # 必须给y_true！
          batch_size=batch_size, epochs=epochs, verbose=1, initial_epoch=0, shuffle=True,
          callbacks=[Weight(w0, w1, w2, w3)])