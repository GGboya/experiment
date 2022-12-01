#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from scipy import stats
import scipy.io as sio
from scipy.sparse.linalg import svds
import sklearn as sk
from sklearn import cluster
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
from sklearn.cluster import KMeans
# import xlsxwriter
from random import choice, shuffle
from numpy import array
from numpy.linalg import cholesky

from sklearn.metrics import f1_score
import time
import sys
sys.path.insert(0, 'D:/python')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.logging.set_verbosity(tf.logging.ERROR)



class ConvAE(object):

    def __init__(self, flag, n_input, kernel_size, n_hidden, m, k, dictionary=None, reg_constant1=None,
                 re_constant2=None,
                 re_constant3=None,
                 batch_size=200, reg=None,
                 denoise=False, model_path=None, restore_path=None,
                 logs_path=None):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.m = m  # the middle level of the dictionary part
        self.k = k  # the number of the dictionary atoms
        self.dictionary = dictionary
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        # print("问题处在ConvAE吗")
        self.x = tf.placeholder(tf.float32, [None,  self.n_input[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])
        weights = self._initialize_weights()



        if denoise == False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)  # 传入权重，传入输入，得到卷积后的一个张量
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                      mean=0,
                                                      stddev=0.2,
                                                      dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)
        print("latent",latent)
        print("latent_dimensions", latent_dimensions)
        latent_X = tf.reshape(latent, [-1, latent_dimensions])   # 对应d维，xi
        print("latent_X得维度",tf.shape(latent_X))
        print("latent_X",latent_X)


        # 两个带偏置的非线性层，一个无偏置的线性层
        # middle = self.add_dense_non_linear_layer(latent_X, weights["dense_nonlinear_w0"], weights["dense_nonlinear_b0"])
        latent_C = self.add_dense_non_linear_layer(latent_X, weights["dense_nonlinear_w1"], weights["dense_nonlinear_b1"])
        print("latent_C的维度",latent_C)  # d维得一个向量，比如1100*2500 传入得数据集得样本数为1100，2500相当于是特征
        D = weights["dense_linear_w"]    # A，2500 *
        latent_Ci = tf.matmul(latent_C, D)  # A*Ci
        decoder_input = tf.reshape(latent_Ci, tf.shape(latent)) # 要放入解码器，就需要跟编码器的输出一样，所以也要reshape


        self.Coef = latent_C
        self.latent_C = latent_C
        self.latent_Ci = latent_Ci
        self.x_r = self.decoder(decoder_input, weights, shape) # 放入解码器，让解码器得到最后的输出，X_r
        print(self.x_r)


        #损失
        self.reconst_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))
        tf.summary.scalar("recons_loss", self.reconst_cost)
        # self.sparse_losses = tf.reduce_sum(tf.abs(self.Coef))
        self.sparse_losses = tf.reduce_sum(tf.pow(self.Coef, 2.0))
        # 这里用的是一范数呢
        tf.summary.scalar("reg_loss", reg_constant1 * self.sparse_losses)
        self.dict_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(latent_X, latent_Ci), 2.0))
        # F范数，平方后再开方，由于公式还有一个平方，所以就直接等于开方
        tf.summary.scalar("dict_loss", re_constant2 * self.dict_losses)
        self.loss = self.reconst_cost * reg_constant1 + re_constant2 * self.sparse_losses + re_constant3 * self.dict_losses
        self.merged_summary_op = tf.summary.merge_all()

        # 我们的损失函数，由三个部分组成，最后要优化的就是这个损失函数


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss) # 训练
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.init = tf.global_variables_initializer()

        #测试能否使用GPU加速
        # self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)

        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("dense"))])
        self.saver_all = tf.train.Saver([v for v in tf.trainable_variables()])
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


    # 初始化网络参数
    def _initialize_weights(self):
        all_weights = dict()  # 字典
        n_layers = len(self.n_hidden)

        # 初始化编码器参数
        all_weights['enc_w0'] = tf.get_variable("enc_w0",
                                                shape=[self.kernel_size[0], 1, n_hidden[0]],
                                                initializer=layers.variance_scaling_initializer(), regularizer=self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([n_hidden[0]], dtype=tf.float32))



        all_weights['enc_w1'] = tf.get_variable("enc_w1",
                                                shape=[self.kernel_size[1], n_hidden[0],
                                                       n_hidden[1]],
                                                initializer=layers.variance_scaling_initializer(), regularizer=self.reg)
        all_weights['enc_b1'] = tf.Variable(tf.zeros([n_hidden[1]], dtype=tf.float32))

        all_weights['enc_w2'] = tf.get_variable("enc_w2",
                                                shape=[self.kernel_size[2], n_hidden[1],
                                                       n_hidden[2]],
                                                initializer=layers.variance_scaling_initializer(), regularizer=self.reg)
        all_weights['enc_b2'] = tf.Variable(tf.zeros([n_hidden[2]], dtype=tf.float32))

        # 初始化全连接神经网络的参数
        shape_c = latent_dimensions
        all_weights["dense_nonlinear_w0"] = tf.get_variable(name="dense_nonlinear_w0",
                                                            shape=[shape_c, self.m],
                                                            dtype=tf.float32,
                                                            initializer=layers.variance_scaling_initializer())
        all_weights["dense_nonlinear_b0"] = tf.Variable(tf.zeros(self.m, dtype=tf.float32), name="dense_nonlinear_b0")
        all_weights["dense_nonlinear_w1"] = tf.get_variable(name="dense_nonlinear_w1",
                                                            shape=[shape_c, self.k],
                                                            dtype=tf.float32,
                                                            initializer=layers.variance_scaling_initializer())
        all_weights["dense_nonlinear_b1"] = tf.Variable(tf.zeros(self.k, dtype=tf.float32), name="dense_nonlinear_b1")
        # initial with svd dictionary
        #         all_weights["dense_linear_w"] = tf.Variable(self.dictionary, dtype=tf.float32, name="dense_linear_w")
        all_weights["dense_linear_w"] = tf.Variable(1.0e-4 * tf.ones([self.k, shape_c], tf.float32),
                                                    name='dense_linear_w')
        # 3 initial with xavier_initializer
        #         all_weights["dense_linear_w"] = tf.get_variable(name="dense_linear_w",
        #                                       shape=[self.k,1080],
        #                                       dtype=tf.float32,
        #                                       initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # all_weights["dense_linear_b"] = tf.Variable(tf.zeros(shape_c, dtype=tf.float32), name="dense_linear_b")

        # 初始化解码器参数
        all_weights['dec_w0'] = tf.get_variable("dec_w0",
                                                shape=[self.kernel_size[2], n_hidden[1],
                                                       n_hidden[2]],
                                                initializer=layers.variance_scaling_initializer(), regularizer=self.reg)
        all_weights['dec_b0'] = tf.Variable(tf.zeros([n_hidden[1]], dtype=tf.float32))

        all_weights['dec_w1'] = tf.get_variable("dec_w1",
                                                shape=[self.kernel_size[1], n_hidden[0],n_hidden[1]],
                                                initializer=layers.variance_scaling_initializer(), regularizer=self.reg)
        all_weights['dec_b1'] = tf.Variable(tf.zeros([n_hidden[0]], dtype=tf.float32))

        all_weights['dec_w2'] = tf.get_variable("dec_w2",
                                                shape=[self.kernel_size[0],1, n_hidden[0]],
                                                initializer=layers.variance_scaling_initializer(), regularizer=self.reg)
        all_weights['dec_b2'] = tf.Variable(tf.zeros([1], dtype=tf.float32))

        return all_weights

        # 编码器

    def encoder(self, x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        layer1 = tf.nn.bias_add(tf.nn.conv1d(x, weights['enc_w0'], stride=1, padding='SAME'),
                                weights['enc_b0'])

        layer1 = tf.nn.sigmoid(layer1)
        shapes.append(layer1.get_shape().as_list())
        layer2 = tf.nn.bias_add(tf.nn.conv1d(layer1, weights['enc_w1'], stride=1, padding='SAME'),
                                weights['enc_b1'])
        layer2 = tf.nn.sigmoid(layer2)
        shapes.append(layer2.get_shape().as_list())
        layer3 = tf.nn.bias_add(tf.nn.conv1d(layer2, weights['enc_w2'], stride=1, padding='SAME'),
                                weights['enc_b2'])
        layer3 = tf.nn.sigmoid(layer3)

        return layer3, shapes

        # 解码器

    def decoder(self, z, weights, shapes):
        # Encoder Hidden layer with relu activation #1
        shape_de1 = shapes[2]

        layer1 = tf.add(tf.nn.conv1d_transpose(z, weights['dec_w0'], tf.stack(
            [tf.shape(self.x)[0], shape_de1[1], shape_de1[2]]), \
                                               strides=[1], padding='SAME'), weights['dec_b0'])
        layer1 = tf.nn.sigmoid(layer1)

        shape_de2 = shapes[1]
        layer2 = tf.add(tf.nn.conv1d_transpose(layer1, weights['dec_w1'], tf.stack(
            [tf.shape(self.x)[0], shape_de2[1], shape_de2[2]]), \
                                               strides=[1], padding='SAME'), weights['dec_b1'])
        layer2 = tf.nn.sigmoid(layer2)

        shape_de3 = shapes[0]
        layer3 = tf.add(tf.nn.conv1d_transpose(layer2, weights['dec_w2'], tf.stack(
            [tf.shape(self.x)[0], shape_de3[1], shape_de3[2]]), \
                                               strides=[1], padding='SAME'), weights['dec_b2'])
        layer3 = tf.nn.sigmoid(layer3)

        return layer3




    def add_dense_linear_layer(self, input_layer, weight, bias):
        output_layer = tf.nn.xw_plus_b(x=input_layer,
                                       weights=weight,
                                       biases=bias,
                                       name=None)
        return output_layer

    def add_dense_non_linear_layer(self, input_layer, weight, bias):
        output_layer = tf.nn.xw_plus_b(x=input_layer,
                                       weights=weight,
                                       biases=bias,
                                       name=None)
        return tf.nn.sigmoid(output_layer)


    # cost1：重建损失， cost2：系数损失， cost3：字典损失， Coef : 全连接系数，自表达系数
    def partial_fit(self, X, lr):  # X为一个batch
        cost1, cost2, cost3,  summary, _, Coef = self.sess.run((self.reconst_cost, self.sparse_losses, self.dict_losses,
                                                               self.merged_summary_op, self.optimizer, self.Coef),
                                                              feed_dict={self.x: X, self.learning_rate: lr})  #
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost1, cost2, cost3,  Coef

    def convert_z(self, t):
        self.z = t


    def get_latent_C(self, X):
        print("self.latent_C的维度",self.latent_C)
        print("X的维度",X.shape)
        print("self.x:",self.x)
        latent_C = self.sess.run((self.latent_C), feed_dict={self.x: X})
        return latent_C

    def get_latent_Ci(self, X):
        latent_Ci = self.sess.run((self.latent_Ci), feed_dict={self.x: X})
        return latent_Ci

    def initlization(self):
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def save_model_all(self):
        save_path = self.saver_all.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore(self):
        path = self.restore_path
        self.saver.restore(self.sess, path)
        print("model restored: %s" % path)

    def restore_all(self):
        path = self.model_path
        self.saver_all.restore(self.sess, path)
        print("model restored: %s" % path)

    def close(self):
        self.sess.close()
        print("close session")



# 生成下一个批量
def next_batch(data, _index_in_epoch, batch_size, _epochs_completed):

    _num_examples = data.shape[0]
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
        # Finished epoch
        _epochs_completed += 1
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        data = data[perm]
        # label = label[perm]
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch
    return data[start:end], _index_in_epoch, _epochs_completed






def best_map(L1, L2): # 先看其他部分
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)  # Find the unique elements of an array.
    # print("Label1",Label1)
    nClass1 = len(Label1)  # num of unique elements in L1
    # print("nClass1",nClass1)
    Label2 = np.unique(L2)
    # print("Label2", Label2)
    nClass2 = len(Label2)
    # print("nClass2", nClass2)
    nClass = np.maximum(nClass1, nClass2)

    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2




# def build_laplacian(C):
# def thrC(C, ro):
# def build_aff(C):
# def post_process_latentC(C, K, d, alpha):




def post_process_latentC(C, K): # 谱聚类
    C_norm = normalize(C, norm='l2', axis=1, copy=True, return_norm=False)
    # C_norm = C
    neighbors_graph = kneighbors_graph(C_norm,
                                       n_neighbors=cnt_neighbor,
                                       mode='connectivity',
                                       metric="cosine", ).toarray()
    L = neighbors_graph
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K,
                                          eigen_solver='arpack',
                                          affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    group = spectral.fit_predict(L) + 1
    return group, L




def err_rate(gt_s, s): # gt_s就是Label，s就是y_x，我们预测的
    c_x = best_map(gt_s, s) # 不需要改
    a = np.sum(gt_s[:] != s[:])
    err_x = np.sum(gt_s[:] != c_x[:])
    true_label = tf.squeeze(gt_s, axis = 0)
    with tf.Session() as sess:
        true_label = true_label.eval()

    mf1 = f1_score(true_label, c_x, average='macro')
    result_NMI = metrics.normalized_mutual_info_score(c_x, true_label)
    result_homo = homogeneity_score(true_label, c_x)
    result_ARI = adjusted_rand_score(true_label, c_x)
    # print("err_x:",err_x)
    missrate = err_x.astype(float) / (gt_s.shape[1])
    return missrate, result_NMI, result_ARI, result_homo, mf1



def train_face(Img, CAE, learning_rate, epoch, display_step, batch_size):
    CAE.initlization()
    CAE.restore()  # restore from pretrain model

    display_step = display_step
    lr = learning_rate
    bs = batch_size
    it = 0
    repeat = Img.shape[0] // batch_size
    _index_in_epoch = 0
    _epochs = 0

    while it < epoch:
        it += 1
        for i in range(repeat):
            batch_x, _index_in_epoch, _epochs = next_batch(Img, _index_in_epoch, batch_size, _epochs)
            batch_x = np.reshape(batch_x, [batch_size, n_input[1], 1])
            cost1, cost2, cost3,  latent_C = CAE.partial_fit(batch_x, lr)



            avg_cost1 = cost1 / (batch_size)
            avg_cost2 = cost2 / (batch_size)
            avg_cost3 = cost3 / (batch_size)

        if it % display_step == 0:
            print("epoch: {:4d} ; recon_cost: {:16.8f}; sparse_cost: {:16.8f}; dict_cost: {:16.8f};".format(it, avg_cost1,
                                                                                                         avg_cost2,
                                                                                                         avg_cost3,
                                                                                       ))
            # latent_C = CAE.get_latent_C(Img)  # 稀疏表示,这一步错了
            # y_x, _ = post_process_latentC(latent_C, num)  # 3代表有3类 ，这一步出Bug
            # missrate_x, NMI, ARI, homo, mf1 = err_rate(Label, y_x)
            # acc_x = 1 - missrate_x
            # print("-------------------------------------------")
            # print('模型结果: acc = {:.3f}, NMI = {:.3f}, ARI = {:.3f}, Homogeneity = {:.3f},mf1 = {:.3f}'.format(acc_x, NMI,
            #                                                                                                  ARI, homo,
            #                                                                                                  mf1))

    global pre_train
    if pre_train == False:
        CAE.save_model_all()
    return

def residual_average_gini_index(gene_scores, df_clusters,
                                housekeeping_genes, marker_genes,
                                min_cells_per_cluster=10):
    # Subset from the main matrix the housekeeping genes and marker genes
    df_matrix_housekeeping = gene_scores.loc[gene_scores.index.intersection(housekeeping_genes),]
    df_matrix_marker = gene_scores.loc[gene_scores.index.intersection(marker_genes),]

    # Define a function to compute the Gini score
    def gini(list_of_values):
        sorted_list = sorted(list_of_values)
        height, area = 0, 0
        for value in sorted_list:
            height += value
            area += height - value / 2.
            fair_area = height * len(list_of_values) / 2.
        return (fair_area - area) / fair_area

    # Function to calculate Gini value for all the genes
    def calculate_gini(df_matrix, gene_name, clustering_info):
        return gini(get_avg_per_cluster(df_matrix, gene_name, clustering_info, use_log2=False))

    # Function to calculate Gini value for all the genes
    def calculate_gini_values(df_matrix, clustering_info):
        gini_values = []
        for gene_name in df_matrix.index:
            gini_values.append(calculate_gini(df_matrix, gene_name, clustering_info))
        return gini_values

    # Write a function to compute delta difference of the average accessibility in Marker vs Housekeeping and Kolmogorov Smirnov test
    def score_clustering_solution(df_matrix_marker, df_matrix_housekeeping, clustering_info):
        gini_values_housekeeping = calculate_gini_values(df_matrix_housekeeping, clustering_info)
        gini_values_marker = calculate_gini_values(df_matrix_marker, clustering_info)
        statistic, p_value = stats.ks_2samp(gini_values_marker, gini_values_housekeeping)

        return np.mean(gini_values_marker), np.mean(gini_values_housekeeping), np.mean(gini_values_marker) - np.mean(
            gini_values_housekeeping), statistic, p_value

    # Function to compute the average accessibility value per cluster
    def get_avg_per_cluster(df_matrix, gene_name, clustering_info, use_log2=False):
        N_clusters = len(clustering_info.index.unique())
        avg_per_cluster = np.zeros(N_clusters)
        for idx, idx_cluster in enumerate(sorted(np.unique(clustering_info.index.unique()))):
            if use_log2:
                values_cluster = df_matrix.loc[gene_name, clustering_info.loc[idx_cluster, :].values.flatten()].apply(
                    lambda x: np.log2(x + 1))
            else:
                values_cluster = df_matrix.loc[gene_name, clustering_info.loc[idx_cluster, :].values.flatten()]

            avg_per_cluster[idx] = values_cluster.mean()
            if avg_per_cluster[idx] > 0:
                avg_per_cluster[idx] = avg_per_cluster[idx]  # /values_cluster.std()

        return avg_per_cluster

    # Run the method for all the clustering solutions

    df_metrics = pd.DataFrame(
        columns=['Method', 'Clustering', 'Gini_Marker_Genes', 'Gini_Housekeeping_Genes', 'Difference', 'KS_statistics',
                 'p-value'])

    for method in df_clusters.columns:
        print(method)
        df_method_i = df_clusters[method]
        clustering_info = pd.DataFrame(df_method_i)
        clustering_info['Barcode'] = clustering_info.index
        clustering_info = clustering_info.set_index(method)

        # REMOVE CLUSTERS WITH FEW CELLS
        cluster_sizes = pd.value_counts(clustering_info.index)
        clustering_info = clustering_info.loc[cluster_sizes[cluster_sizes > min_cells_per_cluster].index.values, :]

        mean_gini_marker, mean_gini_housekeeping, mean_gini_difference, statistics, p_value = score_clustering_solution(
            df_matrix_marker, df_matrix_housekeeping, clustering_info)

        df_metrics = df_metrics.append({'Method': method, 'Clustering': method,
                                        'Gini_Marker_Genes': mean_gini_marker,
                                        'Gini_Housekeeping_Genes': mean_gini_housekeeping,
                                        'Difference': mean_gini_difference, 'KS_statistics': statistics,
                                        'p-value': p_value},
                                       ignore_index=True)

    return df_metrics


# 计算准确率
def accurate(Img, Label, CAE):
    CAE.initlization()
    CAE.restore_all()  # restore from fine-tuned model
    latent_C = CAE.get_latent_C(Img) # 稀疏表示,这一步错了

    print("latent_C.shape:",len(latent_C),len(latent_C[0]))
    print("我们的稀疏表示:",latent_C)

    y_x, _ = post_process_latentC(latent_C, num) # 3代表有3类 ，这一步出Bug
    # y_x = np.reshape(-1,200)

    print("y_x.shape",y_x.shape)
    print("经过谱聚类之后的稀疏表示",y_x)
    # print("Label",Label)


    df_clusters = pd.DataFrame(index=metadata.index)

    gene_scores = pd.read_csv('FM_GeneScoring_10xpbmc5k.tsv',
                              sep='\t', index_col=0)
    # https://www.tau.ac.il/~elieis/Housekeeping_genes.html List of Housekeeping genes
    housekeeping_genes = ['ACTB', 'ALDOA', 'GAPDH', 'PGK1', 'LDHA', 'RPS27A', 'RPL19', 'RPL11', 'NONO', 'ARHGDIA',
                          'RPL32', 'RPS18', 'HSPCB',
                          'C1orf43', 'CHMP2A', 'EMC7', 'GPI', 'PSMB2,''PSMB4', 'RAB7A', 'REEP5', 'SNRPD3', 'VCP',
                          'VPS29']

    # List of Marker Genes
    marker_genes = ['CD209', 'ENG', 'FOXP3', 'CD34', 'BATF3', 'S100A12', 'THBD', 'CD3D', 'THY1', 'CD8A', 'CD8B', 'CD14',
                    'PROM1', 'IL2RA', 'FCGR3A',
                    'IL3RA', 'FCGR1A', 'CD19', 'IL7R', 'CD79A', 'MS4A1', 'NCAM1', 'CD3E', 'CD3G', 'KIT', 'CD1C', 'CD68',
                    'CD4']

    y = {"our_model": y_x}
    df = pd.DataFrame(y)
    df.index = metadata.index
    df_clusters = pd.merge(df_clusters, df, left_index=True, right_index=True)
    df_metrics = residual_average_gini_index(gene_scores, df_clusters,
                                             housekeeping_genes, marker_genes,
                                             min_cells_per_cluster=10)

    missrate_x , NMI, ARI,homo, mf1= err_rate(Label, y_x)

    acc_x = 1 - missrate_x
    print("-------------------------------------------")
    print('模型结果: acc = {:.3f}, NMI = {:.3f}, ARI = {:.3f}, Homogeneity = {:.3f},mf1 = {:.3f}, time = {:.3f}'.format(acc_x, NMI, ARI, homo, mf1, time.time()-start_time))
    return acc_x, NMI, ARI, homo, mf1, df_metrics['Difference'][0]




def acc_run(m, k, lr, epoch, batch_size, reg1, reg2, reg3):

    print("问题处在acc_run嘛")
    parent_path = "./train_model"
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    model_dir = parent_path
    model_path = model_dir + "/model" + str(int(time.time())) + ".ckpt"
    # model_path = model_dir + "/model" + str(epoch) + "_" + str(batch_size) + ".ckpt"

    restore_path = pretrain_model_path

    if not os.path.exists("./logs_path"):
        os.mkdir("./logs_path")
    logs_path = "./logs_path"

    # 上面的代码加载了与训练的参数，预训练的模型

    tf.reset_default_graph()
    # 构图，如果只需要预训练，下面这个训练的步骤，就完全不需要
    global pre_train

    CAE_train = ConvAE(flag=0,
                       n_input=n_input,               # 学习到了好的参数
                       n_hidden=n_hidden,
                       m=m,
                       k=k,
                       dictionary=dictionary,
                       reg_constant1=reg1,
                       re_constant2=reg2,
                       re_constant3=reg3,
                       kernel_size=kernel_size,
                       batch_size=batch_size,
                       model_path=model_path,
                       restore_path=restore_path,
                       logs_path=logs_path)
    train_face(Img=Img, CAE=CAE_train, learning_rate=lr, epoch=epoch, display_step=show, batch_size=batch_size)
    CAE_train.close()
    tf.reset_default_graph()

    if pre_train: # 如果要恢复之前保存的模型的结果，把模型结果存为预训练结果
        model_path = pretrain_model_path
    restore_path = model_path
    logs_path = "./logs_path"


    batch_size2 = Img.shape[0]
    CAE_acc = ConvAE(flag=1,
                    n_input=n_input,
                     n_hidden=n_hidden,
                     m=m,
                     k=k,
                     dictionary=dictionary,
                     reg_constant1=reg1,
                     re_constant2=reg2,
                     re_constant3=reg3,
                     kernel_size=kernel_size,
                     batch_size=batch_size2,
                     model_path=model_path,
                     restore_path=restore_path,
                     logs_path=logs_path)

    acc, NMI, ARI, homo, mf1, RAGI= accurate(Img=Img, Label=Label, CAE=CAE_acc)
    CAE_acc.close()
    print("====================================================================================================")
    print(
        "m = {},k = {}, lr = {}, epoch = {}, batchsize = {}, reg1 = {}, reg2 = {}, reg3 = {}, neighbor_num = {},NMI = {:.4f},ARI = {:.4f}, homo = {:.4f}, RAGI = {:.4f}".format(
            m,k, lr, epoch, batch_size,reg1, reg2, reg3, cnt_neighbor, NMI, ARI, homo, RAGI))
    acc_row = {}
    acc_row['m'], acc_row['k'], acc_row['lr'], acc_row['epoch'], acc_row['batch_size'], acc_row['reg1'], acc_row[
        'reg2'] , acc_row['reg3']= m, k, lr, epoch, batch_size, reg1, reg2, reg3
    acc_row['acc'] = acc
    acc_row['train_model_path'] = model_path
    acc_row['pretrain_model_path'] = pretrain_model_path
    acc_row['neighbor_num'] = cnt_neighbor
    acc_row["NMI"] = round(NMI, 4)
    acc_row["ARI"] = round(ARI, 4)
    acc_row["homo"] = round(homo, 4)
    acc_row["RAGI"] = round(RAGI, 4)
    acc_result.append(acc_row)


# 加载图像，得到图像tensor和对应标签

name = "pbmc_5k300"

ATAC = np.load('./DataSets/bavaria_data/' + name  + '.npy')

Label = np.load('./Datasets/bavaria_data/label_pbmc_5k.npy')
Label = Label.reshape(1,-1)
Img = ATAC
Img = np.expand_dims(Img[:], 2)

pretrain_model_path =  './pretrain-model/' + name +'.ckpt' # 读入预训练文件路径

# z = np.load("./latent_C/best_latent.npy")

metadata = pd.read_csv('./DataSets/bavaria_data/metadata.tsv', sep='\t', index_col=0)

num = Label.max()
# face image clustering
n_input = [Img.shape[0],Img.shape[1]]

temp = n_input[1]

kernel_size = [1, 1, 1] # 预训练也要改
n_hidden = [10, 20, 30] #把这个加进去
batch_size = 50
latent_dimensions = temp*n_hidden[2]   # 图像经过编码器之后的总维度，不同数据集需要更改这个变量

show = 100

current_time = time.strftime("%Y%m%d-%H%M%S")
dictionary = []
acc_result = []

start_time = time.time()

pre_train = False
if pre_train:
    pretrain_model_path = './train_model/model1661872996.ckpt'

for kk in range(1650,1660):
    run_num = 20
    for i in range(run_num):
        print("########################################################################################################################")
        print("iteration {}".format(i + 1))
        print("########################################################################################################################")
        iter_m = 1400 # 神经网络的全连接编程5000维
        iter_k = kk # 第二个/8
        iter_lr = 1e-5
        cnt_neighbor = 5
        iter_epoch = 200 # 100-0.46  200-0,4950  300--0.4700
        iter_bs = 100 # batch——size,1--36.5%,15--41.5%,10--40%,20--39.5%,40--39.5%,80--40%
        iter_reg1 = 1
        iter_reg2 = 5
        iter_reg3 = 1
        acc_run(m=iter_m, k=iter_k, lr=iter_lr, epoch=iter_epoch, batch_size=iter_bs, reg1=iter_reg1, reg2=iter_reg2,
                reg3=iter_reg3)

        # 将run_num个实验结果写入excel
        if not os.path.exists("./acc_result_" + name):
            os.mkdir("./acc_result_" + name)
        acc_dir = "./acc_result_" + name  # create accurate path to save result excel
        print("==================================================")
        print("begin write to excel", end="")
        col_names = ['m',"k", "lr", "epoch", "batch_size", "reg1", "reg2", "reg3", "acc", "NMI","ARI","homo", "RAGI", "neighbor_num","train_model_path", "pretrain_model_path"]
        df = pd.DataFrame(acc_result, columns=col_names)
        writer = pd.ExcelWriter(acc_dir + '/acc_result' + current_time + '.xlsx', engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1', index=False)  # write acc to excel
        writer.save()
        print(", done!!!")