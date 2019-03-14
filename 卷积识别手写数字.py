# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:57:55 2019

@author: 汤国频
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#定义初始化权重的函数
def weight_variavles(shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))   
    return w

#定义一个初始化偏置的函数
def bias_variavles(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))   
    return b 


def model():
    
    #1.建立数据的占位符 x [None, 784]  y_true [None, 10]
    with tf.variable_scope("date"):
        x = tf.placeholder(tf.float32, [None, 784])
        
        y_true = tf.placeholder(tf.int32, [None, 10])
        
        
    
    #2.卷积层1  卷积:5*5*1,32个filter,strides= 1-激活-池化
    with tf.variable_scope("conv1"):
        #随机初始化权重
        w_conv1 = weight_variavles([5, 5, 1, 32])
        b_conv1 = bias_variavles([32])
        
        #对x进行形状的改变[None, 784] ----- [None,28,28,1]
        x_reshape = tf.reshape(x,[-1, 28, 28, 1])  #不能填None,不知道就填-1
        
        # [None,28, 28, 1] -------- [None, 28, 28, 32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding = "SAME") + b_conv1)
        
        #池化 2*2，步长为2，【None, 28,28, 32]--------[None,14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1],strides = [1,2,2,1],padding = "SAME")
        
    #3.卷积层2  卷积:5*5*32,64个filter,strides= 1-激活-池化
    with tf.variable_scope("conv2"):
        #随机初始化权重和偏置
        w_conv2 = weight_variavles([5, 5, 32, 64])
        b_conv2 = bias_variavles([64])
        
        #卷积、激活、池化
        #[None,14, 14, 32]----------【NOne, 14, 14, 64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2,strides=[1, 1, 1, 1], padding = "SAME") + b_conv2)
        
        #池化 2*2，步长为2 【None, 14,14，64]--------[None,7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1],strides = [1,2,2,1],padding = "SAME")
    
    #4.全连接层 [None,7, 7, 64] --------- [None, 7*7*64] * [7*7*64, 10]+[10] = [none, 10]
    with tf.variable_scope("fc"):
        #随机初始化权重和偏置:
        w_fc = weight_variavles([7*7*64, 10])
        b_fc = bias_variavles([10])
        
        #修改形状 [none, 7, 7, 64] ----------[None, 7*7*64]
        x_fc_reshape = tf.reshape(x_pool2,[-1,7 * 7 * 64])
        
        #进行矩阵运算得出每个样本的10个结果[NONE, 10]
        y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc
    
    return x, y_true, y_predict


def conv_fc():
    #获取数据
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    #定义模型，得出输出
    x,y_true,y_predict = model()
    
    #进行交叉熵损失计算
    #3.计算交叉熵损失    
    with tf.variable_scope("soft_cross"): 
        
        #求平均交叉熵损失,tf.reduce_mean对列表求平均值
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)) #返回损失值的列表
       
    #4.梯度下降求出最小损失,注意在深度学习中，或者网络层次比较复杂的情况下，学习率通常不能太高    
    with tf.variable_scope("optimizer"):
        
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    #5.计算准确率   
    with tf.variable_scope("acc"):
        
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        #equal_list None个样本 类型为列表1为预测正确，0为预测错误[1, 0, 1, 0......]
        
        accuray = tf.reduce_mean(tf.cast(equal_list, tf.float32))
        
    init_op = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    #开启会话运行
    with tf.Session() as sess:
        sess.run(init_op)
        #循环去训练
        for i in range(2000):
            #取出真实存在的特征值和目标值
            mnist_x, mnist_y = mnist.train.next_batch(50)
            
            #运行train_op训练
            sess.run(train_op, feed_dict = {x: mnist_x, y_true: mnist_y})
                  

            #print(sess.run(tf.argmax(y_predict[:,0], 1), feed_dict = {x: mnist_x, y_true: mnist_y}))
            #print(sess.run(tf.argmax(mnist_y[0], 0)))
            
            
            print("训练第%d步，准确率为：%f" % (i, sess.run(accuray,feed_dict = {x: mnist_x, y_true: mnist_y})))
            
        saver.save(sess, "D:/Dict/a/fcc_model.ckpt")
    
    return None


if __name__ == "__main__":
    conv_fc()
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    