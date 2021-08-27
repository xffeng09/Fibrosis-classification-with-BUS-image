# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:28:33 2020

@author: Think
"""

import vgg16
import os
from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix,accuracy_score
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import cv2
import numpy as np
import tensorflow as tf
from sklearn import metrics
from PIL import Image
batch_size = 128


def read_and_decode(filename):  #解码tfrecord文件
#    print("111111111111111111111")
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                   features={
                                           'label' : tf.FixedLenFeature([], tf.int64),  
                                           'img_raw' : tf.FixedLenFeature([], tf.string)
                                       
                                   })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [224,224])
    label = tf.cast(features['label'], tf.int32)
   
    return image,label

#label = tf.cast(features['label'], tf.int32)
    
#make tfrecord dataset
def CreateBatch(list_image,batchsize): #产生shuffle batch 图像用于feed 到网络中    

  
    image_batch,label_batch = tf.train.shuffle_batch(list_image,   
                                                        batch_size=batchsize,   
                                                        capacity=20000,   
                                                        min_after_dequeue=10000  
                                                        )  
  
    label_batch = tf.one_hot(label_batch,depth=5)  
    return image_batch,label_batch 
def CreateBatch_list(list_image,batchsize):
    
    image_batch,label_batch1 =tf.train.batch(list_image,batch_size=batchsize,
                                        capacity=8000,
                                        num_threads=4)
    label_batch = tf.one_hot(label_batch1,depth=5)  
    return image_batch,label_batch,label_batch1 
def avg_pool_liver(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def init_w( shape):

    w = tf.get_variable(name='weigth', shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    return w

def init_b(shape):
		
#    return tf.Variable(initial_value=tf.random_normal(shape=shape, dtype=tf.float32), name='bias')
    return tf.get_variable(name='bias', shape=shape, initializer=tf.zeros_initializer())
def weight_scale1(name):
    init = tf.constant(1/3, shape = [batch_size, 28,28,256],name = name)
    return init
def weight_scale2(name):
    init2 = tf.constant(1/2, shape = [batch_size, 14,14,512],name = name)
    return init2    
        
def conv_layer_liver( bottom, shape, train_flag, name):
        with tf.variable_scope(name ):
            filt = init_w(shape)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = init_b([shape[3]])
            bias = tf.nn.bias_add(conv, conv_biases)
            bias_BN = tf.layers.batch_normalization(bias, training=train_flag, name = 'conv_bn')
            relu = tf.nn.relu(bias_BN)
            return relu
        
        
def mean_filter_w(shape,name):
    init = tf.constant(1/9, shape = shape,name = name)
    return  init
def mean_attention(bottom,shape,name):
    w_filter = mean_filter_w(shape,name)
    filter_outp = tf.nn.conv2d(bottom, w_filter, [1, 1, 1, 1], padding='SAME')
    return filter_outp
    
    

def up_sample(bottom, shape, output_shape,train_flag,name):
    with tf.variable_scope(name):
            filt = init_w(shape)

            conv =tf.nn.conv2d_transpose(
				value=bottom, filter=filt,
				output_shape=output_shape,
				strides=[1, 2, 2, 1], padding='SAME', name='Up_Sample')

            conv_biases = init_b([shape[2]])
            bias = tf.nn.bias_add(conv, conv_biases)
            bias_BN = tf.layers.batch_normalization(bias, training=train_flag, name = 'up_conv_bn')

            relu = tf.nn.relu(bias_BN)
            return relu
def max_pool_liver( bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)    

with tf.name_scope("add_vgg_fcn"):
    images1 = tf.placeholder('float',[batch_size, 224,224,3])
    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images1)
#    f_feature = tf.placeholder(tf.float32,shape=[None, 224,224,3],name = 'F_input')
    y_ = tf.placeholder('float', shape=[None, 5], name='y_')
    keep_f = tf.placeholder('float', name = 'rate')
    is_training = tf.placeholder(tf.bool)
    
    f_feature = vgg.pool3
    
    scale_1 = avg_pool_liver(f_feature, name = 'scale1')    
    cov_scale1 = conv_layer_liver(scale_1, [3,3,256,256], is_training,name = 'cov_scale1')
    cov_scale1_1 = conv_layer_liver(cov_scale1, [3,3,256,256],is_training, name = 'cov_scale1_1')
    cov_scale11 = tf.multiply(tf.nn.softmax(mean_attention(cov_scale1_1,[3,3,256,256], name = 'attention_scale11')), cov_scale1_1) + cov_scale1_1
    cov_scale1_up = up_sample(cov_scale11, [2,2,256,256],  [batch_size, 28,28,256],is_training, name = 'cov_scale1_up')
    
    
    scale_2 = avg_pool_liver(scale_1, name = 'scale2')
    cov_scale2 = conv_layer_liver(scale_2, [3,3,256,256], is_training, name = 'cov_scale2')
    cov_scale2_2 = conv_layer_liver(cov_scale2, [3,3,256,256], is_training, name = 'cov_scale2_2')
    
    cov_scale22 = tf.multiply(tf.nn.softmax(mean_attention(cov_scale2_2,[3,3,256,256], name = 'attention_scale22')), cov_scale2_2) + cov_scale2_2
    cov_scale2_up1 = up_sample(cov_scale22, [2,2,256,256],  [batch_size, 14,14,256],is_training, name = 'cov_scale2_up1')
    cov_scale2_up2 = up_sample(cov_scale2_up1, [2,2,256,256],  [batch_size, 28,28,256], is_training,name = 'cov_scale2_up2')

    
    scale_0_0 = conv_layer_liver(f_feature, [3,3,256,256],is_training, name = 'scale_0_0' )
    scale_00 = conv_layer_liver(scale_0_0, [3,3,256,256],is_training, name = 'scale_00')
    scale_0 = tf.multiply(tf.nn.softmax(mean_attention(scale_00,[3,3,256,256], name = 'attention_scale0')), scale_00) + scale_00
    
    fusion_1 = scale_0 + cov_scale1_up + cov_scale2_up2 
    
    w_1 = weight_scale1(name = 'fusion_w1')
    mult_fusion_w1 = tf.multiply(fusion_1, w_1)
    
    
    cov_4 = conv_layer_liver(mult_fusion_w1, [3,3,256,512],is_training, name = 'cov4' )
    cov_4_4 = conv_layer_liver(cov_4, [3,3,512,512],is_training, name = 'cov4_4')
    pool_4 = max_pool_liver(cov_4_4, name = 'pool4')
    

    scale_3 = avg_pool_liver(pool_4, name = 'scale3')   #7*7 
    cov_scale3 = conv_layer_liver(scale_3, [3,3,512,512], is_training,name = 'cov_scale3')
    cov_scale3_3 = conv_layer_liver(cov_scale3, [3,3,512,512],is_training, name = 'cov_scale3_3')
    cov_scale33 = tf.multiply(tf.nn.softmax(mean_attention(cov_scale3_3,[3,3,512,512], name='attention_scale33' )), cov_scale3_3) + cov_scale3_3
    cov_scale3_up = up_sample(cov_scale33, [2,2,512,512],  [batch_size, 14,14,512], is_training,name = 'cov_scale3_up')
    

    scale_11_1 = conv_layer_liver(pool_4, [3,3,512,512], is_training,name = 'scale_11_1')
    scale_11 = conv_layer_liver(scale_11_1, [3,3,512,512], is_training,name = 'scale_11')
    scale_1 = tf.multiply(tf.nn.softmax(mean_attention(scale_11, [3,3,512,512], name = 'attention_scale1')), scale_11) + scale_11
    fusion_2 = scale_1 + cov_scale3_up
    
    
    w_2 = weight_scale2(name = 'fusion_w2')
    mult_fusion_w2 = tf.multiply(fusion_2, w_2)
    
    cov_5 = conv_layer_liver(mult_fusion_w2, [3,3,512,512],is_training, name = 'cov5')
    cov_5_5 = conv_layer_liver(cov_5, [3,3,512,512],is_training, name = 'cov5_5')
    pool_5 = max_pool_liver(cov_5_5, name = 'pool5')
    pooling_feature = tf.reduce_mean(pool_5,axis=[1,2], name = 'pooling_feature')
    
    

    fc3_d = tf.nn.dropout(pooling_feature, keep_f,name = 'fc36')
    fc4 = tf.contrib.layers.fully_connected(fc3_d, 5,activation_fn=None)
#    fc3_d = tf.nn.dropout(fc3, 0.5)
    
    soft_fc = tf.nn.softmax(fc4)

    loss_pro = soft_fc[:, 0]
    grads_active_map = tf.convert_to_tensor(tf.gradients(loss_pro, pool_5))
    caset_conv_output = tf.cast(pool_5 > 0, tf.float32)
    cast_grads = tf.cast(grads_active_map > 0, tf.float32)
    guided_grads = caset_conv_output * cast_grads * cast_grads

    conv_outputs = pool_5[0]
    guided_grads = guided_grads[0]

    weights_grad = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights_grad, conv_outputs), axis=-1)



    predictions = tf.argmax(fc4, 1)
    cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=fc4)
    
    cross_entropy = tf.reduce_mean(cross_entropy1, name = 'loss')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):            
        train_step =tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(fc4), 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
    
        
def tain_vgg():
    train_file_path = '../liver_train_fivefold_1.tfrecords'
    CHECK_POINT_PATH = './proposed_model_fivefold_1/FCNnet_model.ckpt'


    train_batch = 128

    train_image, train_label = read_and_decode(train_file_path)
    train_images, train_labels = CreateBatch([ train_image, train_label],train_batch)


    model_saver = tf.train.Saver()
    def get_VGG_input( Vgg_input_image,Ultrasound_batch_size):
		
         u_image = np.resize(Vgg_input_image,(Ultrasound_batch_size, 224,224,1))
         U_batch = np.concatenate((u_image, u_image,u_image), 3)  

         U_batch = (U_batch - np.min(U_batch))/(np.max(U_batch) - np.min(U_batch))
         
         return U_batch
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        
        for epoch in range(2):#30-35-40 -------40
            for i in range(3000):
                img, label = sess.run([train_images, train_labels])
                batch_img = get_VGG_input(img, train_batch)

                if i% 100 == 0:
                    los, acc = sess.run([cross_entropy, accuracy],
                                        feed_dict={y_: label, images1: batch_img, keep_f: 1.0, is_training: False})
                    print('epoch: % d, i: % d, loss: %.6f and accuracy: %.6f' %(epoch, i, los, acc))
                
                sess.run([train_step], feed_dict={images1: batch_img, y_ : label, keep_f: 0.5, is_training: True})
                
            model_saver.save(sess=sess, save_path = CHECK_POINT_PATH)
        coord.request_stop()
        coord.join(threads) 
    print("Done training")
                

