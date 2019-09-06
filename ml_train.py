import numpy as np
import math
import os
import tensorflow as tf
import time
import glob
import random
import numexpr as ne
from scipy.special import sph_harm
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
#import tensorflow_probability as tfp
#from compact_bilinear_pooling import compact_bilinear_pooling_layer
from tensorflow.python.framework import graph_util    
import logging
print(tf.__version__)
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 

raw_points_init = tf.placeholder(tf.float32, shape=[ None, 3], name="raw_points")

centered_points = tf.subtract(raw_points_init, tf.reduce_mean(raw_points_init, axis = 0, keepdims = True))

centered_points_expanded = tf.expand_dims(centered_points, 0,
		                                     name="cn_caps1_output_expanded")

adjoint_mat = tf.matmul(tf.transpose(centered_points_expanded, [0,2,1]), centered_points_expanded)

e,ev = tf.self_adjoint_eig(adjoint_mat, name="eigendata")

logger = logging.getLogger()    # initialize logging class
#logger.setLevel(logging.DEBUG)  # default log level
format_ = logging.Formatter("%(asctime)s - %(message)s")    # output format 
sh = logging.StreamHandler(stream=sys.stdout)    # output to standard output
sh.setFormatter(format_)
logger.addHandler(sh)

normal_vec = ev[:,:,0]
normalized_normal_vec = tf.nn.l2_normalize(normal_vec, axis = 1)


rot_theta = tf.acos(tf.matmul(normalized_normal_vec, tf.transpose(tf.constant([[0.0,0.0,1.0]]),[1,0])))

b_vec = tf.nn.l2_normalize(tf.cross(tf.constant([[0.0,0.0,1.0]]), normalized_normal_vec), axis = 1)

q0 = tf.cos(rot_theta/2.0)
q1 = tf.sin(rot_theta/2.0) * b_vec[0,0]
q2 = tf.sin(rot_theta/2.0) * b_vec[0,1]
q3 = tf.sin(rot_theta/2.0) * b_vec[0,2]

el_0_0 = tf.square(q0) + tf.square(q1) - tf.square(q2) - tf.square(q3)
el_0_1 = 2*(q1*q2-q0*q3)
el_0_2 = 2*(q1*q3+q0*q2)
el_1_0 = 2*(q1*q2+q0*q3)
el_1_1 = tf.square(q0) - tf.square(q1) + tf.square(q2) - tf.square(q3)
el_1_2 = 2*(q2*q3+q0*q1)
el_2_0 = 2*(q1*q3-q0*q2)
el_2_1 = 2*(q2*q3+q0*q1)
el_2_2 = tf.square(q0) - tf.square(q1) - tf.square(q2) + tf.square(q3)

Q = tf.concat([tf.concat([el_0_0,el_0_1,el_0_2], axis = 1), tf.concat([el_1_0,el_1_1,el_1_2], axis = 1), tf.concat([el_2_0,el_2_1,el_2_2], axis = 1)], axis=0)

u_ = tf.matmul(Q,tf.transpose(tf.constant([[1.0,0.0,0.0]]), [1,0]))
v_ = tf.matmul(Q,tf.transpose(tf.constant([[0.0,1.0,0.0]]), [1,0]))
w_ = tf.matmul(Q,tf.transpose(tf.constant([[0.0,0.0,1.0]]), [1,0]))

transform_mat = tf.concat([u_,v_,w_], axis = 1)
    
#transformed_coordinates_ = tf.matmul(centered_points,transform_mat)  
transformed_coordinates_ = centered_points

transformed_coordinates__  = tf.subtract(transformed_coordinates_, tf.stack([tf.reduce_mean(transformed_coordinates_[:,0], keep_dims = True), tf.reduce_mean(transformed_coordinates_[:,1], keep_dims = True), tf.constant([0.0]) ], axis = 1))


a = tf.matmul(tf.transpose(tf.slice(transformed_coordinates__,[0,0], [-1,1]), [1,0]),tf.slice(transformed_coordinates__, [0,1],[-1,1]  )) / tf.matmul(tf.transpose(tf.slice(transformed_coordinates__, [0,0], [-1,1]), [1,0]),tf.slice(transformed_coordinates__, [0,0], [-1,1]))

angle = tf.atan2(a,1)

print("adasdas")
print(angle)


rot_z = tf.concat([[[tf.cos(angle[0,0]), tf.sin(angle[0,0]), 0.0]], [[-tf.sin(angle[0,0]), tf.cos(angle[0,0]), 0.0]], [[0.0, 0.0, 1.0]]], axis = 0)


#transformed_coordinates___ = tf.matmul(transformed_coordinates__, rot_z)
transformed_coordinates___ = centered_points
b = tf.add(tf.slice(transformed_coordinates___, [0, 0], [-1, 1]) * 100000, tf.slice(transformed_coordinates___, [0, 1], [-1, 1]))

reordered = tf.gather(transformed_coordinates___, tf.nn.top_k(b[:, 0], k=tf.shape(transformed_coordinates___)[0], sorted=True).indices)
transformed_coordinates = tf.reverse(reordered, axis=[0])
#transformed_coordinates = centered_points

print(rot_z)

#transformed_coordinates = transformed_coordinates_ 





mask = tf.greater(transformed_coordinates[:,2],0)

points_from_side_one = tf.boolean_mask(transformed_coordinates, mask) 

mask2 = tf.less(transformed_coordinates[:,2],0)

points_from_side_two = tf.boolean_mask(transformed_coordinates, mask2) 


indices_one_x = tf.nn.top_k(points_from_side_one[:,0], k=tf.shape(points_from_side_one)[0]).indices
reordered_points_one_x = tf.gather(points_from_side_one, indices_one_x, axis=0)

indices_two_x = tf.nn.top_k(points_from_side_two[:, 0], k=tf.shape(points_from_side_two)[0]).indices
reordered_points_two_x = tf.gather(points_from_side_two, indices_two_x, axis=0)


indices_one_y = tf.nn.top_k(points_from_side_one[:,1], k=tf.shape(points_from_side_one)[0]).indices
reordered_points_one_y = tf.gather(points_from_side_one, indices_one_y, axis=0)

indices_two_y = tf.nn.top_k(points_from_side_two[:, 1], k=tf.shape(points_from_side_two)[0]).indices
reordered_points_two_y = tf.gather(points_from_side_two, indices_two_y, axis=0)


#b = tf.add(tf.slice(transformed_coordinates, [0, 0], [-1, 1]) * 100000, tf.slice(transformed_coordinates, [0, 1], [-1, 1]))

#reordered = tf.gather(transformed_coordinates, tf.nn.top_k(b[:, 0], k=tf.shape(transformed_coordinates)[0], sorted=True).indices)
#reordered_s = tf.reverse(reordered, axis=[0])


input1_1_x = tf.expand_dims([reordered_points_one_x[:,2]],2)


filter1_1_x = tf.get_variable("a_1", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0.1))#, trainable=False)

output1_1_x = tf.nn.conv1d(input1_1_x, filter1_1_x, stride=2, padding="SAME")

filter2_1_x = tf.get_variable("a_2", [3, 10, 20], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output2_1_x_temp = tf.nn.conv1d(output1_1_x, filter2_1_x, stride=2, padding="SAME")

output2_1_x = tf.cond(tf.shape(output2_1_x_temp)[1] >= 100, lambda: tf.slice(output2_1_x_temp, [0,0,0], [-1,100,-1]), lambda: tf.concat([output2_1_x_temp, tf.zeros([1,100-tf.shape(output2_1_x_temp)[1],20])], axis = 1))


input1_2_x = tf.expand_dims([reordered_points_two_x[:,2]],2)


filter1_2_x = tf.get_variable("a_3", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output1_2_x = tf.nn.conv1d(input1_2_x, filter1_2_x, stride=2, padding="SAME")

filter2_2_x = tf.get_variable("a_4", [3, 10, 20], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output2_2_x_temp= tf.nn.conv1d(output1_2_x, filter2_2_x, stride=2, padding="SAME")

output2_2_x = tf.cond(tf.shape(output2_2_x_temp)[1] >= 100, lambda: tf.slice(output2_2_x_temp, [0,0,0], [-1,100,-1]), lambda: tf.concat([output2_2_x_temp, tf.zeros([1,100-tf.shape(output2_2_x_temp)[1],20])], axis = 1))



input1_1_y = tf.expand_dims([reordered_points_one_y[:,2]],2)


filter1_1_y = tf.get_variable("a_5", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output1_1_y = tf.nn.conv1d(input1_1_y, filter1_1_y, stride=2, padding="SAME")

filter2_1_y = tf.get_variable("a_6", [3, 10, 20],initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output2_1_y_temp = tf.nn.conv1d(output1_1_y, filter2_1_y, stride=2, padding="SAME")

output2_1_y = tf.cond(tf.shape(output2_1_y_temp)[1] >= 100, lambda: tf.slice(output2_1_y_temp, [0,0,0], [-1,100,-1]), lambda: tf.concat([output2_1_y_temp, tf.zeros([1,100-tf.shape(output2_1_y_temp)[1],20])], axis = 1))




input1_2_y = tf.expand_dims([reordered_points_two_y[:,2]],2)


filter1_2_y = tf.get_variable("a_7", [6, 1, 10], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output1_2_y = tf.nn.conv1d(input1_2_y, filter1_2_y, stride=2, padding="SAME")

filter2_2_y = tf.get_variable("a_8", [3, 10, 20], initializer=tf.random_normal_initializer(seed=0.1))#,trainable=False)

output2_2_y_temp = tf.nn.conv1d(output1_2_y, filter2_2_y, stride=2, padding="SAME")

output2_2_y = tf.cond(tf.shape(output2_2_y_temp)[1] >= 100, lambda: tf.slice(output2_2_y_temp, [0,0,0], [-1,100,-1]), lambda: tf.concat([output2_2_y_temp, tf.zeros([1,100-tf.shape(output2_2_y_temp)[1],20])], axis = 1))


#side_1_descriptor = tf.matmul(tf.transpose(output2_1_x, [0,2,1]), output2_1_y)
#side_2_descriptor = tf.matmul(tf.transpose(output2_2_x, [0,2,1]), output2_2_y)

#print(side_1_descriptor)

# concat_layer = tf.reshape(tf.concat([output2_1_x, output2_2_x, output2_1_y, output2_2_y], axis = 1), [1, 1600])
concat_layer = tf.reshape(tf.concat([output2_1_x ,output2_2_x,output2_1_y,output2_2_y ], axis = 0), [1, 8000])


rot_angles_temp_ = tf.layers.dense(concat_layer,3, trainable=True, name = "a_9")
rot_angles = tf.constant([[0.0, 0.0, 0.0]])

_rot_angles = tf.nn.l2_normalize(rot_angles_temp_) 

#rot_angles_ = tf.reshape(rot_angles, [3,3,3])

# rotation_matrix_one = tf.squeeze(tf.slice(rot_angles_, [0,0,0], [1,-1,-1]),squeeze_dims=[0])
# rotation_matrix_two =  tf.squeeze(tf.slice(rot_angles_, [1,0,0], [1,-1,-1]),squeeze_dims=[0])
# rotation_matrix_three =  tf.squeeze(tf.slice(rot_angles_, [2,0,0], [1,-1,-1]),squeeze_dims=[0])

#rot_angles = tf.constant([[22.0/28.0,22.0/14.0,0.0]]) 

rotation_matrix_one = tf.concat([tf.constant([[1.0, 0.0, 0.0]]), [[0.0, tf.cos(rot_angles[0,0]), -tf.sin(rot_angles[0,0])]], [[0.0, tf.sin(rot_angles[0,0]), tf.cos(rot_angles[0,0])]]], axis = 0)
rotation_matrix_two = tf.concat([[[tf.cos(rot_angles[0,1]), 0.0, tf.sin(rot_angles[0,1])]], [[0.0, 1.0, 0.0]], [[-tf.sin(rot_angles[0,1]), 0.0,tf.cos(rot_angles[0,1]) ]]], axis = 0)
rotation_matrix_three = tf.concat([[[tf.cos(rot_angles[0,2]), -tf.sin(rot_angles[0,2]),0.0 ]], [[tf.sin(rot_angles[0,2]), tf.cos(rot_angles[0,2]), 0.0]], [[0.0, 0.0,1.0 ]]], axis = 0)

print(rotation_matrix_one)

centered_points_expanded_ = tf.reshape(transformed_coordinates, [-1, 3])
point_count = tf.shape(centered_points_expanded_)[0]

#rotation_matrix_one = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_one")
trasformed_points_one = tf.matmul(centered_points_expanded_, rotation_matrix_one, name="trans_point_one")
trasformed_points_one_reshaped = tf.reshape(trasformed_points_one, [-1, point_count, 3], name = "trans_point_one_reshape")

#rotation_matrix_two = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_two")
trasformed_points_two = tf.matmul(centered_points_expanded_, rotation_matrix_two, name="trans_point_two")
trasformed_points_two_reshaped = tf.reshape(trasformed_points_two, [-1, point_count, 3], name = "trans_point_two_reshape")

#rotation_matrix_three = tf.placeholder(tf.float32, shape=[3, 3], name="rot_mat_three")
trasformed_points_three = tf.matmul(centered_points_expanded_, rotation_matrix_three, name="trans_point_three")
trasformed_points_three_reshaped = tf.reshape(trasformed_points_three, [-1, point_count, 3], name = "trans_point_three_reshape")

########################################################################################

trasformed_points_one_reshaped_ = tf.reshape(trasformed_points_one_reshaped, [-1, 3])

#point_distance_one = tf.reduce_sum(tf.square(trasformed_points_one_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

#scale_metric_one = tf.exp(-point_distance_one*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

#scale_metric_tiled_one = tf.tile(scale_metric_one, [1, 3], name="cn_W_tiled")

calibrated_points_one = trasformed_points_one_reshaped_ 








trasformed_points_two_reshaped_ = tf.reshape(trasformed_points_two_reshaped, [-1, 3])

#point_distance_two = tf.reduce_sum(tf.square(trasformed_points_two_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

#scale_metric_two = tf.exp(-point_distance_two*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

#scale_metric_tiled_two = tf.tile(scale_metric_two, [1, 3], name="cn_W_tiled")

calibrated_points_two = trasformed_points_two_reshaped_


trasformed_points_three_reshaped_ = tf.reshape(trasformed_points_three_reshaped, [-1, 3])

#point_distance_three = tf.reduce_sum(tf.square(trasformed_points_three_reshaped_), axis=1, keepdims = True)

#point_distance_one = tf.reduce_sum(trasformed_points_one_reshaped_, axis=1, keepdims = True)

#scale_metric_three = tf.exp(-point_distance_three*0.0000001)

#scale_metric_one = tf.multiply(point_distance_one,0.01)

#scale_metric_tiled_three = tf.tile(scale_metric_three, [1, 3], name="cn_W_tiled")

calibrated_points_three = trasformed_points_three_reshaped_

calibrated_points_one_corrected_shape = tf.reshape(calibrated_points_one, [-1, point_count, 3])


centered_calib_points_one_temp_  = tf.subtract(calibrated_points_one,tf.reduce_mean(calibrated_points_one,axis=0,keep_dims=True))
centered_calib_points_two_temp_ = tf.subtract(calibrated_points_two,tf.reduce_mean(calibrated_points_two,axis=0,keep_dims=True))
centered_calib_points_three_temp_  = tf.subtract(calibrated_points_three,tf.reduce_mean(calibrated_points_three,axis=0,keep_dims=True))



b1 = tf.add(tf.slice(centered_calib_points_one_temp_ , [0, 0], [-1, 1]) * 100000, tf.slice(centered_calib_points_one_temp_ , [0, 1], [-1, 1]))

reordered1 = tf.gather(centered_calib_points_one_temp_ , tf.nn.top_k(b1[:, 0], k=tf.shape(centered_calib_points_one_temp_ )[0], sorted=True).indices)
centered_calib_points_one_temp = tf.reverse(reordered1, axis=[0])

b2 = tf.add(tf.slice(centered_calib_points_two_temp_ , [0, 0], [-1, 1]) * 100000, tf.slice(centered_calib_points_two_temp_ , [0, 1], [-1, 1]))

reordered2 = tf.gather(centered_calib_points_two_temp_ , tf.nn.top_k(b2[:, 0], k=tf.shape(centered_calib_points_two_temp_ )[0], sorted=True).indices)
centered_calib_points_two_temp = tf.reverse(reordered2, axis=[0])


b3 = tf.add(tf.slice(centered_calib_points_three_temp_ , [0, 0], [-1, 1]) * 100000, tf.slice(centered_calib_points_three_temp_ , [0, 1], [-1, 1]))

reordered3 = tf.gather(centered_calib_points_three_temp_ , tf.nn.top_k(b3[:, 0], k=tf.shape(centered_calib_points_three_temp_ )[0], sorted=True).indices)
centered_calib_points_three_temp = tf.reverse(reordered3, axis=[0])







#mask_one  = tf.greater(centered_calib_points_one_temp[:,2],0)
indices_one_x_temp = tf.nn.top_k(centered_calib_points_one_temp[:,2], k=tf.shape(centered_calib_points_one_temp)[0]).indices
reordered_points_one_x_temp = tf.gather(centered_calib_points_one_temp, indices_one_x_temp, axis=0)

index_ = tf.shape(reordered_points_one_x_temp)[0]
index = index_ / 2

centered_calib_points_one_t  = tf.slice(reordered_points_one_x_temp, [0, 0], [index, -1])


indices_two_x_temp = tf.nn.top_k(centered_calib_points_two_temp[:,2], k=tf.shape(centered_calib_points_two_temp)[0]).indices
reordered_points_two_x_temp = tf.gather(centered_calib_points_two_temp, indices_two_x_temp, axis=0)


centered_calib_points_two_t  = tf.slice(reordered_points_two_x_temp, [0, 0], [tf.shape(reordered_points_two_x_temp)[0]/2, -1])


indices_three_x_temp = tf.nn.top_k(centered_calib_points_three_temp[:,2], k=tf.shape(centered_calib_points_three_temp)[0]).indices
reordered_points_three_x_temp = tf.gather(centered_calib_points_three_temp, indices_three_x_temp, axis=0)


centered_calib_points_three_t  = tf.slice(reordered_points_three_x_temp, [0, 0], [tf.shape(reordered_points_three_x_temp)[0]/2, -1])





#indices_one_x_tempi = tf.nn.top_k(centered_calib_points_one_temp[:,2], k=tf.shape(centered_calib_points_one_temp[0]).indices
#reordered_points_one_x_tempi = tf.gather(points_from_side_one_temp, indices_one_x_temp, axis=0)


centered_calib_points_one_t_i  = tf.slice(reordered_points_one_x_temp, [tf.shape(reordered_points_one_x_temp)[0]/2 , 0], [tf.shape(reordered_points_one_x_temp)[0]/2, -1])


centered_calib_points_two_t_i  = tf.slice(reordered_points_two_x_temp, [tf.shape(reordered_points_two_x_temp)[0]/2, 0], [tf.shape(reordered_points_two_x_temp)[0]/2, -1])

centered_calib_points_three_t_i  = tf.slice(reordered_points_three_x_temp, [tf.shape(reordered_points_three_x_temp)[0]/2, 0], [tf.shape(reordered_points_three_x_temp)[0]/2, -1])



####################################################################################################3

indices_one_xx_temp = tf.nn.top_k(centered_calib_points_one_temp[:,0], k=tf.shape(centered_calib_points_one_temp)[0]).indices
reordered_points_one_xx_temp = tf.gather(centered_calib_points_one_temp, indices_one_xx_temp, axis=0)

#mask_onex  = tf.greater(centered_calib_points_one_temp[:,0],0)

#centered_calib_points_one_tx  = tf.boolean_mask(centered_calib_points_one_temp, mask_onex)

centered_calib_points_one_tx  = tf.slice(reordered_points_one_xx_temp, [0, 0], [tf.shape(reordered_points_one_xx_temp)[0]/2, -1])


#mask_twox  = tf.greater(centered_calib_points_two_temp[:,0],0)

indices_two_xx_temp = tf.nn.top_k(centered_calib_points_two_temp[:,0], k=tf.shape(centered_calib_points_two_temp)[0]).indices
reordered_points_two_xx_temp = tf.gather(centered_calib_points_two_temp, indices_two_xx_temp, axis=0)

centered_calib_points_two_tx  = tf.slice(reordered_points_two_xx_temp, [0, 0], [tf.shape(reordered_points_two_xx_temp)[0]/2, -1])

#centered_calib_points_two_tx  = tf.boolean_mask(centered_calib_points_two_temp, mask_twox)

#mask_threex  = tf.greater(centered_calib_points_three_temp[:,0],0)

indices_three_xx_temp = tf.nn.top_k(centered_calib_points_three_temp[:,0], k=tf.shape(centered_calib_points_three_temp)[0]).indices
reordered_points_three_xx_temp = tf.gather(centered_calib_points_three_temp, indices_three_xx_temp, axis=0)

#centered_calib_points_three_tx  = tf.boolean_mask(centered_calib_points_three_temp, mask_threex)
centered_calib_points_three_tx  = tf.slice(reordered_points_three_xx_temp, [0, 0], [tf.shape(reordered_points_three_xx_temp)[0]/2, -1])





centered_calib_points_one_txi  = tf.slice(reordered_points_one_xx_temp, [tf.shape(reordered_points_one_xx_temp)[0]/2 , 0], [tf.shape(reordered_points_one_xx_temp)[0]/2, -1])


centered_calib_points_two_txi  = tf.slice(reordered_points_two_xx_temp, [tf.shape(reordered_points_two_xx_temp)[0]/2, 0], [tf.shape(reordered_points_two_xx_temp)[0]/2, -1])

centered_calib_points_three_txi  = tf.slice(reordered_points_three_xx_temp, [tf.shape(reordered_points_three_xx_temp)[0]/2, 0], [tf.shape(reordered_points_three_xx_temp)[0]/2, -1])

#############################################################################################################################3333


indices_one_y_temp = tf.nn.top_k(centered_calib_points_one_temp[:,1], k=tf.shape(centered_calib_points_one_temp)[0]).indices
reordered_points_one_y_temp = tf.gather(centered_calib_points_one_temp, indices_one_y_temp, axis=0)

centered_calib_points_one_ty  = tf.slice(reordered_points_one_y_temp, [0, 0], [tf.shape(reordered_points_one_y_temp)[0]/2, -1])


#mask_oney  = tf.greater(centered_calib_points_one_temp[:,1],0)

#centered_calib_points_one_ty  = tf.boolean_mask(centered_calib_points_one_temp, mask_oney)
indices_two_y_temp = tf.nn.top_k(centered_calib_points_two_temp[:,1], k=tf.shape(centered_calib_points_two_temp)[0]).indices
reordered_points_two_y_temp = tf.gather(centered_calib_points_two_temp, indices_two_y_temp, axis=0)

centered_calib_points_two_ty  = tf.slice(reordered_points_two_y_temp, [0, 0], [tf.shape(reordered_points_two_y_temp)[0]/2, -1])

#mask_twoy  = tf.greater(centered_calib_points_two_temp[:,1],0)

#centered_calib_points_two_ty  = tf.boolean_mask(centered_calib_points_two_temp, mask_twoy)

#mask_threey  = tf.greater(centered_calib_points_three_temp[:,1],0)

#centered_calib_points_three_ty  = tf.boolean_mask(centered_calib_points_three_temp, mask_threey)
indices_three_y_temp = tf.nn.top_k(centered_calib_points_three_temp[:,1], k=tf.shape(centered_calib_points_three_temp)[0]).indices
reordered_points_three_y_temp = tf.gather(centered_calib_points_three_temp, indices_three_y_temp, axis=0)

centered_calib_points_three_ty  = tf.slice(reordered_points_three_y_temp, [0, 0], [tf.shape(reordered_points_three_y_temp)[0]/2, -1])


#mask_oneyi  = tf.less(centered_calib_points_one_temp[:,1],0)
centered_calib_points_one_tyi  = tf.slice(reordered_points_one_y_temp, [tf.shape(reordered_points_one_y_temp)[0]/2 , 0], [tf.shape(reordered_points_one_y_temp)[0]/2, -1])

centered_calib_points_two_tyi  = tf.slice(reordered_points_two_y_temp, [tf.shape(reordered_points_two_y_temp)[0]/2 , 0], [tf.shape(reordered_points_two_y_temp)[0]/2, -1])


centered_calib_points_three_tyi  = tf.slice(reordered_points_three_y_temp, [tf.shape(reordered_points_three_y_temp)[0]/2 , 0], [tf.shape(reordered_points_three_y_temp)[0]/2, -1])










centered_calib_points_one  = tf.cond(tf.shape(centered_calib_points_one_t )[0] >= 1000, lambda: tf.slice(centered_calib_points_one_t , [0,0], [1000,-1]), lambda: tf.concat([centered_calib_points_one_t, tf.zeros([1000-tf.shape(centered_calib_points_one_t)[0],3])], axis = 0))
centered_calib_points_two = tf.cond(tf.shape(centered_calib_points_two_t )[0] >= 1000, lambda: tf.slice(centered_calib_points_two_t , [0,0], [1000,-1]), lambda: tf.concat([centered_calib_points_two_t, tf.zeros([1000-tf.shape(centered_calib_points_two_t)[0],3])], axis = 0))
centered_calib_points_three = tf.cond(tf.shape(centered_calib_points_three_t )[0] >= 1000, lambda: tf.slice(centered_calib_points_three_t , [0,0], [1000,-1]), lambda: tf.concat([centered_calib_points_three_t, tf.zeros([1000-tf.shape(centered_calib_points_three_t)[0],3])], axis = 0))


dist_12 = -1.0 * tf.log(tf.reduce_sum(tf.sqrt(tf.multiply(centered_calib_points_one[:,2], centered_calib_points_two[:,2])+0.0001)))
dist_13 = -1.0 * tf.log(tf.reduce_sum(tf.sqrt(tf.multiply(centered_calib_points_one[:,2], centered_calib_points_three[:,2])+0.0001)))
dist_23 = -1.0 * tf.log(tf.reduce_sum(tf.sqrt(tf.multiply(centered_calib_points_three[:,2], centered_calib_points_two[:,2])+ 0.0001)))

_, var_1 = tf.nn.moments(centered_calib_points_one[:,2], axes = [0])
_, var_2 = tf.nn.moments(centered_calib_points_two[:,2], axes = [0])
_, var_3 = tf.nn.moments(centered_calib_points_three[:,2], axes = [0])



sim_term = -1.0 * (dist_12 + dist_13  + dist_23)/3
info_term = -1.0 * (tf.sqrt(var_1) + tf.sqrt(var_2) + tf.sqrt(var_3))/3

rot_loss = 0.01 *  sim_term + 0.03 * info_term

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


def atan2(y, x):
    angle = tf.select(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
    return angle


r_one_temp = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_t), axis=1, keepdims = True))
r_one = tf.divide(r_one_temp,tf.reduce_max(r_one_temp, axis = 0, keep_dims = True))
theta_one = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_t[:,2],1), tf.maximum(r_one_temp,0.001)),0.99),-0.99))
phi_one = tf.atan2(tf.expand_dims(centered_calib_points_one_t[:,1],1),tf.expand_dims(centered_calib_points_one_t[:,0],1))


r_two_temp = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_t), axis=1, keepdims = True))
r_two = tf.divide(r_two_temp,tf.reduce_max(r_two_temp, axis = 0, keep_dims = True))

#r_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_two = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_t[:,2],1), tf.maximum(r_two_temp,0.001)),0.99),-0.99))
phi_two = tf.atan2(tf.expand_dims(centered_calib_points_two_t[:,1],1),tf.expand_dims(centered_calib_points_two_t[:,0],1))

r_three_temp = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_t), axis=1, keepdims = True))
r_three = tf.divide(r_three_temp,tf.reduce_max(r_three_temp, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_three = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_t [:,2],1), tf.maximum(r_three_temp,0.001)),0.99),-0.99))
phi_three  = tf.atan2(tf.expand_dims(centered_calib_points_three_t [:,1],1),tf.expand_dims(centered_calib_points_three_t [:,0],1))


rp = tf.concat([r_one, r_two, r_three], axis = 0)
thetap = tf.concat([theta_one, theta_two, theta_three], axis = 0)
phip = tf.concat([phi_one, phi_two, phi_three], axis = 0)


#########################################################################################################################

r_one_temp_i = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_t_i), axis=1, keepdims = True))
r_one_i = tf.divide(r_one_temp_i,tf.reduce_max(r_one_temp_i, axis = 0, keep_dims = True))
theta_one_i = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_t_i[:,2],1), tf.maximum(r_one_temp_i,0.001)),0.99),-0.99))
phi_one_i = tf.atan2(tf.expand_dims(centered_calib_points_one_t_i[:,1],1),tf.expand_dims(centered_calib_points_one_t_i[:,0],1))


r_two_temp_i = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_t_i), axis=1, keepdims = True))
r_two_i = tf.divide(r_two_temp_i,tf.reduce_max(r_two_temp_i, axis = 0, keep_dims = True))

#Cr_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_two_i = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_t_i[:,2],1), tf.maximum(r_two_temp_i,0.001)),0.99),-0.99))
phi_two_i = tf.atan2(tf.expand_dims(centered_calib_points_two_t_i[:,1],1),tf.expand_dims(centered_calib_points_two_t_i[:,0],1))

r_three_temp_i = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_t_i), axis=1, keepdims = True))
r_three_i = tf.divide(r_three_temp_i,tf.reduce_max(r_three_temp_i, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_three_i = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_t_i [:,2],1), tf.maximum(r_three_temp_i,0.001)),0.99),-0.99))
phi_three_i  = tf.atan2(tf.expand_dims(centered_calib_points_three_t_i [:,1],1),tf.expand_dims(centered_calib_points_three_t_i [:,0],1))


r_i = tf.concat([r_one_i, r_two_i, r_three_i], axis = 0)
theta_i = tf.concat([theta_one_i, theta_two_i, theta_three_i], axis = 0)
phi_i = tf.concat([phi_one_i, phi_two_i, phi_three_i], axis = 0)


###############################################################################################################################

r_one_tempx = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_tx), axis=1, keepdims = True))
r_onex = tf.divide(r_one_tempx,tf.reduce_max(r_one_tempx, axis = 0, keep_dims = True))
theta_onex = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_tx[:,2],1), tf.maximum(r_one_tempx,0.001)),0.99),-0.99))
phi_onex = tf.atan2(tf.expand_dims(centered_calib_points_one_tx[:,1],1),tf.expand_dims(centered_calib_points_one_tx[:,0],1))


r_two_tempx = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_tx), axis=1, keepdims = True))
r_twox = tf.divide(r_two_tempx,tf.reduce_max(r_two_tempx, axis = 0, keep_dims = True))

#r_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_twox = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_tx[:,2],1), tf.maximum(r_two_tempx,0.001)),0.99),-0.99))
phi_twox = tf.atan2(tf.expand_dims(centered_calib_points_two_tx[:,1],1),tf.expand_dims(centered_calib_points_two_tx[:,0],1))

r_three_tempx = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_tx), axis=1, keepdims = True))
r_threex = tf.divide(r_three_tempx,tf.reduce_max(r_three_tempx, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_threex = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_tx [:,2],1), tf.maximum(r_three_tempx,0.001)),0.99),-0.99))
phi_threex  = tf.atan2(tf.expand_dims(centered_calib_points_three_tx [:,1],1),tf.expand_dims(centered_calib_points_three_tx [:,0],1))


rx = tf.concat([r_onex, r_twox, r_threex], axis = 0)
thetax = tf.concat([theta_onex, theta_twox, theta_threex], axis = 0)
phix = tf.concat([phi_onex, phi_twox, phi_threex], axis = 0)


#########################################################################################################################################

r_one_tempxi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_txi), axis=1, keepdims = True))
r_onexi = tf.divide(r_one_tempxi,tf.reduce_max(r_one_tempxi, axis = 0, keep_dims = True))
theta_onexi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_txi[:,2],1), tf.maximum(r_one_tempxi,0.001)),0.99),-0.99))
phi_onexi = tf.atan2(tf.expand_dims(centered_calib_points_one_txi[:,1],1),tf.expand_dims(centered_calib_points_one_txi[:,0],1))


r_two_tempxi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_txi), axis=1, keepdims = True))
r_twoxi = tf.divide(r_two_tempxi,tf.reduce_max(r_two_tempxi, axis = 0, keep_dims = True))

#r_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_twoxi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_txi[:,2],1), tf.maximum(r_two_tempxi,0.001)),0.99),-0.99))
phi_twoxi = tf.atan2(tf.expand_dims(centered_calib_points_two_txi[:,1],1),tf.expand_dims(centered_calib_points_two_txi[:,0],1))

r_three_tempxi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_txi), axis=1, keepdims = True))
r_threexi = tf.divide(r_three_tempxi,tf.reduce_max(r_three_tempxi, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_threexi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_txi [:,2],1), tf.maximum(r_three_tempxi,0.001)),0.99),-0.99))
phi_threexi  = tf.atan2(tf.expand_dims(centered_calib_points_three_txi [:,1],1),tf.expand_dims(centered_calib_points_three_txi [:,0],1))


rxi = tf.concat([r_onexi, r_twoxi, r_threexi], axis = 0)
thetaxi = tf.concat([theta_onexi, theta_twoxi, theta_threexi], axis = 0)
phixi = tf.concat([phi_onexi, phi_twoxi, phi_threexi], axis = 0)


#############################################################################################################33


r_one_tempy = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_ty), axis=1, keepdims = True))
r_oney = tf.divide(r_one_tempy,tf.reduce_max(r_one_tempy, axis = 0, keep_dims = True))
theta_oney = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_ty[:,2],1), tf.maximum(r_one_tempy,0.001)),0.99),-0.99))
phi_oney = tf.atan2(tf.expand_dims(centered_calib_points_one_ty[:,1],1),tf.expand_dims(centered_calib_points_one_ty[:,0],1))


r_two_tempy = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_ty), axis=1, keepdims = True))
r_twoy = tf.divide(r_two_tempy,tf.reduce_max(r_two_tempy, axis = 0, keep_dims = True))

#r_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_twoy = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_ty[:,2],1), tf.maximum(r_two_tempy,0.001)),0.99),-0.99))
phi_twoy = tf.atan2(tf.expand_dims(centered_calib_points_two_ty[:,1],1),tf.expand_dims(centered_calib_points_two_ty[:,0],1))

r_three_tempy = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_ty), axis=1, keepdims = True))
r_threey = tf.divide(r_three_tempy,tf.reduce_max(r_three_tempy, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_threey = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_ty [:,2],1), tf.maximum(r_three_tempy,0.001)),0.99),-0.99))
phi_threey  = tf.atan2(tf.expand_dims(centered_calib_points_three_ty [:,1],1),tf.expand_dims(centered_calib_points_three_ty [:,0],1))


ry = tf.concat([r_oney, r_twoy, r_threey], axis = 0)
thetay = tf.concat([theta_oney, theta_twoy, theta_threey], axis = 0)
phiy = tf.concat([phi_oney, phi_twoy, phi_threey], axis = 0)

##########################################################################################################################################333


r_one_tempyi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_one_tyi), axis=1, keepdims = True))
r_oneyi = tf.divide(r_one_tempyi,tf.reduce_max(r_one_tempyi, axis = 0, keep_dims = True))
theta_oneyi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_one_tyi[:,2],1), tf.maximum(r_one_tempyi,0.001)),0.99),-0.99))
phi_oneyi = tf.atan2(tf.expand_dims(centered_calib_points_one_tyi[:,1],1),tf.expand_dims(centered_calib_points_one_tyi[:,0],1))


r_two_tempyi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two_tyi), axis=1, keepdims = True))
r_twoyi = tf.divide(r_two_tempyi,tf.reduce_max(r_two_tempyi, axis = 0, keep_dims = True))

#r_two = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_two), axis=1, keepdims = True))
theta_twoyi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_two_tyi[:,2],1), tf.maximum(r_two_tempyi,0.001)),0.99),-0.99))
phi_twoyi = tf.atan2(tf.expand_dims(centered_calib_points_two_tyi[:,1],1),tf.expand_dims(centered_calib_points_two_tyi[:,0],1))

r_three_tempyi = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three_tyi), axis=1, keepdims = True))
r_threeyi = tf.divide(r_three_tempyi,tf.reduce_max(r_three_tempyi, axis = 0, keep_dims = True))

#r_three = tf.sqrt(tf.reduce_sum(tf.square(centered_calib_points_three), axis=1, keepdims = True))
theta_threeyi = tf.acos(tf.maximum(tf.minimum(tf.divide(tf.expand_dims(centered_calib_points_three_tyi [:,2],1), tf.maximum(r_three_tempyi,0.001)),0.99),-0.99))
phi_threeyi  = tf.atan2(tf.expand_dims(centered_calib_points_three_tyi [:,1],1),tf.expand_dims(centered_calib_points_three_tyi [:,0],1))


ryi = tf.concat([r_oneyi, r_twoyi, r_threeyi], axis = 0)
thetayi = tf.concat([theta_oneyi, theta_twoyi, theta_threeyi], axis = 0)
phiyi = tf.concat([phi_oneyi, phi_twoyi, phi_threeyi], axis = 0)




r = tf.concat([r_one, r_one_i], axis = 0)
theta = tf.concat([theta_one, theta_one_i], axis = 0)
phi = tf.concat([phi_one, phi_one_i], axis = 0)




"""
r = tf.concat([rp, r_i, rx,rxi,ry,ryi], axis = 0)
theta = tf.concat([thetap, theta_i, thetax, thetaxi, thetay, thetayi], axis = 0)
phi = tf.concat([phip, phi_i, phix, phixi, phiy, phiyi], axis = 0)
"""



# polar_coordinates_one = tf.concat([r,theta, phi], axis=1)

print("######################################################")
print(phi)





def spherical_harmonic(m,l):
	return math.pow(-1.0,m) * math.sqrt(((2.0*l + 1.0)*7.0/88.0) * (math.factorial(l-m)*1.0/(math.factorial(l+m)*1.0)))  
	
def radial_poly_(rho, m, n):
	if n == 0 and m == 0:
		return tf.ones(tf.shape(rho))
        if n == 1 and m == 1:
                return rho
	if n == 2 and m == 0:
                return 2.0 * tf.pow(rho,2) - 1
        if n == 2 and m == 2:
                return tf.pow(rho,2)
        if n == 3 and m == 1:
                return 3.0* tf.pow(rho, 3) - 2.0 * rho
        if n == 3 and m == 3:
                return tf.pow(rho,3)
	if n == 4 and m == 0:
		return 6.0 * tf.pow(rho,4) - 6.0 * tf.pow(rho,2) + 1
        if n == 4 and m == 2:
                return 4.0* tf.pow(rho, 3.5) - 3.0 * tf.pow(rho,1.5)
        if n == 4 and m == 4:
                return tf.pow(rho,3.5)
        if n == 5 and m == 1:
                return 10.0* tf.pow(rho, 4.5) - 12.0 * tf.pow(rho, 2.5) + 3.0 * tf.sqrt(rho+0.0001)
        if n == 5 and m == 3:
                return 5.0 * tf.pow(rho, 4.5) - 4.0 * tf.pow(rho, 2.5)
        if n == 5 and m == 5:
                return tf.pow(rho,4.5)
        if n == 6 and m == 2:
                return 10.0* tf.pow(rho, 5.5) - 20.0 * 10.0* tf.pow(rho, 3.5) + 6.0 * tf.pow(rho,1.5)
        if n == 6 and m == 4:
                return 6.0 * tf.pow(rho, 5.5)  - 5.0 * tf.pow(rho,3.5)
        if n == 6 and m == 6:
                return tf.pow(rho, 5.5)


def radial_e(r,n,l):
        k = np.zeros(r.shape)
        for i in range(2,n+1):
                k += (((n-1+1)*(-r))**i)/math.factorial(i)
   #print(k)
        return np.exp((n-1+1)*(-r))

def radial_poly_m(x,n,l):
	Q00 = tf.constant(radial_e(x,0,0),dtype =tf.float32 )
	Q10 = tf.constant(radial_e(x,1,0),dtype =tf.float32 )
	Q11 = tf.constant(radial_e(x,1,1),dtype =tf.float32 )
	Q20 = tf.constant(radial_e(x,2,0),dtype =tf.float32 ) - tf.get_variable("c20111", [1,1,1])
	Q21 = tf.constant(radial_e(x,2,1),dtype =tf.float32 ) - tf.get_variable("c2111", [1,1,1]) * Q11 - tf.get_variable("c2120", [1,1,1]) * Q20
	Q22 = tf.constant(radial_e(x,2,2),dtype =tf.float32 ) - tf.get_variable("c2211", [1,1,1]) * Q11 - tf.get_variable("c2220", [1,1,1]) * Q20 - tf.get_variable("c2221", [1,1,1]) * Q21
	Q30 = tf.constant(radial_e(x,3,0),dtype =tf.float32 ) - tf.get_variable("c3011", [1,1,1]) * Q11 - tf.get_variable("c3020", [1,1,1]) * Q20 - tf.get_variable("c3021", [1,1,1]) * Q21 - tf.get_variable("c3022", [1,1,1]) * Q22
	Q31 = tf.constant(radial_e(x,3,1),dtype =tf.float32 ) - tf.get_variable("c3111", [1,1,1]) * Q11 - tf.get_variable("c3120", [1,1,1]) * Q20 - tf.get_variable("c3121", [1,1,1]) * Q21 - tf.get_variable("c3122", [1,1,1]) * Q22 - tf.get_variable("c3130", [1,1,1]) * Q30
	Q32 = tf.constant(radial_e(x,3,2),dtype =tf.float32 ) - tf.get_variable("c3211", [1,1,1]) * Q11 - tf.get_variable("c3220", [1,1,1]) * Q20 - tf.get_variable("c3221", [1,1,1]) * Q21 - tf.get_variable("c3222", [1,1,1]) * Q22 - tf.get_variable("c3230", [1,1,1]) * Q30 - tf.get_variable("c3231", [1,1,1]) * Q31
	Q33 = tf.constant(radial_e(x,3,3),dtype =tf.float32 ) - tf.get_variable("c3311", [1,1,1]) * Q11 - tf.get_variable("c3320", [1,1,1]) * Q20 - tf.get_variable("c3321", [1,1,1]) * Q21 - tf.get_variable("c3322", [1,1,1]) * Q22 - tf.get_variable("c3330", [1,1,1]) * Q30 - tf.get_variable("c3331", [1,1,1]) * Q31 - tf.get_variable("c3332", [1,1,1]) * Q32
	Q40 = tf.constant(radial_e(x,4,0),dtype =tf.float32 ) - tf.get_variable("c4011", [1,1,1]) * Q11 - tf.get_variable("c4020", [1,1,1]) * Q20 - tf.get_variable("c4021", [1,1,1]) * Q21 - tf.get_variable("c4022", [1,1,1]) * Q22 - tf.get_variable("c4030", [1,1,1]) * Q30 - tf.get_variable("c4031", [1,1,1]) * Q31 - tf.get_variable("c4032", [1,1,1]) * Q32 - tf.get_variable("c4033", [1,1,1]) * Q33
	Q41 = tf.constant(radial_e(x,4,1),dtype =tf.float32 ) - tf.get_variable("c4111", [1,1,1]) * Q11 - tf.get_variable("c4120", [1,1,1]) * Q20 - tf.get_variable("c4121", [1,1,1]) * Q21 - tf.get_variable("c4122", [1,1,1]) * Q22 - tf.get_variable("c4130", [1,1,1]) * Q30 - tf.get_variable("c4131", [1,1,1]) * Q31 - tf.get_variable("c4132", [1,1,1]) * Q32 - tf.get_variable("c4133", [1,1,1]) * Q33 - tf.get_variable("c4140", [1,1,1]) * Q40
	Q42 = tf.constant(radial_e(x,4,2),dtype =tf.float32 ) - tf.get_variable("c4211", [1,1,1]) * Q11 - tf.get_variable("c4220", [1,1,1]) * Q20 - tf.get_variable("c4221", [1,1,1]) * Q21 - tf.get_variable("c4222", [1,1,1]) * Q22 - tf.get_variable("c4230", [1,1,1]) * Q30 - tf.get_variable("c4231", [1,1,1]) * Q31 - tf.get_variable("c4232", [1,1,1]) * Q32 - tf.get_variable("c4233", [1,1,1]) * Q33 - tf.get_variable("c4240", [1,1,1]) * Q40 - tf.get_variable("c4241", [1,1,1]) * Q41
	
	Q43 = tf.constant(radial_e(x,4,3),dtype =tf.float32 ) - tf.get_variable("c4311", [1,1,1]) * Q11 - tf.get_variable("c4320", [1,1,1]) * Q20 - tf.get_variable("c4321", [1,1,1]) * Q21 - tf.get_variable("c4322", [1,1,1]) * Q22 - tf.get_variable("c4330", [1,1,1]) * Q30 - tf.get_variable("c4331", [1,1,1]) * Q31 - tf.get_variable("c4332", [1,1,1]) * Q32 - tf.get_variable("c4333", [1,1,1]) * Q33 - tf.get_variable("c4340", [1,1,1]) * Q40 - tf.get_variable("c4341", [1,1,1]) * Q41 - tf.get_variable("c4342", [1,1,1]) * Q42

	Q44 = tf.constant(radial_e(x,4,4),dtype =tf.float32 ) - tf.get_variable("c4411", [1,1,1]) * Q11 - tf.get_variable("c4420", [1,1,1]) * Q20 - tf.get_variable("c4421", [1,1,1]) * Q21 - tf.get_variable("c4422", [1,1,1]) * Q22 - tf.get_variable("c4430", [1,1,1]) * Q30 - tf.get_variable("c4431", [1,1,1]) * Q31 - tf.get_variable("c4432", [1,1,1]) * Q32 - tf.get_variable("c4433", [1,1,1]) * Q33 - tf.get_variable("c4440", [1,1,1]) * Q40 - tf.get_variable("c4441", [1,1,1]) * Q41 - tf.get_variable("c4442", [1,1,1]) * Q42 - tf.get_variable("c4443", [1,1,1]) * Q43
	
	Q50 = tf.constant(radial_e(x,5,0),dtype =tf.float32 ) - tf.get_variable("c5011", [1,1,1]) * Q11 - tf.get_variable("c5020", [1,1,1]) * Q20 - tf.get_variable("c5021", [1,1,1]) * Q21 - tf.get_variable("c5022", [1,1,1]) * Q22 - tf.get_variable("c5030", [1,1,1]) * Q30 - tf.get_variable("c5031", [1,1,1]) * Q31 - tf.get_variable("c5032", [1,1,1]) * Q32 - tf.get_variable("c5033", [1,1,1]) * Q33 - tf.get_variable("c5040", [1,1,1]) * Q40 - tf.get_variable("c5041", [1,1,1]) * Q41 - tf.get_variable("c5042", [1,1,1]) * Q42 - tf.get_variable("c5043", [1,1,1]) * Q43 - tf.get_variable("c5044", [1,1,1]) * Q44

	Q51 = tf.constant(radial_e(x,5,1),dtype =tf.float32 ) - tf.get_variable("c5111", [1,1,1]) * Q11 - tf.get_variable("c5120", [1,1,1]) * Q20 - tf.get_variable("c5121", [1,1,1]) * Q21 - tf.get_variable("c5122", [1,1,1]) * Q22 - tf.get_variable("c5130", [1,1,1]) * Q30 - tf.get_variable("c5131", [1,1,1]) * Q31 - tf.get_variable("c5132", [1,1,1]) * Q32 - tf.get_variable("c5133", [1,1,1]) * Q33 - tf.get_variable("c5140", [1,1,1]) * Q40 - tf.get_variable("c5141", [1,1,1]) * Q41 - tf.get_variable("c5142", [1,1,1]) * Q42 - tf.get_variable("c5143", [1,1,1]) * Q43 - tf.get_variable("c5144", [1,1,1]) * Q44  - tf.get_variable("c5150", [1,1,1]) * Q50

	Q52 = tf.constant(radial_e(x,5,2),dtype =tf.float32 ) - tf.get_variable("c5211", [1,1,1]) * Q11 - tf.get_variable("c5220", [1,1,1]) * Q20 - tf.get_variable("c5221", [1,1,1]) * Q21 - tf.get_variable("c5222", [1,1,1]) * Q22 - tf.get_variable("c5230", [1,1,1]) * Q30 - tf.get_variable("c5231", [1,1,1]) * Q31 - tf.get_variable("c5232", [1,1,1]) * Q32 - tf.get_variable("c5233", [1,1,1]) * Q33 - tf.get_variable("c5240", [1,1,1]) * Q40 - tf.get_variable("c5241", [1,1,1]) * Q41 - tf.get_variable("c5242", [1,1,1]) * Q42 - tf.get_variable("c5243", [1,1,1]) * Q43 - tf.get_variable("c5244", [1,1,1]) * Q44  - tf.get_variable("c5250", [1,1,1]) * Q50 - tf.get_variable("c5251", [1,1,1]) * Q51

	Q53 = tf.constant(radial_e(x,5,3),dtype =tf.float32 ) - tf.get_variable("c5311", [1,1,1]) * Q11 - tf.get_variable("c5320", [1,1,1]) * Q20 - tf.get_variable("c5321", [1,1,1]) * Q21 - tf.get_variable("c5322", [1,1,1]) * Q22 - tf.get_variable("c5330", [1,1,1]) * Q30 - tf.get_variable("c5331", [1,1,1]) * Q31 - tf.get_variable("c5332", [1,1,1]) * Q32 - tf.get_variable("c5333", [1,1,1]) * Q33 - tf.get_variable("c5340", [1,1,1]) * Q40 - tf.get_variable("c5341", [1,1,1]) * Q41 - tf.get_variable("c5342", [1,1,1]) * Q42 - tf.get_variable("c5343", [1,1,1]) * Q43 - tf.get_variable("c5344", [1,1,1]) * Q44  - tf.get_variable("c5350", [1,1,1]) * Q50 - tf.get_variable("c5351", [1,1,1]) * Q51 - tf.get_variable("c5352", [1,1,1]) * Q52


	Q54 = tf.constant(radial_e(x,5,4),dtype =tf.float32 ) - tf.get_variable("c5411", [1,1,1]) * Q11 - tf.get_variable("c5420", [1,1,1]) * Q20 - tf.get_variable("c5421", [1,1,1]) * Q21 - tf.get_variable("c5422", [1,1,1]) * Q22 - tf.get_variable("c5430", [1,1,1]) * Q30 - tf.get_variable("c5431", [1,1,1]) * Q31 - tf.get_variable("c5432", [1,1,1]) * Q32 - tf.get_variable("c5433", [1,1,1]) * Q33 - tf.get_variable("c5440", [1,1,1]) * Q40 - tf.get_variable("c5441", [1,1,1]) * Q41 - tf.get_variable("c5442", [1,1,1]) * Q42 - tf.get_variable("c5443", [1,1,1]) * Q43 - tf.get_variable("c5444", [1,1,1]) * Q44  - tf.get_variable("c5450", [1,1,1]) * Q50 - tf.get_variable("c5451", [1,1,1]) * Q51 - tf.get_variable("c5452", [1,1,1]) * Q52 - tf.get_variable("c5453", [1,1,1]) * Q53

	Q55 = tf.constant(radial_e(x,5,5),dtype =tf.float32 ) - tf.get_variable("c5511", [1,1,1]) * Q11 - tf.get_variable("c5520", [1,1,1]) * Q20 - tf.get_variable("c5521", [1,1,1]) * Q21 - tf.get_variable("c5522", [1,1,1]) * Q22 - tf.get_variable("c5530", [1,1,1]) * Q30 - tf.get_variable("c5531", [1,1,1]) * Q31 - tf.get_variable("c5532", [1,1,1]) * Q32 - tf.get_variable("c5533", [1,1,1]) * Q33 - tf.get_variable("c5540", [1,1,1]) * Q40 - tf.get_variable("c5541", [1,1,1]) * Q41 - tf.get_variable("c5542", [1,1,1]) * Q42 - tf.get_variable("c5543", [1,1,1]) * Q43 - tf.get_variable("c5544", [1,1,1]) * Q44  - tf.get_variable("c5550", [1,1,1]) * Q50 - tf.get_variable("c5551", [1,1,1]) * Q51 - tf.get_variable("c5552", [1,1,1]) * Q52 - tf.get_variable("c5553", [1,1,1]) * Q53 - tf.get_variable("c5554", [1,1,1]) * Q54

	if n==0 and l == 0:
                return Q00
	if n==1 and l == 0:
                return Q10
        if n==1 and l == 1:
                return Q11
        if n==2 and l == 0:
                return Q20
        if n == 2 and l == 1:
                return Q21
        if n == 2 and l == 2:
                return Q22
        if n == 3 and l == 0:
                return Q30
        if n == 3 and l == 1:
                return Q31
        if n == 3 and l == 2:
                return Q32
        if n == 3 and l == 3:
                return Q33
        if n == 4 and l ==0:
                return Q40
        if n == 4 and l == 1:
                return Q41
        if n == 4 and l == 2:
                return Q42
        if n == 4 and l == 3:
                return Q43
        if n == 4 and l == 4:
                return Q44
        if n == 5 and l == 0:
                return Q50
        if n == 5 and l == 1:
                return Q51
        if n == 5 and l ==2:
                return Q52
        if n == 5 and l == 3:
                return Q53
        if n == 5 and l == 4:
                return Q54
        if n == 5 and l == 5:
                return Q55


def radial_poly__(x,n,l):
	if n==0 and l ==0:
		return 1. - 1.*x
	if n==1 and l==0:
		return 1. - 1.*x
	if n==1 and l==1:
		return 1. - 1.*x
	if n==1 and l == 1:
		return -1. - 1.*x
	if n==2 and l == 0:
		return -4.935483870967742 - 0.9354838709677419*x + 9.*np.power(x,2)
	if n == 2 and l == 1:
		return 0.04261445436222286 - 0.1357136001535656*x + 0.09876187733947596*np.power(x,2)
	if n == 2 and l == 2:
		return 0
	if n == 3 and l == 0:
		return -5.714285714285715 + 34.285714285714285*x - 60.*np.power(x,2) + 32.*np.power(x,3)
	if n == 3 and l == 1:
		return -4.163780431554187*1e-12 + 2.497024809144932*1e-11*x - 4.369837824924616*1e-11*np.power(x,2) + 2.3305801732931286*1e-11*np.power(x,3)
	if n == 3 and l == 2:
		return 2.4289459332749175*1e-12 - 1.4573342532742117*1e-11*x + 2.5503599232479246*1e-11*np.power(x,2) - 1.3602452497707418*1e-11*np.power(x,3)
	if n == 3 and l == 3:
		return -3.602673714908633*1e-13 + 2.1613821843402548*1e-12*x - 3.782252289141752*1e-12*np.power(x,2) + 2.017164213441447*1e-12*np.power(x,3)
	if n == 4 and l ==0:
		return 7.440476190638279 - 69.44444444541688*x + 208.3333333350351*np.power(x,2) - 250.00000000090768*np.power(x,3) + 104.16666666666667*np.power(x,4)
	if n == 4 and l == 1:
		return -1.963478268862673*1e-9 + 1.8308398352928634*1e-8*x - 5.488601573233609*1e-8*np.power(x,2) + 6.582659040077488*1e-8*np.power(x,3) - 2.741556670571299*1e-8*np.power(x,4)
	if n == 4 and l == 2:
		return -1.374908409346176*1e-9 + 1.2823202055756155*1e-8*x - 3.844862561663831*1e-8*np.power(x,2) + 4.611872839177522*1e-8*np.power(x,3) - 1.9209622337257315*1e-8*np.power(x,4)
	if n == 4 and l == 3:
		return 7.009438307559179*1e-10 - 6.537717922583397*1e-9*x + 1.9603168865955922*1e-8*np.power(x,2) - 2.351445171910882*1e-8*np.power(x,3) + 9.794585587030724*1e-9*np.power(x,4)
	if n == 4 and l == 4:
		return -2.1674161597429187*1e-10 + 2.0174599490996314*1e-9*x - 6.0400779378078084*1e-9*np.power(x,2) + 7.236603871696445*1e-9*np.power(x,3) - 3.0114259141900135*1e-9*np.power(x,4)
	if n == 5 and l == 0:
		return  -8.590909289732906 + 114.54545639644891*x - 515.45455099696*np.power(x,2) + 1030.909097550118*np.power(x,3) - 945.0000027638082*np.power(x,4) + 324.*np.power(x,5)
	if n == 5 and l == 1:
		return  1.927501564580325*1e-6 - 0.00002568559852988983*x + 0.00011553473518688406*np.power(x,2) - 0.00023098875874438818*np.power(x,3) + 0.00021167917489606225*np.power(x,4) - 0.0000725584284850811*np.power(x,5)
	if n == 5 and l ==2:
		return 0.0006869920280685449 - 0.009154762803518679*x + 0.0411784815441365*np.power(x,2) - 0.08232824900557034*np.power(x,3) + 0.07544603068063083*np.power(x,4) - 0.025861059890907256*np.power(x,5)
	if n == 5 and l == 3:
		return  -0.24856780498715203 + 3.3123809379145968*x - 14.899219170168017*np.power(x,2) + 29.788049021077427*np.power(x,3) - 27.29792129722461*np.power(x,4) + 9.357061883809266*np.power(x,5)
	if n == 5 and l == 4:
		return 0.060824031786470556 - 0.8105328202587492*x + 3.6458083732857745*np.power(x,2) - 7.289074513642131*np.power(x,3) + 6.679745374855816*np.power(x,4) - 2.28965386158944*np.power(x,5)
	if n == 5 and l == 5:
		return -0.0004485726132123515 + 0.005985870202233776*x - 0.026953566210279903*np.power(x,2)+ 0.05393456559134158*np.power(x,3) - 0.04946058887972546*np.power(x,4) + 0.01696379243557726*np.power(x,5)

"""
Q00 = 0.
Q01 = 1. + 2.*x
Q11 = -1. - 1.*x
Q20 = -4.935483870967742 - 0.9354838709677419*x + 9.*x^2
Q21 = 0.04261445436222286 - 0.1357136001535656*x + 0.09876187733947596*x^2
Q22 = 0.
Q30 = -5.714285714285715 + 34.285714285714285*x - 60.*x^2 + 32.*x^3
Q31 = -4.163780431554187*1e-12 + 2.497024809144932*1e-11*x - 4.369837824924616*1e-11*x^2 + 2.3305801732931286*1e-11*x^3
Q32 = 2.4289459332749175*1e-12 - 1.4573342532742117*1e-11*x + 2.5503599232479246*1e-11*x^2 - 1.3602452497707418*1e-11*x^3
Q33 = -3.602673714908633*1e-13 + 2.1613821843402548*1e-12*x - 3.782252289141752*1e-12*x^2 + 2.017164213441447*1e-12*x^3
Q40 = 7.440476190638279 - 69.44444444541688*x + 208.3333333350351*x^2 - 250.00000000090768*x^3 + 104.16666666666667*x^4
Q41 = -1.963478268862673*1e-9 + 1.8308398352928634*1e-8*x - 5.488601573233609*1e-8*x^2 + 6.582659040077488*1e-8*x^3 - 2.741556670571299*1e-8*x^4
Q42 = -1.374908409346176*^-9 + 1.2823202055756155*^-8*x - 3.844862561663831*^-8*x^2 + 4.611872839177522*^-8*x^3 - 1.9209622337257315*^-8*x^4
Q43 = 7.009438307559179*^-10 - 6.537717922583397*^-9*x + 1.9603168865955922*^-8*x^2 - 2.351445171910882*^-8*x^3 + 9.794585587030724*^-9*x^4
Q44 = -2.1674161597429187*1e-10 + 2.0174599490996314*1e-9*x - 6.0400779378078084*1e-9*x^2 + 7.236603871696445*1e-9*x^3 - 3.0114259141900135*1e-9*x^4
Q50 = -8.590909289732906 + 114.54545639644891*x - 515.45455099696*x^2 + 1030.909097550118*x^3 - 945.0000027638082*x^4 + 324.*x^5
Q51 = 1.927501564580325*1e-6 - 0.00002568559852988983*x + 0.00011553473518688406*x^2 - 0.00023098875874438818*x^3 + 0.00021167917489606225*x^4 - 0.0000725584284850811*x^5
Q52 = 0.0006869920280685449 - 0.009154762803518679*x + 0.0411784815441365*x^2 - 0.08232824900557034*x^3 + 0.07544603068063083*x^4 - 0.025861059890907256*x^5
Q53 = -0.24856780498715203 + 3.3123809379145968*x - 14.899219170168017*x^2 + 29.788049021077427*x^3 - 27.29792129722461*x^4 + 9.357061883809266*x^5
Q54 = 0.060824031786470556 - 0.8105328202587492*x + 3.6458083732857745*x^2 - 7.289074513642131*x^3 + 6.679745374855816*x^4 - 2.28965386158944*x^5
Q55 = -0.0004485726132123515 + 0.005985870202233776*x - 0.026953566210279903*x^2 + 0.05393456559134158*x^3 - 0.04946058887972546*x^4 + 0.01696379243557726*x^5
"""


###########################################################################

y_0_0 =  spherical_harmonic(0.0,0.0)     * (tf.zeros(tf.shape(theta)) + 1)
y_0_1 =  spherical_harmonic(0.0,1.0)   * tf.cos(theta)
y_1_1 =   spherical_harmonic(1.0,1.0)   * (-1.0) * tf.sqrt(1-tf.square(tf.cos(theta)))
y_0_2 =  spherical_harmonic(0.0,2.0)    *(1.0/2.0) * (3* tf.square(tf.cos(theta)) - 1)
y_1_2 =   spherical_harmonic(1.0,2.0)   * (-1.0) * tf.sqrt(1-tf.square(tf.cos(theta))) * 3.0 * tf.cos(theta)
y_2_2 =   spherical_harmonic(2.0,2.0)   * (1-tf.square(tf.cos(theta))) * 3.0
y_0_3 =  spherical_harmonic(0.0,3.0)    * (1.0/2.0) * (5 * tf.pow(tf.cos(theta),3) - 3 * tf.cos(theta))
y_1_3 =   spherical_harmonic(1.0,3.0)    * (-1.0) *  (1.0/2.0)*  tf.sqrt(1-tf.square(tf.cos(theta))) * (15 * tf.square(tf.cos(theta)) - 3 )
y_2_3 =  spherical_harmonic(2.0,3.0)    * 15 * tf.cos(theta) * (1.0-tf.square(tf.cos(theta)))
y_3_3 =   spherical_harmonic(3.0,3.0)     * (-1.0) * 15.0 * tf.pow((1.0-tf.square(tf.cos(theta))),3.0/2.0)
y_0_4 =  spherical_harmonic(0.0,4.0) * (1.0/8.0) * (35.0 * tf.pow(tf.cos(theta),4) - 30.0 * tf.square(tf.cos(theta)) + 3)
y_1_4 = spherical_harmonic(1.0,4.0) * (-5.0/2.0) * (7.0 * tf.pow(tf.cos(theta),3) - 3.0 * tf.cos(theta)) * (tf.sqrt(1-tf.square(tf.cos(theta))))
y_2_4 = spherical_harmonic(2.0,4.0) * (15.0/2.0) * (7.0 * tf.square(tf.cos(theta)) - 1.0) * (1.0 - tf.square(tf.cos(theta)))
y_3_4 = spherical_harmonic(3.0,4.0) * (-105.0) * tf.cos(theta) * tf.pow(tf.sqrt(1.0-tf.square(tf.cos(theta))), 3.0/2)
y_4_4 = spherical_harmonic(4.0,4.0) * (105.0) * tf.square(1.0 - tf.square(tf.cos(theta)))
y_0_5 = spherical_harmonic(0.0,5.0) * (1.0/8.0)*tf.cos(theta)*(63.0*tf.pow(tf.cos(theta),4)-70.0*tf.square(tf.cos(theta)) +  15)
y_1_5 = spherical_harmonic(1.0,5.0)* (-15.0/8.0)*tf.sqrt(1-tf.square(tf.cos(theta)))*(21.0*tf.pow(tf.cos(theta),4) - 14.0*tf.square(tf.cos(theta)) + 1)
y_2_5 = spherical_harmonic(2.0,5.0)* (105.0/2.0)*tf.cos(theta)*(1-tf.square(tf.cos(theta)))*(3.0*tf.square(tf.cos(theta))-1)
y_3_5 = spherical_harmonic(3.0,5.0) * (-105.0/2.0)*tf.pow(tf.sin(theta),3)*(9.0*tf.square(tf.cos(theta))-1)
y_4_5 = spherical_harmonic(4.0,5.0)*945.0*tf.cos(theta)*tf.pow(tf.sin(theta),4)
y_5_5 = spherical_harmonic(5.0,5.0)*-945.0*tf.pow(tf.sin(theta),5)
y_0_6 = spherical_harmonic(0.0,6.0)*(1.0/16)*(231.0*tf.pow(tf.cos(theta),6)-315.0*tf.pow(tf.cos(theta),315.0)+105.0*tf.square(tf.cos(theta))-5)
y_1_6 = spherical_harmonic(1.0,6.0)*(-21.0/8.0)*tf.cos(theta)*(33.0*tf.pow(tf.cos(theta),4)-30.0*tf.square(tf.cos(theta))+5)*tf.sin(theta)
y_2_6 = spherical_harmonic(2.0,6.0)*(105.0/8.0)*tf.square(tf.sin(theta))*(33.0*tf.pow(tf.cos(theta),4)-18.0*tf.square(tf.cos(theta))+1)
y_3_6 = spherical_harmonic(3.0,6.0)*(-315.0/2.0)*(11*tf.square(tf.cos(theta))-3)*tf.cos(theta)*tf.pow(tf.sin(theta),3)
y_4_6 = spherical_harmonic(4.0,6.0)*(945.0/2.0)*(11.0*tf.square(tf.cos(theta))-1)*tf.pow(tf.sin(theta),4)
y_5_6 = spherical_harmonic(5.0,6.0)*-10395.0*tf.cos(theta)*tf.pow(tf.sin(theta),5)
y_6_6 = spherical_harmonic(6.0,6.0)*10395.0*tf.pow(tf.sin(theta),6)

def Y(l,m,theta, phi, scope= "test"):
	with tf.variable_scope(scope):
		if l == 0 and m == 0:
			return spherical_harmonic(0.0,0.0)     * (tf.zeros(tf.shape(theta)) + 1) *tf.cos(0.0 * phi)
                if l == 1 and m == 0:
			return spherical_harmonic(0.0,1.0)   * tf.cos(theta)*tf.cos(0.0 * phi)
                if l == 1 and m == 1:
			return spherical_harmonic(1.0,1.0)   * (-1.0) * tf.sqrt(1-tf.square(tf.cos(theta)))*tf.cos(1.0 * phi)
                if l == 2 and m == 0:
			return spherical_harmonic(0.0,2.0)    *(1.0/2.0) * (3* tf.square(tf.cos(theta)) - 1)*tf.cos(0.0 * phi)
                if l == 2 and m == 1:
			return spherical_harmonic(1.0,2.0)   * (-1.0) * tf.sqrt(1-tf.square(tf.cos(theta))) * 3.0 * tf.cos(theta)*tf.cos(1.0 * phi)
                if l == 2 and m == 2:
			return spherical_harmonic(2.0,2.0)   * (1-tf.square(tf.cos(theta))) * 3.0*tf.cos(2.0 * phi)
                if l == 3 and m == 0:
			return spherical_harmonic(0.0,3.0)    * (1.0/2.0) * (5 * tf.pow(tf.cos(theta),3) - 3 * tf.cos(theta))	*tf.cos(0.0 * phi)		
                if l == 3 and m == 1:
			return spherical_harmonic(1.0,3.0)    * (-1.0) *  (1.0/2.0)*  tf.sqrt(1-tf.square(tf.cos(theta))) * (15 * tf.square(tf.cos(theta)) - 3 )*tf.cos(1.0 * phi)
                if l == 3 and m == 2:
			return spherical_harmonic(2.0,3.0)    * 15 * tf.cos(theta) * (1.0-tf.square(tf.cos(theta)))*tf.cos(2.0 * phi)
                if l == 3 and m == 3:
			return spherical_harmonic(3.0,3.0)     * (-1.0) * 15.0 * tf.pow((1.0-tf.square(tf.cos(theta))),3.0/2.0)*tf.cos(3.0 * phi)
                if l == 4 and m == 0:
			return spherical_harmonic(0.0,4.0) * (1.0/8.0) * (35.0 * tf.pow(tf.cos(theta),4) - 30.0 * tf.square(tf.cos(theta)) + 3)*tf.cos(0.0 * phi)
                if l == 4 and m == 1:
			return spherical_harmonic(1.0,4.0) * (-5.0/2.0) * (7.0 * tf.pow(tf.cos(theta),3) - 3.0 * tf.cos(theta)) * (tf.sqrt(1-tf.square(tf.cos(theta))))*tf.cos(1.0 * phi)
                if l == 4 and m == 2:
			return spherical_harmonic(2.0,4.0) * (15.0/2.0) * (7.0 * tf.square(tf.cos(theta)) - 1.0) * (1.0 - tf.square(tf.cos(theta)))*tf.cos(2.0 * phi)
                if l == 4 and m == 3:
			return spherical_harmonic(3.0,4.0) * (-105.0) * tf.cos(theta) * tf.pow(tf.sqrt(1.0-tf.square(tf.cos(theta))), 3.0/2)*tf.cos(3.0 * phi)
                if l == 4 and m == 4:
			return spherical_harmonic(4.0,4.0) * (105.0) * tf.square(1.0 - tf.square(tf.cos(theta)))*tf.cos(4.0 * phi)
                if l == 5 and m == 0:
			return spherical_harmonic(0.0,5.0) * (1.0/8.0)*tf.cos(theta)*(63.0*tf.pow(tf.cos(theta),4)-70.0*tf.square(tf.cos(theta)) +  15)*tf.cos(0.0 * phi)
                if l == 5 and m == 1:
			return spherical_harmonic(1.0,5.0)* (-15.0/8.0)*tf.sqrt(1-tf.square(tf.cos(theta)))*(21.0*tf.pow(tf.cos(theta),4) - 14.0*tf.square(tf.cos(theta)) + 1)*tf.cos(1.0 * phi)
                if l == 5 and m == 2:
			return spherical_harmonic(2.0,5.0)* (105.0/2.0)*tf.cos(theta)*(1-tf.square(tf.cos(theta)))*(3.0*tf.square(tf.cos(theta))-1)*tf.cos(2.0 * phi)
                if l == 5 and m == 3:
			return spherical_harmonic(3.0,5.0) * (-105.0/2.0)*tf.pow(tf.sin(theta),3)*(9.0*tf.square(tf.cos(theta))-1)*tf.cos(3.0 * phi)
                if l == 5 and m == 4:
			return spherical_harmonic(4.0,5.0)*945.0*tf.cos(theta)*tf.pow(tf.sin(theta),4)*tf.cos(4.0 * phi)
                if l == 5 and m == 5:
			return spherical_harmonic(5.0,5.0)*-945.0*tf.pow(tf.sin(theta),5)*tf.cos(5.0 * phi)

test_10 =  y_4_4
print("1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(math.factorial(4.0-4.0))


def Yi(l,m,theta, phi, scope="test"):
        with tf.variable_scope(scope):
                if l == 0 and m == 0:
                        return spherical_harmonic(0.0,0.0)     * (tf.zeros(tf.shape(theta)) + 1) *tf.sin(0.0 * phi)
                if l == 1 and m == 0:
                        return spherical_harmonic(0.0,1.0)   * tf.cos(theta)*tf.sin(0.0 * phi)
                if l == 1 and m == 1:
                        return spherical_harmonic(1.0,1.0)   * (-1.0) * tf.sqrt(1-tf.square(tf.cos(theta)))*tf.sin(1.0 * phi)
                if l == 2 and m == 0:
                        return spherical_harmonic(0.0,2.0)    *(1.0/2.0) * (3* tf.square(tf.cos(theta)) - 1)*tf.sin(0.0 * phi)
                if l == 2 and m == 1:
                        return spherical_harmonic(1.0,2.0)   * (-1.0) * tf.sqrt(1-tf.square(tf.cos(theta))) * 3.0 * tf.sin(theta)*tf.cos(1.0 * phi)
                if l == 2 and m == 2:
                        return spherical_harmonic(2.0,2.0)   * (1-tf.square(tf.cos(theta))) * 3.0*tf.sin(2.0 * phi)
                if l == 3 and m == 0:
                        return spherical_harmonic(0.0,3.0)    * (1.0/2.0) * (5 * tf.pow(tf.cos(theta),3) - 3 * tf.sin(theta))   *tf.cos(0.0 * phi)
                if l == 3 and m == 1:
                        return spherical_harmonic(1.0,3.0)    * (-1.0) *  (1.0/2.0)*  tf.sqrt(1-tf.square(tf.cos(theta))) * (15 * tf.square(tf.cos(theta)) - 3 )*tf.sin(1.0 * phi)
                if l == 3 and m == 2:
                        return spherical_harmonic(2.0,3.0)    * 15 * tf.cos(theta) * (1.0-tf.square(tf.cos(theta)))*tf.sin(2.0 * phi)
                if l == 3 and m == 3:
                        return spherical_harmonic(3.0,3.0)     * (-1.0) * 15.0 * tf.pow((1.0-tf.square(tf.cos(theta))),3.0/2.0)*tf.sin(3.0 * phi)
                if l == 4 and m == 0:
                        return spherical_harmonic(0.0,4.0) * (1.0/8.0) * (35.0 * tf.pow(tf.cos(theta),4) - 30.0 * tf.square(tf.cos(theta)) + 3)*tf.sin(0.0 * phi)
                if l == 4 and m == 1:
                        return spherical_harmonic(1.0,4.0) * (-5.0/2.0) * (7.0 * tf.pow(tf.cos(theta),3) - 3.0 * tf.cos(theta)) * (tf.sqrt(1-tf.square(tf.cos(theta))))*tf.sin(1.0 * phi)
                if l == 4 and m == 2:
                        return spherical_harmonic(2.0,4.0) * (15.0/2.0) * (7.0 * tf.square(tf.cos(theta)) - 1.0) * (1.0 - tf.square(tf.cos(theta)))*tf.sin(2.0 * phi)
                if l == 4 and m == 3:
                        return spherical_harmonic(3.0,4.0) * (-105.0) * tf.cos(theta) * tf.pow(tf.sqrt(1.0-tf.square(tf.cos(theta))), 3.0/2)*tf.sin(3.0 * phi)

                if l == 4 and m == 4:
                        return spherical_harmonic(4.0,4.0) * (105.0) * tf.square(1.0 - tf.square(tf.cos(theta)))*tf.sin(4.0 * phi)
                if l == 5 and m == 0:
                        return spherical_harmonic(0.0,5.0) * (1.0/8.0)*tf.cos(theta)*(63.0*tf.pow(tf.cos(theta),4)-70.0*tf.square(tf.cos(theta)) +  15)*tf.sin(0.0 * phi)
                if l == 5 and m == 1:
                        return spherical_harmonic(1.0,5.0)* (-15.0/8.0)*tf.sqrt(1-tf.square(tf.cos(theta)))*(21.0*tf.pow(tf.cos(theta),4) - 14.0*tf.square(tf.cos(theta)) + 1)*tf.sin(1.0 * phi)
                if l == 5 and m == 2:
                        return spherical_harmonic(2.0,5.0)* (105.0/2.0)*tf.cos(theta)*(1-tf.square(tf.cos(theta)))*(3.0*tf.square(tf.cos(theta))-1)*tf.sin(2.0 * phi)
                if l == 5 and m == 3:
                        return spherical_harmonic(3.0,5.0) * (-105.0/2.0)*tf.pow(tf.sin(theta),3)*(9.0*tf.square(tf.cos(theta))-1)*tf.sin(3.0 * phi)
                if l == 5 and m == 4:
                        return spherical_harmonic(4.0,5.0)*945.0*tf.cos(theta)*tf.pow(tf.sin(theta),4)*tf.sin(4.0 * phi)
                if l == 5 and m == 5:
                        return spherical_harmonic(5.0,5.0)*-945.0*tf.pow(tf.sin(theta),5)*tf.sin(5.0 * phi)


#u_0_0_0 = tf.multiply(y_0_0, tf.cos(0.0)) * radial_poly(r,0,0)
u_1_1_0 = tf.multiply(y_0_1, tf.cos(0.0)) * radial_poly(r,1,1)
u_1_1_1 = tf.multiply(y_1_1, tf.cos(phi)) * radial_poly(r,1,1)
u_2_0_0 = tf.multiply(y_0_0, tf.cos(0.0)) * radial_poly(r,2,0)
u_2_1_0 = tf.multiply(y_0_1, tf.cos(0.0)) * radial_poly(r,2,1)
u_2_1_1 = tf.multiply(y_1_1, tf.cos(phi)) * radial_poly(r,2,1)
u_2_2_0 = tf.multiply(y_0_2, tf.cos(0.0)) * radial_poly(r,2,2)
u_2_2_1 = tf.multiply(y_1_2, tf.cos(phi)) * radial_poly(r,2,2)
u_2_2_2 = tf.multiply(y_2_2, tf.cos(2.0*phi)) * radial_poly(r,2,2)
u_3_0_0 = tf.multiply(y_0_0, tf.cos(0.0)) * radial_poly(r,3,0)
u_3_1_0 = tf.multiply(y_0_1, tf.cos(phi)) * radial_poly(r,3,1)
u_3_1_1 = tf.multiply(y_1_1, tf.cos(phi)) * radial_poly(r,3,1)
u_3_2_0 = tf.multiply(y_0_2, tf.cos(0.0)) * radial_poly(r,3,2)
u_3_2_1 = tf.multiply(y_1_2, tf.cos(phi)) * radial_poly(r,3,2)
u_3_2_2 = tf.multiply(y_2_2, tf.cos(2.0*phi)) * radial_poly(r,3,2)
u_3_3_0 = tf.multiply(y_0_3, tf.cos(0.0)) * radial_poly(r,3,3)
u_3_3_1 = tf.multiply(y_1_3, tf.cos(phi)) * radial_poly(r,3,3)
u_3_3_2 = tf.multiply(y_2_3, tf.cos(2.0*phi))* radial_poly(r,3,3)
u_3_3_3 = tf.multiply(y_3_3, tf.cos(3.0*phi)) * radial_poly(r,3,3)
u_4_0_0 = tf.multiply(y_0_0, tf.cos(0.0))* radial_poly(r,4,0)
u_4_1_0 = tf.multiply(y_0_1, tf.cos(0.0))* radial_poly(r,4,1)
u_4_1_1 = tf.multiply(y_1_1, tf.cos(1.0 * phi))* radial_poly(r,4,1)
u_4_2_0 = tf.multiply(y_0_2, tf.cos(0.0 * phi))* radial_poly(r,4,2)
u_4_2_1 = tf.multiply(y_1_2, tf.cos(1.0 * phi))* radial_poly(r,4,2)
u_4_2_2 = tf.multiply(y_2_2, tf.cos(2.0 * phi))* radial_poly(r,4,2)
u_4_3_0 = tf.multiply(y_0_3, tf.cos(0.0 * phi))* radial_poly(r,4,3)
u_4_3_1 = tf.multiply(y_1_3, tf.cos(1.0 * phi))* radial_poly(r,4,3)
u_4_3_2 = tf.multiply(y_2_3, tf.cos(2.0 * phi))* radial_poly(r,4,3)
u_4_3_3 = tf.multiply(y_3_3, tf.cos(3.0 * phi))* radial_poly(r,4,3)
u_4_4_0 = tf.multiply(y_0_4, tf.cos(0.0 * phi))* radial_poly(r,4,4)
u_4_4_1 = tf.multiply(y_1_4, tf.cos(1.0 * phi))* radial_poly(r,4,4)
u_4_4_2 = tf.multiply(y_2_4, tf.cos(2.0 * phi))* radial_poly(r,4,4)
u_4_4_3 = tf.multiply(y_3_4, tf.cos(3.0 * phi))* radial_poly(r,4,4)
u_4_4_4 = tf.multiply(y_4_4, tf.cos(4.0 * phi))* radial_poly(r,4,4)
u_5_0_0 = tf.multiply(y_0_0, tf.cos(0.0 * phi))* radial_poly(r,5,0)
u_5_1_0 = tf.multiply(y_0_1, tf.cos(0.0 * phi))* radial_poly(r,5,1)
u_5_1_1 = tf.multiply(y_1_1, tf.cos(1.0 * phi))* radial_poly(r,5,1)
u_5_2_0 = tf.multiply(y_0_2, tf.cos(0.0 * phi))* radial_poly(r,5,2)
u_5_2_1 = tf.multiply(y_1_2, tf.cos(1.0 * phi))* radial_poly(r,5,2)
u_5_2_2 = tf.multiply(y_2_2, tf.cos(2.0 * phi))* radial_poly(r,5,2)
u_5_3_0 = tf.multiply(y_0_3, tf.cos(0.0 * phi))* radial_poly(r,5,3)
u_5_3_1 = tf.multiply(y_1_3, tf.cos(1.0 * phi))* radial_poly(r,5,3)
u_5_3_2 = tf.multiply(y_2_3, tf.cos(2.0 * phi))* radial_poly(r,5,3)
u_5_3_3 = tf.multiply(y_3_3, tf.cos(3.0 * phi))* radial_poly(r,5,3)
u_5_4_0 = tf.multiply(y_0_4, tf.cos(0.0 * phi))* radial_poly(r,5,4)
u_5_4_1 = tf.multiply(y_1_4, tf.cos(1.0 * phi))* radial_poly(r,5,4)
u_5_4_2 = tf.multiply(y_2_4, tf.cos(2.0 * phi))* radial_poly(r,5,4)
u_5_4_3 = tf.multiply(y_3_4, tf.cos(3.0 * phi))* radial_poly(r,5,4)
u_5_4_4 = tf.multiply(y_4_4, tf.cos(4.0 * phi))* radial_poly(r,5,4)
u_5_5_0 = tf.multiply(y_0_5, tf.cos(0.0 * phi))* radial_poly(r,5,5)
u_5_5_1 = tf.multiply(y_1_5, tf.cos(1.0 * phi))* radial_poly(r,5,5)
u_5_5_2 = tf.multiply(y_2_5, tf.cos(2.0 * phi))* radial_poly(r,5,5)
u_5_5_3 = tf.multiply(y_3_5, tf.cos(3.0 * phi))* radial_poly(r,5,5)
u_5_5_4 = tf.multiply(y_4_5, tf.cos(4.0 * phi))* radial_poly(r,5,5)
u_5_5_5 = tf.multiply(y_5_5, tf.cos(5.0 * phi))* radial_poly(r,5,5)


v_1_1_0 = tf.multiply(y_0_1, tf.sin(0.0)) * radial_poly(r,1,1)
v_1_1_1 = tf.multiply(y_1_1, tf.sin(phi)) * radial_poly(r,1,1)
v_2_0_0 = tf.multiply(y_0_0, tf.sin(0.0)) * radial_poly(r,2,0)
v_2_1_0 = tf.multiply(y_0_1, tf.sin(0.0)) * radial_poly(r,2,1)
v_2_1_1 = tf.multiply(y_1_1, tf.sin(phi)) * radial_poly(r,2,1)
v_2_2_0 = tf.multiply(y_0_2, tf.sin(0.0)) * radial_poly(r,2,2)
v_2_2_1 = tf.multiply(y_1_2, tf.sin(phi)) * radial_poly(r,2,2)
v_2_2_2 = tf.multiply(y_2_2, tf.sin(2.0*phi)) * radial_poly(r,2,2)
v_3_0_0 = tf.multiply(y_0_0, tf.sin(0.0)) * radial_poly(r,3,0)
v_3_1_0 = tf.multiply(y_0_1, tf.sin(phi)) * radial_poly(r,3,1)
v_3_1_1 = tf.multiply(y_1_1, tf.sin(phi)) * radial_poly(r,3,1)
v_3_2_0 = tf.multiply(y_0_2, tf.sin(0.0)) * radial_poly(r,3,2)
v_3_2_1 = tf.multiply(y_1_2, tf.sin(phi)) * radial_poly(r,3,2)
v_3_2_2 = tf.multiply(y_2_2, tf.sin(2.0*phi)) * radial_poly(r,3,2)
v_3_3_0 = tf.multiply(y_0_3, tf.sin(0.0)) * radial_poly(r,3,3)
v_3_3_1 = tf.multiply(y_1_3, tf.sin(phi)) * radial_poly(r,3,3)
v_3_3_2 = tf.multiply(y_2_3, tf.sin(2.0*phi))* radial_poly(r,3,3)
v_3_3_3 = tf.multiply(y_3_3, tf.sin(3.0*phi)) * radial_poly(r,3,3)
v_4_0_0 = tf.multiply(y_0_0, tf.sin(0.0))* radial_poly(r,4,0)
v_4_1_0 = tf.multiply(y_0_1, tf.sin(0.0))* radial_poly(r,4,1)
v_4_1_1 = tf.multiply(y_1_1, tf.sin(1.0 * phi))* radial_poly(r,4,1)
v_4_2_0 = tf.multiply(y_0_2, tf.sin(0.0 * phi))* radial_poly(r,4,2)
v_4_2_1 = tf.multiply(y_1_2, tf.sin(1.0 * phi))* radial_poly(r,4,2)
v_4_2_2 = tf.multiply(y_2_2, tf.sin(2.0 * phi))* radial_poly(r,4,2)
v_4_3_0 = tf.multiply(y_0_3, tf.sin(0.0 * phi))* radial_poly(r,4,3)
v_4_3_1 = tf.multiply(y_1_3, tf.sin(1.0 * phi))* radial_poly(r,4,3)
v_4_3_2 = tf.multiply(y_2_3, tf.sin(2.0 * phi))* radial_poly(r,4,3)
v_4_3_3 = tf.multiply(y_3_3, tf.sin(3.0 * phi))* radial_poly(r,4,3)
v_4_4_0 = tf.multiply(y_0_4, tf.sin(0.0 * phi))* radial_poly(r,4,4)
v_4_4_1 = tf.multiply(y_1_4, tf.sin(1.0 * phi))* radial_poly(r,4,4)
v_4_4_2 = tf.multiply(y_2_4, tf.sin(2.0 * phi))* radial_poly(r,4,4)
v_4_4_3 = tf.multiply(y_3_4, tf.sin(3.0 * phi))* radial_poly(r,4,4)
v_4_4_4 = tf.multiply(y_4_4, tf.sin(4.0 * phi))* radial_poly(r,4,4)
v_5_0_0 = tf.multiply(y_0_0, tf.sin(0.0 * phi))* radial_poly(r,5,0)
v_5_1_0 = tf.multiply(y_0_1, tf.sin(0.0 * phi))* radial_poly(r,5,1)
v_5_1_1 = tf.multiply(y_1_1, tf.sin(1.0 * phi))* radial_poly(r,5,1)
v_5_2_0 = tf.multiply(y_0_2, tf.sin(0.0 * phi))* radial_poly(r,5,2)
v_5_2_1 = tf.multiply(y_1_2, tf.sin(1.0 * phi))* radial_poly(r,5,2)
v_5_2_2 = tf.multiply(y_2_2, tf.sin(2.0 * phi))* radial_poly(r,5,2)
v_5_3_0 = tf.multiply(y_0_3, tf.sin(0.0 * phi))* radial_poly(r,5,3)
v_5_3_1 = tf.multiply(y_1_3, tf.sin(1.0 * phi))* radial_poly(r,5,3)
v_5_3_2 = tf.multiply(y_2_3, tf.sin(2.0 * phi))* radial_poly(r,5,3)
v_5_3_3 = tf.multiply(y_3_3, tf.sin(3.0 * phi))* radial_poly(r,5,3)
v_5_4_0 = tf.multiply(y_0_4, tf.sin(0.0 * phi))* radial_poly(r,5,4)
v_5_4_1 = tf.multiply(y_1_4, tf.sin(1.0 * phi))* radial_poly(r,5,4)
v_5_4_2 = tf.multiply(y_2_4, tf.sin(2.0 * phi))* radial_poly(r,5,4)
v_5_4_3 = tf.multiply(y_3_4, tf.sin(3.0 * phi))* radial_poly(r,5,4)
v_5_4_4 = tf.multiply(y_4_4, tf.sin(4.0 * phi))* radial_poly(r,5,4)
v_5_5_0 = tf.multiply(y_0_5, tf.sin(0.0 * phi))* radial_poly(r,5,5)
v_5_5_1 = tf.multiply(y_1_5, tf.sin(1.0 * phi))* radial_poly(r,5,5)
v_5_5_2 = tf.multiply(y_2_5, tf.sin(2.0 * phi))* radial_poly(r,5,5)
v_5_5_3 = tf.multiply(y_3_5, tf.sin(3.0 * phi))* radial_poly(r,5,5)
v_5_5_4 = tf.multiply(y_4_5, tf.sin(4.0 * phi))* radial_poly(r,5,5)
v_5_5_5 = tf.multiply(y_5_5, tf.sin(5.0 * phi))* radial_poly(r,5,5)

def my_func(x):
  # x will be a numpy array with the contents of the placeholder below
  return np.linalg.pinv(x)

inp = tf.placeholder(tf.float32)
inverse = tf.py_func(my_func, [inp], Tout = tf.float32)





v_0_0_0 = tf.multiply(y_0_0, tf.sin(0.0)) * radial_poly(r,0,0)
v_0_1_1 = tf.multiply(y_0_1, tf.sin(0.0)) * radial_poly(r,1,1)
v_1_1_1 = tf.multiply(y_1_1, tf.sin(phi)) * radial_poly(r,1,1)
v_0_0_2 = tf.multiply(y_0_2, tf.sin(0.0)) * radial_poly(r,0,2)
v_0_2_2 = tf.multiply(y_0_2, tf.sin(0.0)) * radial_poly(r,2,2)
v_1_2_2 = tf.multiply(y_1_2, tf.sin(phi)) * radial_poly(r,2,2)
v_2_2_2 = tf.multiply(y_2_2, tf.sin(2.0*phi)) * radial_poly(r,2,2)
v_0_1_3 = tf.multiply(y_0_1, tf.sin(0.0)) * radial_poly(r,1,3)
v_1_1_3 = tf.multiply(y_1_1, tf.sin(phi)) * radial_poly(r,1,3)
v_0_3_3 = tf.multiply(y_0_3, tf.sin(0.0)) * radial_poly(r,3,3)
v_1_3_3 = tf.multiply(y_1_3, tf.sin(phi)) * radial_poly(r,3,3)
v_2_3_3 = tf.multiply(y_2_3, tf.sin(2.0*phi)) * radial_poly(r,3,3)
v_3_3_3 = tf.multiply(y_3_3, tf.sin(3.0*phi)) * radial_poly(r,3,3)
v_0_0_4 = tf.multiply(y_0_0, tf.sin(0.0)) * radial_poly(r,0,4)
v_0_2_4 = tf.multiply(y_0_2, tf.sin(0.0)) * radial_poly(r,2,4)
v_1_2_4 = tf.multiply(y_1_2, tf.sin(phi)) * radial_poly(r,2,4)
v_2_2_4 = tf.multiply(y_2_2, tf.sin(2.0*phi)) * radial_poly(r,2,4)
v_0_4_4 = tf.multiply(y_0_4, tf.sin(0.0))* radial_poly(r,4,4)
v_1_4_4 = tf.multiply(y_1_4, tf.sin(phi)) * radial_poly(r,4,4)
v_2_4_4 = tf.multiply(y_2_4, tf.sin(2.0 * phi))* radial_poly(r,4,4)
v_3_4_4 = tf.multiply(y_3_4, tf.sin(3.0 * phi))* radial_poly(r,4,4)
v_4_4_4 = tf.multiply(y_4_4, tf.sin(4.0 * phi))* radial_poly(r,4,4)

################################################################################################
keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")


V = tf.concat([v_1_1_0, v_1_1_1, v_2_0_0, v_2_1_0, v_2_1_1, v_2_2_0, v_2_2_1, 
			v_2_2_2, v_3_0_0, v_3_1_0, v_3_1_1, v_3_2_0, v_3_2_1, v_3_2_2, v_3_3_0, v_3_3_1, v_3_3_2, v_3_3_3,
				v_4_0_0, v_4_1_0, v_4_1_1, v_4_2_0, v_4_2_1, v_4_2_2, v_4_3_0, v_4_3_1, v_4_3_2, v_4_3_3, v_4_4_0, v_4_4_1, v_4_4_2, v_4_4_3, v_4_4_4 ,
	v_5_0_0, v_5_1_0, v_5_1_1, v_5_2_0, v_5_2_1, v_5_2_2, v_5_3_0, v_5_3_1, v_5_3_2, v_5_3_3, v_5_4_0, v_5_4_1, v_5_4_2, v_5_4_3, v_5_4_4, v_5_5_0, v_5_5_1, v_5_5_2, v_5_5_3, v_5_5_4, v_5_5_5],  axis=1)


U = tf.concat([u_1_1_0, u_1_1_1, u_2_0_0, u_2_1_0, u_2_1_1, u_2_2_0, u_2_2_1, 
                        u_2_2_2, u_3_0_0, u_3_1_0, u_3_1_1, u_3_2_0, u_3_2_1, u_3_2_2, u_3_3_0, u_3_3_1, u_3_3_2, u_3_3_3,
                                u_4_0_0, u_4_1_0, u_4_1_1, u_4_2_0, u_4_2_1, u_4_2_2, u_4_3_0, u_4_3_1, u_4_3_2, u_4_3_3, u_4_4_0, u_4_4_1, u_4_4_2, u_4_4_3, u_4_4_4 ,
        u_5_0_0, u_5_1_0, u_5_1_1, u_5_2_0, u_5_2_1, u_5_2_2, u_5_3_0, u_5_3_1, u_5_3_2, u_5_3_3, u_5_4_0, u_5_4_1, u_5_4_2, u_5_4_3, u_5_4_4, u_5_5_0, u_5_5_1, u_5_5_2, u_5_5_3, u_5_5_4, u_5_5_5],  axis=1)

X_ = tf.concat([U,V] ,  axis=1)

X__1 = tf.slice(X_, [0,0], [tf.shape(r_one)[0], -1])
X__2 = tf.slice(X_, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X__3 = tf.slice(X_, [tf.shape(r_two)[0] + tf.shape(r_one)[0], 0], [tf.shape(r_three)[0], -1])


annl_ = [2.1333333333333333, 1.0333333333333332, 0., 10.366666666666667, 4.433333333333334, -25.8547619047619, 37.576190476190476, 14.928571428571429, 9.537096774193543, -0.0008098186006335328, 0., -83.36210317460316,121.62103174603173, 45.901587301587305, 41.00577956989247, -0.0067672460416487445, 0., -6.383794139984022*1e-14, -2.6612772832010517*1e-14, -8.534031799876057*1e-16,-253.95238095238093, 371.66666666666663, 134.9462632275132, 151.2935483870968, -0.03522603000419955, 0., 2.3413729128011482, -5.912803382502408*1e-13, -1.7302328401196382*1e-13, -7.32051053974624*1e-15, 3.005077588075395*1e-11, -7.537707126831663*1e-12, -1.1392382772030042*1e-12, -6.346607025376294*1e-14] 


annl = [2.1333333333333333, 1.0333333333333332, 0., 10.366666666666667, 4.433333333333334, -25.8547619047619, 37.576190476190476, 14.928571428571429, 9.537096774193543, -0.0008098186006335328, 0., -83.36210317460316,121.62103174603173, 45.901587301587305, 41.00577956989247, -0.0067672460416487445, 0., -6.383794139984022*0, -2.6612772832010517*0, -8.534031799876057*0,-253.95238095238093, 371.66666666666663, 134.9462632275132, 151.2935483870968, -0.03522603000419955, 0., 2.3413729128011482, -5.912803382502408*0, -1.7302328401196382*0, -7.32051053974624*0, 3.005077588075395*0, -7.537707126831663*0, -1.1392382772030042*0, -6.346607025376294*0]


#annl = [float(i)/max(annl_) for i in annl_]

def slice_freq(omega, i):
	om = tf.slice(omega, [0,i,0],[-1,1,1])
	return om 

def e_(r_,n,l, scope):
	with tf.variable_scope(scope):
		e__, p_1 = tf.while_loop(lambda x,i: i < int(n+1),
    	lambda x, i:(x+tf.pow(int(n-l+1)*r_,i)/tf.exp(tf.lgamma(i + 1)) ,i+1),(r_, 0.0))
		return e__

def e__(r_,n,l, scope):
	return tf.exp((n)*r_)
	


print(annl[32])

def b(r_, alpha, beta, scope):
	with tf.variable_scope(scope):
		b1110 = annl[1]*(e_(r_,1.0,1.0,scope) - e_(r_,1.0,1.0,scope)) * Y(1,0,alpha,beta)
		b2110 = annl[4]*(e_(r_,2.0,1.0,scope) - e_(r_,1.0,1.0,scope)) * Y(1,0,alpha,beta)
		b2111 = annl[4]*(e_(r_,2.0,1.0,scope) - e_(r_,1.0,1.0,scope)) * Y(1,1,alpha,beta)
		b3110 = annl[6]*(e_(r_,3.0,2.0,scope) - e_(r_,2.0,1.0,scope)) * Y(1,0,alpha,beta) 
		b3111 = annl[6]*(e_(r_,3.0,1.0,scope) - e_(r_,1.0,1.0,scope))*Y(1,1,alpha,beta) 
		b3200 = annl[7]*(e_(r_,3.0,0.0,scope) - e_(r_,2.0,0.0,scope))*Y(0,0,alpha,beta) 
		b3210 = annl[8]*(e_(r_,3.0,1.0,scope) - e_(r_,2.0,1.0,scope))*Y(1,0,alpha,beta) 
		b3211 = annl[8]*(e_(r_,3.0,1.0,scope) - e_(r_,2.0,1.0,scope)) *Y(1,1,alpha,beta)
		b3220 = annl[9]*(e_(r_,3.0,2.0,scope) - e_(r_,2.0,2.0,scope)) *Y(2,0,alpha,beta)
		b3221 = annl[9]*(e_(r_,3.0,2.0,scope) - e_(r_,2.0,2.0,scope)) *Y(2,1,alpha,beta)
		b3222 = annl[9]*(e_(r_,3.0,2.0,scope) - e_(r_,2.0,2.0,scope)) *Y(2,2,alpha,beta)
		b4110 = annl[11]*(e_(r_,4.0,1.0,scope) - e_(r_,1.0,1.0,scope)) *Y(1,0,alpha,beta)
		b4200 = annl[12]*(e_(r_,4.0,0.0,scope) - e_(r_,2.0,0.0,scope)) *Y(0,0,alpha,beta)
		b4210 = annl[13]*(e_(r_,4.0,1.0,scope) - e_(r_,2.0,1.0,scope)) *Y(1,0,alpha,beta)
		b4211 = annl[13]*(e_(r_,4.0,1.0,scope) - e_(r_,2.0,1.0,scope)) *Y(1,1,alpha,beta)
		b4221 = annl[14]*(e_(r_,4.0,2.0,scope) - e_(r_,2.0,2.0,scope)) *Y(2,1,alpha,beta)
		b4222 = annl[14]*(e_(r_,4.0,2.0,scope) - e_(r_,2.0,2.0,scope))*Y(2,2,alpha,beta)
		b4300 = annl[15]*(e_(r_,4.0,0.0,scope) - e_(r_,3.0,0.0,scope)) *Y(0,0,alpha,beta)
		b4310 = annl[16]*(e_(r_,4.0,1.0,scope) - e_(r_,3.0,1.0,scope))*Y(1,0,alpha,beta)
		b4311 = annl[16]*(e_(r_,4.0,1.0,scope) - e_(r_,3.0,1.0,scope))*Y(1,1,alpha,beta)
		b4320 = annl[17]*(e_(r_,4.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Y(2,0,alpha,beta)
		b4321 = annl[17]*(e_(r_,4.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Y(2,1,alpha,beta)
		b4322 = annl[17]*(e_(r_,4.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Y(2,2,alpha,beta)
		b4330 = annl[18]*(e_(r_,4.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Y(3,0,alpha,beta)
		b4331 = annl[18]*(e_(r_,4.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Y(3,1,alpha,beta)
		b4332 = annl[18]*(e_(r_,4.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Y(3,2,alpha,beta)
		b4333 = annl[18]*(e_(r_,4.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Y(3,3,alpha,beta)
		b5110 = annl[20]*(e_(r_,5.0,1.0,scope) - e_(r_,1.0,1.0,scope))*Y(1,0,alpha,beta)
		b5111 = annl[20]*(e_(r_,5.0,1.0,scope) - e_(r_,1.0,1.0,scope))*Y(1,1,alpha,beta)
		b5200 =  annl[21]*(e_(r_,5.0,0.0,scope) - e_(r_,2.0,0.0,scope))*Y(0,0,alpha,beta)
		b5210 =  annl[22]*(e_(r_,5.0,1.0,scope) - e_(r_,2.0,1.0,scope))*Y(1,0,alpha,beta)
		b5211 =  annl[22]*(e_(r_,5.0,1.0,scope) - e_(r_,2.0,1.0,scope))*Y(1,1,alpha,beta)
		b5220 =  annl[23]*(e_(r_,5.0,2.0,scope) - e_(r_,2.0,2.0,scope))*Y(2,0,alpha,beta)
		b5221 =  annl[23]*(e_(r_,5.0,2.0,scope) - e_(r_,2.0,2.0,scope))*Y(2,1,alpha,beta)
		b5222 =  annl[23]*(e_(r_,5.0,2.0,scope) - e_(r_,2.0,2.0,scope))*Y(2,2,alpha,beta)
		b5300 =  annl[24]*(e_(r_,5.0,0.0,scope) - e_(r_,3.0,0.0,scope))*Y(0,0,alpha,beta)
		b5310 = annl[25]*(e_(r_,5.0,1.0,scope) - e_(r_,3.0,1.0,scope))*Y(1,0,alpha,beta)
		b5311 = annl[25]*(e_(r_,5.0,1.0,scope) - e_(r_,3.0,1.0,scope))*Y(1,1,alpha,beta)
		b5320 = annl[26]*(e_(r_,5.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Y(2,0,alpha,beta)
		b5321 = annl[26]*(e_(r_,5.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Y(2,1,alpha,beta)
		b5322 = annl[26]*(e_(r_,5.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Y(2,2,alpha,beta)
		b5330 = annl[27]*(e_(r_,5.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Y(3,0,alpha,beta)
		b5331 = annl[27]*(e_(r_,5.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Y(3,1,alpha,beta)
		b5332 = annl[27]*(e_(r_,5.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Y(3,2,alpha,beta)
		b5333 = annl[27]*(e_(r_,5.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Y(3,3,alpha,beta)
		b5400 = annl[28]*(e_(r_,5.0,0.0,scope) - e_(r_,4.0,0.0,scope))*Y(0,0,alpha,beta)
		b5410 = annl[29]*(e_(r_,5.0,1.0,scope) - e_(r_,4.0,1.0,scope))*Y(1,0,alpha,beta)
		b5411 = annl[29]*(e_(r_,5.0,1.0,scope) - e_(r_,4.0,1.0,scope))*Y(1,1,alpha,beta)
		b5420 = annl[30]*(e_(r_,5.0,2.0,scope) - e_(r_,4.0,2.0,scope))*Y(2,0,alpha,beta)
		b5421 = annl[30]*(e_(r_,5.0,2.0,scope) - e_(r_,4.0,2.0,scope))*Y(2,1,alpha,beta)
		b5422 = annl[30]*(e_(r_,5.0,2.0,scope) - e_(r_,4.0,2.0,scope))*Y(2,2,alpha,beta)
		b5430 = annl[31]*(e_(r_,5.0,3.0,scope) - e_(r_,4.0,3.0,scope))*Y(3,0,alpha,beta)
		b5431 = annl[31]*(e_(r_,5.0,3.0,scope) - e_(r_,4.0,3.0,scope))*Y(3,1,alpha,beta)
		b5432 = annl[31]*(e_(r_,5.0,3.0,scope) - e_(r_,4.0,3.0,scope))*Y(3,2,alpha,beta)
		b5433 = annl[31]*(e_(r_,5.0,3.0,scope) - e_(r_,4.0,3.0,scope))*Y(3,3,alpha,beta)
		b5440 = annl[32]*(e_(r_,5.0,4.0,scope) - e_(r_,4.0,4.0,scope))*Y(4,0,alpha,beta)
		b5441 = annl[32]*(e_(r_,5.0,4.0,scope) - e_(r_,4.0,4.0,scope))*Y(4,1,alpha,beta)
		b5442 = annl[32]*(e_(r_,5.0,4.0,scope) - e_(r_,4.0,4.0,scope))*Y(4,2,alpha,beta)
		b5443 = annl[32]*(e_(r_,5.0,4.0,scope) - e_(r_,4.0,4.0,scope))*Y(4,3,alpha,beta)
		b5444 = annl[32]*(e_(r_,5.0,4.0,scope) - e_(r_,4.0,4.0,scope))*Y(4,4,alpha,beta)


		bi1110 = annl[1]*(e_(r_,1.0,1.0,scope) - e_(r_,1.0,1.0,scope)) * Yi(1,0,alpha,beta)
                bi2110 = annl[4]*(e_(r_,2.0,1.0,scope) - e_(r_,1.0,1.0,scope)) * Yi(1,0,alpha,beta)
                bi2111 = annl[4]*(e_(r_,2.0,1.0,scope) - e_(r_,1.0,1.0,scope)) * Yi(1,1,alpha,beta)
                bi3110 =  annl[6]*(e_(r_,3.0,2.0,scope) - e_(r_,2.0,1.0,scope)) * Yi(1,0,alpha,beta)
                bi3111 = annl[6]*(e_(r_,3.0,1.0,scope) - e_(r_,1.0,1.0,scope))*Yi(1,1,alpha,beta)
                bi3200 = annl[7]*(e_(r_,3.0,0.0,scope) - e_(r_,2.0,0.0,scope))*Yi(0,0,alpha,beta)
                bi3210 = annl[8]*(e_(r_,3.0,1.0,scope) - e_(r_,2.0,1.0,scope))*Yi(1,0,alpha,beta)
                bi3211 = annl[8]*(e_(r_,3.0,1.0,scope) - e_(r_,2.0,1.0,scope)) *Yi(1,1,alpha,beta)
                bi3220 = annl[9]*(e_(r_,3.0,2.0,scope) - e_(r_,2.0,2.0,scope)) *Yi(2,0,alpha,beta)
                bi3221 = annl[9]*(e_(r_,3.0,2.0,scope) - e_(r_,2.0,2.0,scope)) *Yi(2,1,alpha,beta)
                bi3222 = annl[9]*(e_(r_,3.0,2.0,scope) - e_(r_,2.0,2.0,scope)) *Yi(2,2,alpha,beta)
                bi4110 = annl[11]*(e_(r_,4.0,1.0,scope) - e_(r_,1.0,1.0,scope)) *Yi(1,0,alpha,beta)
                bi4200 = annl[12]*(e_(r_,4.0,0.0,scope) - e_(r_,2.0,0.0,scope)) *Yi(0,0,alpha,beta)
                bi4210 = annl[13]*(e_(r_,4.0,1.0,scope) - e_(r_,2.0,1.0,scope)) *Yi(1,0,alpha,beta)
                bi4211 = annl[13]*(e_(r_,4.0,1.0,scope) - e_(r_,2.0,1.0,scope)) *Yi(1,1,alpha,beta)
                bi4221 = annl[14]*(e_(r_,4.0,2.0,scope) - e_(r_,2.0,2.0,scope)) *Yi(2,1,alpha,beta)
                bi4222 = annl[14]*(e_(r_,4.0,2.0,scope) - e_(r_,2.0,2.0,scope))*Yi(2,2,alpha,beta)
                bi4300 = annl[15]*(e_(r_,4.0,0.0,scope) - e_(r_,3.0,0.0,scope)) *Yi(0,0,alpha,beta)
                bi4310 = annl[16]*(e_(r_,4.0,1.0,scope) - e_(r_,3.0,1.0,scope))*Yi(1,0,alpha,beta)
                bi4311 = annl[16]*(e_(r_,4.0,1.0,scope) - e_(r_,3.0,1.0,scope))*Yi(1,1,alpha,beta)
                bi4320 =  annl[17]*(e_(r_,4.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Yi(2,0,alpha,beta)
                bi4321 = annl[17]*(e_(r_,4.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Yi(2,1,alpha,beta)
                bi4322 = annl[17]*(e_(r_,4.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Yi(2,2,alpha,beta)
                bi4330 = annl[18]*(e_(r_,4.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Yi(3,0,alpha,beta)
                bi4331 = annl[18]*(e_(r_,4.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Yi(3,1,alpha,beta)
                bi4332 = annl[18]*(e_(r_,4.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Yi(3,2,alpha,beta)
                bi4333 = annl[18]*(e_(r_,4.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Yi(3,3,alpha,beta)
                bi5110 = annl[20]*(e_(r_,5.0,1.0,scope) - e_(r_,1.0,1.0,scope))*Yi(1,0,alpha,beta)
                bi5111 = annl[20]*(e_(r_,5.0,1.0,scope) - e_(r_,1.0,1.0,scope))*Yi(1,1,alpha,beta)
                bi5200 =  annl[21]*(e_(r_,5.0,0.0,scope) - e_(r_,2.0,0.0,scope))*Yi(0,0,alpha,beta)
                bi5210 =  annl[22]*(e_(r_,5.0,1.0,scope) - e_(r_,2.0,1.0,scope))*Yi(1,0,alpha,beta)
                bi5211 =  annl[22]*(e_(r_,5.0,1.0,scope) - e_(r_,2.0,1.0,scope))*Yi(1,1,alpha,beta)
                bi5220 = annl[23]*(e_(r_,5.0,2.0,scope) - e_(r_,2.0,2.0,scope))*Yi(2,0,alpha,beta)
                bi5221 = annl[23]*(e_(r_,5.0,2.0,scope) - e_(r_,2.0,2.0,scope))*Yi(2,1,alpha,beta)
                bi5222 = annl[23]*(e_(r_,5.0,2.0,scope) - e_(r_,2.0,2.0,scope))*Yi(2,2,alpha,beta)
                bi5300 = annl[24]*(e_(r_,5.0,0.0,scope) - e_(r_,3.0,0.0,scope))*Yi(0,0,alpha,beta)
                bi5310 = annl[25]*(e_(r_,5.0,1.0,scope) - e_(r_,3.0,1.0,scope))*Yi(1,0,alpha,beta)
                bi5311 = annl[25]*(e_(r_,5.0,1.0,scope) - e_(r_,3.0,1.0,scope))*Yi(1,1,alpha,beta)
                bi5320 = annl[26]*(e_(r_,5.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Yi(2,0,alpha,beta)
                bi5321 = annl[26]*(e_(r_,5.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Yi(2,1,alpha,beta)
		bi5322 = annl[26]*(e_(r_,5.0,2.0,scope) - e_(r_,3.0,2.0,scope))*Yi(2,2,alpha,beta)
                bi5330 = annl[27]*(e_(r_,5.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Yi(3,0,alpha,beta)
                bi5331 = annl[27]*(e_(r_,5.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Yi(3,1,alpha,beta)
                bi5332 = annl[27]*(e_(r_,5.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Yi(3,2,alpha,beta)
                bi5333 = annl[27]*(e_(r_,5.0,3.0,scope) - e_(r_,3.0,3.0,scope))*Yi(3,3,alpha,beta)
                bi5400 = annl[28]*(e_(r_,5.0,0.0,scope) - e_(r_,4.0,0.0,scope))*Yi(0,0,alpha,beta)
                bi5410 = annl[29]*(e_(r_,5.0,1.0,scope) - e_(r_,4.0,1.0,scope))*Yi(1,0,alpha,beta)
                bi5411 = annl[29]*(e_(r_,5.0,1.0,scope) - e_(r_,4.0,1.0,scope))*Yi(1,1,alpha,beta)
                bi5420 = annl[30]*(e_(r_,5.0,2.0,scope) - e_(r_,4.0,2.0,scope))*Yi(2,0,alpha,beta)
                bi5421 = annl[30]*(e_(r_,5.0,2.0,scope) - e_(r_,4.0,2.0,scope))*Yi(2,1,alpha,beta)
                bi5422 = annl[30]*(e_(r_,5.0,2.0,scope) - e_(r_,4.0,2.0,scope))*Yi(2,2,alpha,beta)
                bi5430 = annl[31]*(e_(r_,5.0,3.0,scope) - e_(r_,4.0,3.0,scope))*Yi(3,0,alpha,beta)
                bi5431 = annl[31]*(e_(r_,5.0,3.0,scope) - e_(r_,4.0,3.0,scope))*Yi(3,1,alpha,beta)
                bi5432 = annl[31]*(e_(r_,5.0,3.0,scope) - e_(r_,4.0,3.0,scope))*Yi(3,2,alpha,beta)
                bi5433 = annl[31]*(e_(r_,5.0,3.0,scope) - e_(r_,4.0,3.0,scope))*Yi(3,3,alpha,beta)
                bi5440 = annl[32]*(e_(r_,5.0,4.0,scope) - e_(r_,4.0,4.0,scope))*Yi(4,0,alpha,beta)
                bi5441 = annl[32]*(e_(r_,5.0,4.0,scope) - e_(r_,4.0,4.0,scope))*Yi(4,1,alpha,beta)
                bi5442 = annl[32]*(e_(r_,5.0,4.0,scope) - e_(r_,4.0,4.0,scope))*Yi(4,2,alpha,beta)
                bi5443 = annl[32]*(e_(r_,5.0,4.0,scope) - e_(r_,4.0,4.0,scope))*Yi(4,3,alpha,beta)
                bi5444 = annl[32]*(e_(r_,5.0,4.0,scope) - e_(r_,4.0,4.0,scope))*Yi(4,4,alpha,beta)
		
		print(e_(r_,5.0,4.0,scope))
		print(e_(r_,5.0,3.0,scope))
		print(Yi(4,4,alpha,beta))
		print(annl[32])
		print(bi5444)	
	
		b_list=tf.expand_dims(tf.concat([b1110, b2110,b2111,b3110,b3111,b3200,b3210,b3211,b3220,b3221,b3222,b4110,b4200,b4210,b4211,b4221,b4222,b4300,b4310,
                   b4311,b4320,b4321,b4322,b4330,b4331,b4332,b4333,b5110,b5111,b5200,b5210,b5211,b5220,b5221,b5222,b5300,b5310,b5311,b5320,
                   b5321,b5322, b5330,b5331,b5332,b5333,b5400,b5410,b5411 ,b5420,b5421,b5422,b5430,b5431,b5432,b5433,b5440,b5441,b5442,
                   b5443,b5444], axis = 1), axis = 0)
		b_listi=tf.expand_dims(tf.concat([bi1110, bi2110,bi2111,bi3110,bi3111,bi3200,bi3210,bi3211,bi3220,bi3221,bi3222,bi4110,bi4200,bi4210,bi4211,bi4221,bi4222,bi4300,bi4310,
                   bi4311,bi4320,bi4321,bi4322,bi4330,bi4331,bi4332,bi4333,bi5110,bi5111,bi5200,bi5210,bi5211,bi5220,bi5221,bi5222,bi5300,bi5310,bi5311,bi5320,
                   bi5321,bi5322, bi5330,bi5331,bi5332,bi5333,bi5400,bi5410,bi5411 ,bi5420,bi5421,bi5422,bi5430,bi5431,bi5432,bi5433,bi5440,bi5441,bi5442,
                   bi5443,bi5444], axis = 1), axis = 0)
		
		print(b_list)
		print(b_listi)
		b_list_complex = tf.complex(b_list,b_listi)
		return b_list_complex
	
def omega(cf, cg, scope):
#	tf.concat([b1110, b2110,b2111,b3110,b3111,b3200,b3210,b3211,b3220,b3221,b3222,b4110,b4200,b4210,b4211,b4221,b4222,b4300,b4310,
#		   b4311,b4320,b4321,b4322,b4330,b4331,b4332,b4333,b5110,b5111,b5200,b5210,b5211,b5220,b5221,b5222,b5300,b5310,b5311,b5320,
	with tf.variable_scope(scope):
		b1110 =  slice_freq(cf,0)*slice_freq(cg,0)
        	b2110 =  slice_freq(cf,3)*slice_freq(cg,0)
        	b2111 =  slice_freq(cf,4)*slice_freq(cg,1)
        	b3110 =  slice_freq(cf,3)*slice_freq(cg,0)
        	b3111 =  slice_freq(cf,10)*slice_freq(cg,1)
        	b3200 = slice_freq(cf,8)*slice_freq(cg,2)
        	b3210 = slice_freq(cf,9)*slice_freq(cg,3)
        	b3211 = slice_freq(cf,10)*slice_freq(cg,4)
        	b3220 = slice_freq(cf,11)*slice_freq(cg,5)
        	b3221 = slice_freq(cf,12)*slice_freq(cg,6)
        	b3222 = slice_freq(cf,13)*slice_freq(cg,7)
        	b4110 = slice_freq(cf,19)*slice_freq(cg,0)
        	b4200 = slice_freq(cf,18)*slice_freq(cg,2)
        	b4210 = slice_freq(cf,19)*slice_freq(cg,3)
        	b4211 = slice_freq(cf,20)*slice_freq(cg,4)
        	b4221 = slice_freq(cf,22)*slice_freq(cg,6)
        	b4222 = slice_freq(cf,23)*slice_freq(cg,7)
        	b4300 = slice_freq(cf,18)*slice_freq(cg,8)
        	b4310 = slice_freq(cf,19)*slice_freq(cg,9)
        	b4311 = slice_freq(cf,20)*slice_freq(cg,10)
        	b4320 = slice_freq(cf,21)*slice_freq(cg,11)
        	b4321 = slice_freq(cf,22)*slice_freq(cg,12)
        	b4322 = slice_freq(cf,23)*slice_freq(cg,13)
        	b4330 = slice_freq(cf,24)*slice_freq(cg,14)
        	b4331 = slice_freq(cf,25)*slice_freq(cg,15)
        	b4332 = slice_freq(cf,26)*slice_freq(cg,16)
        	b4333 = slice_freq(cf,27)*slice_freq(cg,17)
        	b5110 = slice_freq(cf,34)*slice_freq(cg,0)
        	b5111 = slice_freq(cf,35)*slice_freq(cg,1)
        	b5200 = slice_freq(cf,33)*slice_freq(cg,2)
        	b5210 = slice_freq(cf,34)*slice_freq(cg,3)
        	b5211 = slice_freq(cf,35)*slice_freq(cg,4)
        	b5220 = slice_freq(cf,36)*slice_freq(cg,5)
        	b5221 = slice_freq(cf,37)*slice_freq(cg,6)
        	b5222 = slice_freq(cf,38)*slice_freq(cg,7)
        	b5300 = slice_freq(cf,33)*slice_freq(cg,8)
        	b5310 = slice_freq(cf,34)*slice_freq(cg,9)
        	b5311 = slice_freq(cf,35)*slice_freq(cg,10)
        	b5320 = slice_freq(cf,36)*slice_freq(cg,11)
        	b5321 = slice_freq(cf,37)*slice_freq(cg,12)
        	b5322 = slice_freq(cf,38)*slice_freq(cg,13)
		b5330 = slice_freq(cf,39)*slice_freq(cg,14)
        	b5331 = slice_freq(cf,40)*slice_freq(cg,15)
        	b5332 = slice_freq(cf,41)*slice_freq(cg,16)
        	b5333 = slice_freq(cf,42)*slice_freq(cg,17)
        	b5400 = slice_freq(cf,33)*slice_freq(cg,18)
        	b5410 = slice_freq(cf,34)*slice_freq(cg,19)
        	b5411 = slice_freq(cf,35)*slice_freq(cg,20)
        	b5420 = slice_freq(cf,36)*slice_freq(cg,21)
        	b5421 = slice_freq(cf,37)*slice_freq(cg,22)
        	b5422 = slice_freq(cf,38)*slice_freq(cg,23)
        	b5430 = slice_freq(cf,39)*slice_freq(cg,24)
        	b5431 = slice_freq(cf,40)*slice_freq(cg,25)
        	b5432 = slice_freq(cf,41)*slice_freq(cg,26)
        	b5433 = slice_freq(cf,42)*slice_freq(cg,27)
        	b5440 = slice_freq(cf,43)*slice_freq(cg,28)
        	b5441 = slice_freq(cf,44)*slice_freq(cg,29)
        	b5442 = slice_freq(cf,45)*slice_freq(cg,30)
        	b5443 = slice_freq(cf,46)*slice_freq(cg,31)
        	b5444 = slice_freq(cf,47)*slice_freq(cg,32)
		

		bi1110 = slice_freq(cf,54+0)*slice_freq(cg,54+0)
                bi2110 = slice_freq(cf,54+3)*slice_freq(cg,54+0)
                bi2111 = slice_freq(cf,54+4)*slice_freq(cg,54+1)
                bi3110 = slice_freq(cf,54+3)*slice_freq(cg,54+0)
                bi3111 = slice_freq(cf,54+10)*slice_freq(cg,54+1)
                bi3200 = slice_freq(cf,54+8)*slice_freq(cg,54+2)
                bi3210 = slice_freq(cf,54+9)*slice_freq(cg,54+3)
                bi3211 = slice_freq(cf,54+10)*slice_freq(cg,54+4)
                bi3220 = slice_freq(cf,54+11)*slice_freq(cg,54+5)
                bi3221 = slice_freq(cf,54+12)*slice_freq(cg,54+6)
                bi3222 = slice_freq(cf,54+13)*slice_freq(cg,54+7)
                bi4110 = slice_freq(cf,54+19)*slice_freq(cg,54+0)
                bi4200 = slice_freq(cf,54+18)*slice_freq(cg,54+2)
                bi4210 = slice_freq(cf,54+19)*slice_freq(cg,54+3)
                bi4211 = slice_freq(cf,54+20)*slice_freq(cg,54+4)
                bi4221 = slice_freq(cf,54+22)*slice_freq(cg,54+6)
                bi4222 = slice_freq(cf,54+23)*slice_freq(cg,54+7)
                bi4300 = slice_freq(cf,54+18)*slice_freq(cg,54+8)
                bi4310 = slice_freq(cf,54+19)*slice_freq(cg,54+9)
                bi4311 = slice_freq(cf,54+20)*slice_freq(cg,54+10)
                bi4320 = slice_freq(cf,54+21)*slice_freq(cg,54+11)
                bi4321 = slice_freq(cf,54+22)*slice_freq(cg,54+12)
                bi4322 = slice_freq(cf,54+23)*slice_freq(cg,54+13)
                bi4330 = slice_freq(cf,54+24)*slice_freq(cg,54+14)
                bi4331 = slice_freq(cf,54+25)*slice_freq(cg,54+15)
                bi4332 = slice_freq(cf,54+26)*slice_freq(cg,54+16)
                bi4333 = slice_freq(cf,54+27)*slice_freq(cg,54+17)
                bi5110 = slice_freq(cf,54+34)*slice_freq(cg,54+0)
                bi5111 = slice_freq(cf,54+35)*slice_freq(cg,54+1)
                bi5200 = slice_freq(cf,54+33)*slice_freq(cg,54+2)
                bi5210 = slice_freq(cf,54+34)*slice_freq(cg,54+3)
                bi5211 = slice_freq(cf,54+35)*slice_freq(cg,54+4)
                bi5220 = slice_freq(cf,54+36)*slice_freq(cg,54+5)
                bi5221 = slice_freq(cf,54+37)*slice_freq(cg,54+6)
                bi5222 = slice_freq(cf,54+38)*slice_freq(cg,54+7)
                bi5300 = slice_freq(cf,54+33)*slice_freq(cg,54+8)
                bi5310 = slice_freq(cf,54+34)*slice_freq(cg,54+9)
                bi5311 = slice_freq(cf,54+35)*slice_freq(cg,54+10)
		bi5320 = slice_freq(cf,54+36)*slice_freq(cg,54+11)
                bi5321 = slice_freq(cf,54+37)*slice_freq(cg,54+12)
                bi5322 = slice_freq(cf,54+38)*slice_freq(cg,54+13)
                bi5330 = slice_freq(cf,54+39)*slice_freq(cg,54+14)
                bi5331 = slice_freq(cf,54+40)*slice_freq(cg,54+15)
                bi5332 = slice_freq(cf,54+41)*slice_freq(cg,54+16)
                bi5333 = slice_freq(cf,54+42)*slice_freq(cg,54+17)
                bi5400 = slice_freq(cf,54+33)*slice_freq(cg,54+18)
                bi5410 = slice_freq(cf,54+34)*slice_freq(cg,54+19)
                bi5411 = slice_freq(cf,54+35)*slice_freq(cg,54+20)
                bi5420 = slice_freq(cf,54+36)*slice_freq(cg,54+21)
                bi5421 = slice_freq(cf,54+37)*slice_freq(cg,54+22)
                bi5422 = slice_freq(cf,54+38)*slice_freq(cg,54+23)
                bi5430 = slice_freq(cf,54+39)*slice_freq(cg,54+24)
                bi5431 = slice_freq(cf,54+40)*slice_freq(cg,54+25)
                bi5432 = slice_freq(cf,54+41)*slice_freq(cg,54+26)
                bi5433 = slice_freq(cf,54+42)*slice_freq(cg,54+27)
                bi5440 = slice_freq(cf,54+43)*slice_freq(cg,54+28)
                bi5441 = slice_freq(cf,54+44)*slice_freq(cg,54+29)
                bi5442 = slice_freq(cf,54+45)*slice_freq(cg,54+30)
                bi5443 = slice_freq(cf,54+46)*slice_freq(cg,54+31)
                bi5444 = slice_freq(cf,54+47)*slice_freq(cg,54+32)



		omega_list=tf.concat([b1110, b2110,b2111,b3110,b3111,b3200,b3210,b3211,b3220,b3221,b3222,b4110,b4200,b4210,b4211,b4221,b4222,b4300,b4310,
                   b4311,b4320,b4321,b4322,b4330,b4331,b4332,b4333,b5110,b5111,b5200,b5210,b5211,b5220,b5221,b5222,b5300,b5310,b5311,b5320,
		   b5321,b5322, b5330,b5331,b5332,b5333,b5400,b5410,b5411 ,b5420,b5421,b5422,b5430,b5431,b5432,b5433,b5440,b5441,b5442,
		   b5443,b5444], axis = 2)


		omega_listi = tf.concat([bi1110, bi2110,bi2111,bi3110,bi3111,bi3200,bi3210,bi3211,bi3220,bi3221,bi3222,bi4110,bi4200,bi4210,bi4211,bi4221,bi4222,bi4300,bi4310,
                   bi4311,bi4320,bi4321,bi4322,bi4330,bi4331,bi4332,bi4333,bi5110,bi5111,bi5200,bi5210,bi5211,bi5220,bi5221,bi5222,bi5300,bi5310,bi5311,bi5320,
                   bi5321,bi5322, bi5330,bi5331,bi5332,bi5333,bi5400,bi5410,bi5411 ,bi5420,bi5421,bi5422,bi5430,bi5431,bi5432,bi5433,bi5440,bi5441,bi5442,
                   bi5443,bi5444], axis = 2)
		
		print(omega_list)
		print(omega_listi)
		omega_list_complex = tf.complex(omega_list,omega_listi)
		return omega_list_complex


#alpha_val= tf.reshape(tf.tile(tf.concat([tf.fill([1,37],0.0), tf.fill([1,37],10.0), tf.fill([1,37],20.0), tf.fill([1,37],30.0), tf.fill([1,37],40.0), tf.fill([1,37],50.0), tf.fill([1,37],60.0), tf.fill([1,37],70.0), tf.fill([1,37],80.0), tf.fill([1,37],90.0), tf.fill([1,37],100.0), tf.fill([1,37],110.0),tf.fill([1,37],120.0),tf.fill([1,37],130.0), tf.fill([1,37],140.0), tf.fill([1,37],150.0), tf.fill([1,37],160.0), tf.fill([1,37],170.0), tf.fill([1,37],180.0)], axis = 1),[1,11]),[7733,1])

#beta_val = tf.reshape(tf.to_float(tf.tile(tf.reshape(tf.range(0,370,10),[1,37]),[1,19*11])),[7733,1])

r_add =  tf.slice(r, [0,0], [110, -1])
theta_add = tf.slice(theta, [0,0], [110, -1])
phi_add = tf.slice(phi, [0,0], [110, -1])

trans_val = tf.reshape(tf.concat([tf.fill([1,10],0.0), tf.fill([1,10],0.1),tf.fill([1,10],0.2),tf.fill([1,10],0.3),tf.fill([1,10],0.4),tf.fill([1,10],0.5),tf.fill([1,10],0.6),tf.fill([1,10],0.7),tf.fill([1,10],0.8),tf.fill([1,10],0.9),tf.fill([1,10],10.0)],axis =1  ),[110,1])

alpha_val =  tf.get_variable("a_alpha", [110,1])
beta_val =  tf.get_variable("b_alpha", [110,1])

b_complex =  b(r_add+trans_val  , theta_add+alpha_val , phi_add +beta_val, "sampling")

#r_val = tf.range(0, 1, 0.01)

r_val = tf.get_variable("a_r", [1,100])

X_10  = 0.0001 * tf.transpose(X__1,[1,0] )
X_20  = 0.0001 * tf.transpose(X__2,[1,0] )
X_30  = 0.0001 * tf.transpose(X__3,[1,0] )

X_cal_1, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(108) + 1.0/4.0 * tf.matmul(tf.eye(108)-tf.matmul(x,X__1),tf.matmul(3.0*tf.eye(108)-tf.matmul(x,X__1),3.0*tf.eye(108)-tf.matmul(x,X__1))),x),i+1)
    ,(X_10, 0))

X_cal_2, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2),3.0*tf.eye(44)-tf.matmul(x,X__2))),x),i+1)
    ,(X_20, 0))

X_cal_3, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3),3.0*tf.eye(44)-tf.matmul(x,X__3))),x),i+1)
    ,(X_30, 0))
X_cal_1_ =1.0 *  tf.div(
   tf.subtract(
      X_cal_1,
      tf.reduce_min(X_cal_1, keepdims = True)
   ),
   tf.subtract(
      tf.reduce_max(X_cal_1, keepdims = True),
      tf.reduce_min(X_cal_1, keepdims = True)
   )
)


X_cal_inv = tf.placeholder(tf.float32)
r_in =  tf.placeholder(tf.float32)


C_1t = tf.tile(tf.expand_dims(tf.matmul(X_cal_inv,r), axis =0), [16,1,1])

C_1 = tf.complex(tf.slice(C_1t,[0,0,0],[-1,54,-1]),tf.slice(C_1t,[0,54,0],[-1,54,-1]))

C_2 = tf.tile(tf.expand_dims(tf.matmul(X_cal_2, r_two), axis =0), [64,1,1])
C_3 = tf.tile(tf.expand_dims(tf.matmul(X_cal_3,r_three), axis =0), [64,1,1])


###############################################################################################################################


ann1 = tf.complex(tf.get_variable("a_an1", [16,2,2,1]),0.0)
ann2 =  tf.complex(tf.get_variable("a_an2", [16,3,3,1]),0.0)
ann3 =  tf.complex(tf.get_variable("a_an3", [16,4,4,1]),0.0)
ann4 =  tf.complex(tf.get_variable("a_an4", [16,5,5,1]),0.0)
ann5 =  tf.complex(tf.get_variable("a_an5", [16,6,6,1]),0.0)

x_filter1 =  tf.get_variable("a_xfilter1", [16,100,1])
x_filter2 =  tf.get_variable("a_xfilter2", [64,150,1])
x_filter3 =  tf.get_variable("a_xfilter3", [64,150,1])


C_1ft = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_inv,[0,0],[108,100]), axis =0), [16,1,1]), x_filter1)
C_2f = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2,[0,0],[44,150]), axis =0), [64,1,1]), x_filter2)
C_3f = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3,[0,0],[44,150]), axis =0), [64,1,1]), x_filter3)


C_1f = tf.complex(tf.slice(C_1ft,[0,0,0],[-1,54,-1]),tf.slice(C_1ft,[0,54,0],[-1,54,-1]))

l1n1n1 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,0,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,0,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,1.0,1.0,"test")-e_(r_val,1.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n2n1 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,3,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,0,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,2.0,1.0,"test")-e_(r_val,1.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n2n2 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,3,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,3,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,2.0,1.0,"test")-e_(r_val,2.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n3n1 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,9,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,0,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,3.0,1.0,"test")-e_(r_val,1.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n3n2 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,9,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,3,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,3.0,1.0,"test")-e_(r_val,3.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n3n3 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,9,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,9,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,3.0,1.0,"test")-e_(r_val,3.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n4n1 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,19,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,0,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,4.0,1.0,"test")-e_(r_val,1.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n4n2 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,19,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,3,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,4.0,1.0,"test")-e_(r_val,2.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n4n3 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,19,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,9,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,4.0,1.0,"test")-e_(r_val,3.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n4n4 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,19,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,19,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,4.0,1.0,"test")-e_(r_val,4.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n5n1 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,34,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,0,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,5.0,1.0,"test")-e_(r_val,1.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n5n2 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,34,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,3,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,5.0,1.0,"test")-e_(r_val,2.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n5n3 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,34,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,9,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,5.0,1.0,"test")-e_(r_val,3.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n5n4 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,34,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,19,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,5.0,1.0,"test")-e_(r_val,4.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l1n5n4 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,34,0,],[-1,2,-1]), tf.transpose(tf.slice(C_1f, [0,34,0,],[-1,2,-1]), [0,2,1])),[16,2,2,1]),tf.reshape(tf.complex(e_(r_val,5.0,1.0,"test")-e_(r_val,5.0,1.0,"test"),tf.zeros([1,100])),[1,1,1,100]))

#l1 = tf.multiply(l1n1n1 + l1n2n1 + l1n2n2 + l1n3n1 + l1n3n2 + l1n3n3 + l1n4n1 + l1n4n2 + l1n4n3 + l1n4n4 + l1n5n1  + l1n5n2 + l1n5n3 + l1n5n4 + l1n5n4,ann1)
l1 = l1n1n1 + l1n2n1 + l1n2n2 + l1n3n1 + l1n3n2 + l1n3n3 + l1n4n1 + l1n4n2 + l1n4n3 + l1n4n4 + l1n5n1  + l1n5n2 + l1n5n3 + l1n5n4 + l1n5n4
l1f = tf.reshape(l1, [-1,1])


l2n2n2 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,5,0,],[-1,3,-1]), tf.transpose(tf.slice(C_1f, [0,5,0,],[-1,3,-1]), [0,2,1])),[16,3,3,1]),tf.reshape(tf.complex(e_(r_val,2.0,2.0,"test")-e_(r_val,2.0,2.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l2n3n2 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,11,0,],[-1,3,-1]), tf.transpose(tf.slice(C_1f, [0,5,0,],[-1,3,-1]), [0,2,1])),[16,3,3,1]),tf.reshape(tf.complex(e_(r_val,3.0,2.0,"test")-e_(r_val,2.0,2.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l2n3n3 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,11,0,],[-1,3,-1]), tf.transpose(tf.slice(C_1f, [0,11,0,],[-1,3,-1]), [0,2,1])),[16,3,3,1]),tf.reshape(tf.complex(e_(r_val,3.0,2.0,"test")-e_(r_val,3.0,2.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l2n4n2 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,21,0,],[-1,3,-1]), tf.transpose(tf.slice(C_1f, [0,5,0,],[-1,3,-1]), [0,2,1])),[16,3,3,1]),tf.reshape(tf.complex(e_(r_val,4.0,2.0,"test")-e_(r_val,2.0,2.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l2n4n3 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,21,0,],[-1,3,-1]), tf.transpose(tf.slice(C_1f, [0,11,0,],[-1,3,-1]), [0,2,1])),[16,3,3,1]),tf.reshape(tf.complex(e_(r_val,4.0,2.0,"test")-e_(r_val,3.0,2.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l2n4n4 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,21,0,],[-1,3,-1]), tf.transpose(tf.slice(C_1f, [0,21,0,],[-1,3,-1]), [0,2,1])),[16,3,3,1]),tf.reshape(tf.complex(e_(r_val,4.0,2.0,"test")-e_(r_val,4.0,2.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l2n5n2 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,36,0,],[-1,3,-1]), tf.transpose(tf.slice(C_1f, [0,5,0,],[-1,3,-1]), [0,2,1])),[16,3,3,1]),tf.reshape(tf.complex(e_(r_val,5.0,2.0,"test")-e_(r_val,2.0,2.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l2n5n3 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,36,0,],[-1,3,-1]), tf.transpose(tf.slice(C_1f, [0,11,0,],[-1,3,-1]), [0,2,1])),[16,3,3,1]),tf.reshape(tf.complex(e_(r_val,5.0,2.0,"test")-e_(r_val,3.0,2.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l2n5n4 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,36,0,],[-1,3,-1]), tf.transpose(tf.slice(C_1f, [0,21,0,],[-1,3,-1]), [0,2,1])),[16,3,3,1]),tf.reshape(tf.complex(e_(r_val,5.0,2.0,"test")-e_(r_val,4.0,2.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l2n5n5 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,36,0,],[-1,3,-1]), tf.transpose(tf.slice(C_1f, [0,36,0,],[-1,3,-1]), [0,2,1])),[16,3,3,1]),tf.reshape(tf.complex(e_(r_val,5.0,2.0,"test")-e_(r_val,5.0,2.0,"test"),tf.zeros([1,100])),[1,1,1,100]))

#l2 = tf.multiply(l2n2n2 + l2n3n2 + l2n3n3 + l2n4n2 + l2n4n3 + l2n4n4 + l2n5n2 + l2n5n3 + l2n5n4 + l2n5n5,ann2)
l2 = l2n2n2 + l2n3n2 + l2n3n3 + l2n4n2 + l2n4n3 + l2n4n4 + l2n5n2 + l2n5n3 + l2n5n4 + l2n5n5

l2f =  tf.reshape(l2, [-1,1])


l3n3n3 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,14,0,],[-1,4,-1]), tf.transpose(tf.slice(C_1f, [0,14,0,],[-1,4,-1]), [0,2,1])),[16,4,4,1]),tf.reshape(tf.complex(e_(r_val,3.0,3.0,"test")-e_(r_val,3.0,3.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l3n4n3 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,24,0,],[-1,4,-1]), tf.transpose(tf.slice(C_1f, [0,14,0,],[-1,4,-1]), [0,2,1])),[16,4,4,1]),tf.reshape(tf.complex(e_(r_val,4.0,3.0,"test")-e_(r_val,3.0,3.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l3n4n4 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,24,0,],[-1,4,-1]), tf.transpose(tf.slice(C_1f, [0,24,0,],[-1,4,-1]), [0,2,1])),[16,4,4,1]),tf.reshape(tf.complex(e_(r_val,4.0,3.0,"test")-e_(r_val,4.0,3.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l3n5n3 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,39,0,],[-1,4,-1]), tf.transpose(tf.slice(C_1f, [0,14,0,],[-1,4,-1]), [0,2,1])),[16,4,4,1]),tf.reshape(tf.complex(e_(r_val,5.0,3.0,"test")-e_(r_val,3.0,3.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l3n5n4 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,39,0,],[-1,4,-1]), tf.transpose(tf.slice(C_1f, [0,24,0,],[-1,4,-1]), [0,2,1])),[16,4,4,1]),tf.reshape(tf.complex(e_(r_val,5.0,3.0,"test")-e_(r_val,4.0,3.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l3n5n5 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,39,0,],[-1,4,-1]), tf.transpose(tf.slice(C_1f, [0,39,0,],[-1,4,-1]), [0,2,1])),[16,4,4,1]),tf.reshape(tf.complex(e_(r_val,5.0,3.0,"test")-e_(r_val,5.0,3.0,"test"),tf.zeros([1,100])),[1,1,1,100]))

#l3 = tf.multiply(l3n3n3 + l3n4n3 + l3n4n4 + l3n5n3 + l3n5n4 + l3n5n5,ann3)
l3 = l3n3n3 + l3n4n3 + l3n4n4 + l3n5n3 + l3n5n4 + l3n5n5

l3f =  tf.reshape(l3, [-1,1])


l4n4n4 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,28,0,],[-1,5,-1]), tf.transpose(tf.slice(C_1f, [0,28,0,],[-1,5,-1]), [0,2,1])),[16,5,5,1]),tf.reshape(tf.complex(e_(r_val,4.0,4.0,"test")-e_(r_val,4.0,4.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l4n5n4 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,44,0,],[-1,5,-1]), tf.transpose(tf.slice(C_1f, [0,28,0,],[-1,5,-1]), [0,2,1])),[16,5,5,1]),tf.reshape(tf.complex(e_(r_val,5.0,4.0,"test")-e_(r_val,4.0,4.0,"test"),tf.zeros([1,100])),[1,1,1,100]))
l4n5n5 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,44,0,],[-1,5,-1]), tf.transpose(tf.slice(C_1f, [0,44,0,],[-1,5,-1]), [0,2,1])),[16,5,5,1]),tf.reshape(tf.complex(e_(r_val,5.0,4.0,"test")-e_(r_val,5.0,4.0,"test"),tf.zeros([1,100])),[1,1,1,100]))

#l4 = tf.multiply(l4n4n4 + l4n5n4 + l4n5n5,ann4)
l4 = l4n4n4 + l4n5n4 + l4n5n5

l4f =  tf.reshape(l4, [-1,1])


l5n5n5 =  tf.multiply(tf.reshape(tf.matmul(tf.slice(C_1, [0,48,0,],[-1,6,-1]), tf.transpose(tf.slice(C_1f, [0,48,0,],[-1,6,-1]), [0,2,1])),[16,6,6,1]),tf.reshape(tf.complex(e_(r_val,5.0,5.0,"test")-e_(r_val,5.0,5.0,"test"),tf.zeros([1,100])),[1,1,1,100]))


#l5 = tf.multiply(l5n5n5,ann5)
l5 = l5n5n5 
l5f =  tf.reshape(l5, [-1,1])

lft = tf.concat([l1f,l2f,l3f,l4f,l5f],axis = 0)
lf = tf.concat([tf.real(lft),tf.imag(lft)],axis = 0)

#C_1f = tf.get_variable("a_c1f", [8,108,1])

#transformed_omega = omega(C_1, C_1f, "transformed_omega")
print("#############################")
#print(b_complex)
#print(transformed_omega)
#output_map = tf.concat([tf.real(tf.reduce_sum(tf.multiply(b_complex,transformed_omega), axis = 2)),tf.imag(tf.reduce_sum(tf.multiply(b_complex,transformed_omega), axis = 2))], axis = 1)



f_mapz1 = tf.matmul(C_1, tf.transpose(C_1f, [0,2,1]))
f_mapz2 = tf.matmul(C_2, tf.transpose(C_2f, [0,2,1]))
f_mapz3 = tf.matmul(C_3, tf.transpose(C_3f, [0,2,1]))

#tf.maximum(tf.maximum(f_mapz1,f_mapz2),f_mapz3)

#f_mapz = tf.nn.dropout(tf.nn.relu(f_mapz1+f_mapz2+f_mapz3), keep_prob = keep_prob)



#########################################################################################3

X__1i = tf.slice(X_, [tf.shape(r_two)[0] + tf.shape(r_one)[0] + tf.shape(r_three)[0],0], [tf.shape(r_one_i)[0], -1])
X__2i = tf.slice(X_, [ tf.shape(r_two)[0] + tf.shape(r_one)[0] + tf.shape(r_three)[0] +tf.shape(r_one_i)[0],0   ],[ tf.shape(r_two_i)[0]  , -1])
X__3i = tf.slice(X_, [tf.shape(r_two)[0] + tf.shape(r_one)[0] + tf.shape(r_three)[0] +tf.shape(r_one_i)[0] +  tf.shape(r_two_i)[0], 0], [tf.shape(r_three_i)[0], -1])

r_addi =  tf.slice(r, [tf.shape(r_two)[0] + tf.shape(r_one)[0] + tf.shape(r_three)[0],0], [110, -1])
theta_addi = tf.slice(theta, [tf.shape(r_two)[0] + tf.shape(r_one)[0] + tf.shape(r_three)[0],0], [110, -1])
phi_addi = tf.slice(phi, [tf.shape(r_two)[0] + tf.shape(r_one)[0] + tf.shape(r_three)[0],0], [110, -1])


X_10i  = 0.0001 * tf.transpose(X__1i,[1,0] )
X_20i  = 0.0001 * tf.transpose(X__2i,[1,0] )
X_30i  = 0.0001 * tf.transpose(X__3i,[1,0] )

X_cal_1i, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(108) + 1.0/4.0 * tf.matmul(tf.eye(108)-tf.matmul(x,X__1i),tf.matmul(3.0*tf.eye(108)-tf.matmul(x,X__1i),3.0*tf.eye(108)-tf.matmul(x,X__1i))),x),i+1)
    ,(X_10i, 0))

X_cal_2i, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2i),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2i),3.0*tf.eye(44)-tf.matmul(x,X__2i))),x),i+1)
    ,(X_20i, 0))

X_cal_3i, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 3 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3i),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3i),3.0*tf.eye(44)-tf.matmul(x,X__3i))),x),i+1)
    ,(X_30i, 0))


X_cal_1i_ =1.0 *  tf.div(
   tf.subtract(
      X_cal_1i, 
      tf.reduce_min(X_cal_1i, keepdims = True)
   ),
   tf.subtract(
      tf.reduce_max(X_cal_1i, keepdims = True),
      tf.reduce_min(X_cal_1i, keepdims = True)
   )
)

C_1i = tf.tile(tf.expand_dims(tf.matmul(X_cal_1i,r_one_i), axis =0), [16,1,1])
C_2i = tf.tile(tf.expand_dims(tf.matmul(X_cal_2i, r_two_i), axis =0), [64,1,1])
C_3i = tf.tile(tf.expand_dims(tf.matmul(X_cal_3i,r_three_i), axis =0), [64,1,1])

##########################################################################


xi_filter1 =  tf.get_variable("a_xfilter1i", [16, 100,1])
xi_filter2 =  tf.get_variable("a_xfilter2i", [64, 150,1])
xi_filter3 =  tf.get_variable("a_xfilter3i", [64, 150,1])

C_1fi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_1i,[0,0],[108,100]),axis =0), [16,1,1]), xi_filter1)
C_2fi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2i,[0,0],[44,150]),axis =0), [64,1,1]), xi_filter2)
C_3fi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3i,[0,0],[44,150]),axis =0), [64,1,1]), xi_filter3)

#C_1fi = tf.get_variable("a_c1fi", [8,108,1])

f_mapz1i = tf.matmul(C_1i, tf.transpose(C_1fi, [0,2,1]))
f_mapz2i = tf.matmul(C_2i, tf.transpose(C_2fi, [0,2,1]))
f_mapz3i = tf.matmul(C_3i, tf.transpose(C_3fi, [0,2,1]))

b_complexi =  b(r_addi + trans_val ,theta_addi+alpha_val, phi_addi+beta_val , "samplingi")

#f_mapzi = tf.nn.dropout(tf.nn.relu(f_mapz1i+f_mapz2i+f_mapz3i),keep_prob = keep_prob)
#transformed_omega2 = omega(C_1i, C_1fi, "transformed_omega2")

print("#############################")
#print(b_complex)
#print(transformed_omega)

#output_map2 = tf.concat([tf.real(tf.reduce_sum(tf.multiply(b_complexi,transformed_omega2), axis = 2)),tf.imag(tf.reduce_sum(tf.multiply(b_complexi,transformed_omega2), axis = 2))],axis = 1)

##########################################################################################

s1 = tf.shape(r_two)[0] + tf.shape(r_one)[0] + tf.shape(r_three)[0] + tf.shape(r_one_i)[0]
s2 = s1 + tf.shape(r_onex)[0]
s3 = s2 + tf.shape(r_twox)[0]



X__1x = tf.slice(X_, [s1,0], [tf.shape(r_onex)[0], -1])
X__2x = tf.slice(X_, [ s2, 0   ],[ tf.shape(r_twox)[0]  , -1])
X__3x = tf.slice(X_, [s3, 0], [tf.shape(r_threex)[0], -1])




X_10x  = 0.0001 * tf.transpose(X__1x,[1,0] )
X_20x  = 0.0001 * tf.transpose(X__2x,[1,0] )
X_30x  = 0.0001 * tf.transpose(X__3x,[1,0] )

X_cal_1x, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__1x),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__1x),3.0*tf.eye(44)-tf.matmul(x,X__1x))),x),i+1)
    ,(X_10x, 0))

X_cal_2x, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2x),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2x),3.0*tf.eye(44)-tf.matmul(x,X__2x))),x),i+1)
    ,(X_20x, 0))

X_cal_3x, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3x),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3x),3.0*tf.eye(44)-tf.matmul(x,X__3x))),x),i+1)
    ,(X_30x, 0))



C_1x = tf.tile(tf.expand_dims(tf.matmul(X_cal_1x,r_onex), axis =0), [64,1,1])

C_2x = tf.tile(tf.expand_dims(tf.matmul(X_cal_2x, r_twox), axis =0), [64,1,1])
C_3x = tf.tile(tf.expand_dims(tf.matmul(X_cal_3x,r_threex), axis =0), [64,1,1])

###########################################################################################################################################33

x_filter1x =  tf.get_variable("a_xfilter1x", [64,150,1])
x_filter2x =  tf.get_variable("a_xfilter2x", [64,150,1])
x_filter3x =  tf.get_variable("a_xfilter3x", [64,150,1])


C_1fx = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_1x,[0,0],[44,150]),axis =0), [64,1,1]), x_filter1x)
C_2fx = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2x,[0,0],[44,150]),axis =0), [64,1,1]), x_filter2x)
C_3fx = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3x,[0,0],[44,150]),axis =0), [64,1,1]), x_filter3x)

f_mapx1x = tf.matmul(C_1x, tf.transpose(C_1fx, [0,2,1]))
f_mapx2x = tf.matmul(C_2x, tf.transpose(C_2fx, [0,2,1]))
f_mapx3x = tf.matmul(C_3x, tf.transpose(C_3fx, [0,2,1]))

f_mapx = tf.nn.dropout(tf.nn.relu(f_mapx1x+f_mapx2x+f_mapx3x),keep_prob = keep_prob)



##############################################################33

s4 = s3 + tf.shape(r_threex)[0]
s5 = s4 + tf.shape(r_onexi)[0]
s6 = s5 + tf.shape(r_twoxi)[0]



X__1xi = tf.slice(X_, [s4,0], [tf.shape(r_onexi)[0], -1])
X__2xi = tf.slice(X_, [ s5, 0   ],[ tf.shape(r_twoxi)[0]  , -1])
X__3xi = tf.slice(X_, [s6, 0], [tf.shape(r_threexi)[0], -1])




X_10xi  = 0.0001 * tf.transpose(X__1xi,[1,0] )
X_20xi  = 0.0001 * tf.transpose(X__2xi,[1,0] )
X_30xi  = 0.0001 * tf.transpose(X__3xi,[1,0] )

X_cal_1xi, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__1xi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__1xi),3.0*tf.eye(44)-tf.matmul(x,X__1xi))),x),i+1)
    ,(X_10xi, 0))

X_cal_2xi, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2xi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2xi),3.0*tf.eye(44)-tf.matmul(x,X__2xi))),x),i+1)
    ,(X_20xi, 0))

X_cal_3xi, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3xi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3xi),3.0*tf.eye(44)-tf.matmul(x,X__3xi))),x),i+1)
    ,(X_30xi, 0))



C_1xi = tf.tile(tf.expand_dims(tf.matmul(X_cal_1xi,r_onexi), axis =0), [64,1,1])

C_2xi = tf.tile(tf.expand_dims(tf.matmul(X_cal_2xi, r_twoxi), axis =0), [64,1,1])
C_3xi = tf.tile(tf.expand_dims(tf.matmul(X_cal_3xi,r_threexi), axis =0), [64,1,1])

#####################################################################3


x_filter1xi =  tf.get_variable("a_xfilter1xi", [64,150,1])
x_filter2xi =  tf.get_variable("a_xfilter2xi", [64,150,1])
x_filter3xi =  tf.get_variable("a_xfilter3xi", [64,150,1])


C_1fxi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_1xi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter1xi)
C_2fxi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2xi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter2xi)
C_3fxi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3xi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter3xi)

f_mapx1xi = tf.matmul(C_1xi, tf.transpose(C_1fxi, [0,2,1]))
f_mapx2xi = tf.matmul(C_2xi, tf.transpose(C_2fxi, [0,2,1]))
f_mapx3xi = tf.matmul(C_3xi, tf.transpose(C_3fxi, [0,2,1]))

f_mapxxi = tf.nn.dropout(tf.nn.relu(f_mapx1xi+f_mapx2xi+ f_mapx3xi), keep_prob = keep_prob)




###################################################################################################################################3


s7 = s6 + tf.shape(r_threexi)[0]
s8 = s7 + tf.shape(r_oney)[0]
s9 = s8 + tf.shape(r_twoy)[0]



X__1y = tf.slice(X_, [s7,0], [tf.shape(r_oney)[0], -1])
X__2y = tf.slice(X_, [ s8, 0   ],[ tf.shape(r_twoy)[0]  , -1])
X__3y = tf.slice(X_, [s9, 0], [tf.shape(r_threey)[0], -1])




X_10y  = 0.0001 * tf.transpose(X__1y,[1,0] )
X_20y  = 0.0001 * tf.transpose(X__2y,[1,0] )
X_30y  = 0.0001 * tf.transpose(X__3y,[1,0] )

X_cal_1y, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__1y),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__1y),3.0*tf.eye(44)-tf.matmul(x,X__1y))),x),i+1)
    ,(X_10y, 0))

X_cal_2y, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2y),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2y),3.0*tf.eye(44)-tf.matmul(x,X__2y))),x),i+1)
    ,(X_20y, 0))

X_cal_3y, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3y),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3y),3.0*tf.eye(44)-tf.matmul(x,X__3y))),x),i+1)
    ,(X_30y, 0))



C_1y = tf.tile(tf.expand_dims(tf.matmul(X_cal_1y,r_oney), axis =0), [64,1,1])

C_2y = tf.tile(tf.expand_dims(tf.matmul(X_cal_2y, r_twoy), axis =0), [64,1,1])
C_3y = tf.tile(tf.expand_dims(tf.matmul(X_cal_3y,r_threey), axis =0), [64,1,1])

###############################################################################################

x_filter1y =  tf.get_variable("a_xfilter1y", [64,150,1])
x_filter2y =  tf.get_variable("a_xfilter2y", [64,150,1])
x_filter3y =  tf.get_variable("a_xfilter3y", [64,150,1])


C_1fy = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_1y,[0,0],[44,150]),axis =0), [64,1,1]), x_filter1y)
C_2fy = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2y,[0,0],[44,150]),axis =0), [64,1,1]), x_filter2y)
C_3fy = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3y,[0,0],[44,150]),axis =0), [64,1,1]), x_filter3y)

f_mapx1y = tf.matmul(C_1y, tf.transpose(C_1fy, [0,2,1]))
f_mapx2y = tf.matmul(C_2y, tf.transpose(C_2fy, [0,2,1]))
f_mapx3y = tf.matmul(C_3y, tf.transpose(C_3fy, [0,2,1]))

f_mapxy = tf.nn.dropout(tf.nn.relu(f_mapx1y+f_mapx2y+ f_mapx3y), keep_prob = keep_prob)


#################################################################################################################33



s10 = s9 + tf.shape(r_threey)[0]
s11 = s10 + tf.shape(r_oneyi)[0]
s12 = s11 + tf.shape(r_twoyi)[0]



X__1yi = tf.slice(X_, [s10,0], [tf.shape(r_oneyi)[0], -1])
X__2yi = tf.slice(X_, [ s11, 0   ],[ tf.shape(r_twoyi)[0]  , -1])
X__3yi = tf.slice(X_, [s12, 0], [tf.shape(r_threeyi)[0], -1])




X_10yi  = 0.0001 * tf.transpose(X__1yi,[1,0] )
X_20yi  = 0.0001 * tf.transpose(X__2yi,[1,0] )
X_30yi  = 0.0001 * tf.transpose(X__3yi,[1,0] )

X_cal_1yi, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__1yi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__1yi),3.0*tf.eye(44)-tf.matmul(x,X__1yi))),x),i+1)
    ,(X_10yi, 0))

#X_cal_1yi = tf.matmul( tf.eye(30) + 1.0/4.0 * tf.matmul(tf.eye(30)-tf.matmul(X_10yi,X__1yi),tf.matmul(3.0*tf.eye(30)-tf.matmul(X_10yi,X__1yi),3.0*tf.eye(30)-tf.matmul(X_10yi,X__1yi))),X_10yi)

X_cal_2yi, p_2 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__2yi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__2yi),3.0*tf.eye(44)-tf.matmul(x,X__2yi))),x),i+1)
    ,(X_20yi, 0))

#X_cal_2yi =  tf.matmul( tf.eye(30) + 1.0/4.0 * tf.matmul(tf.eye(30)-tf.matmul(X_20yi,X__2yi),tf.matmul(3.0*tf.eye(30)-tf.matmul(X_20yi,X__2yi),3.0*tf.eye(30)-tf.matmul(X_20yi,X__2yi)))


X_cal_3yi, p_3 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X__3yi),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X__3yi),3.0*tf.eye(44)-tf.matmul(x,X__3yi))),x),i+1)
    ,(X_30yi, 0))

#X_cal_3yi = tf.matmul( tf.eye(30) + 1.0/4.0 * tf.matmul(tf.eye(30)-tf.matmul(X_30yi,X__3yi),tf.matmul(3.0*tf.eye(30)-tf.matmul(X_30yi,X__3yi),3.0*tf.eye(30)-tf.matmul(X_30yi,X__3yi)))

C_1yi = tf.tile(tf.expand_dims(tf.matmul(X_cal_1yi,r_oneyi), axis =0), [64,1,1])

C_2yi = tf.tile(tf.expand_dims(tf.matmul(X_cal_2yi, r_twoyi), axis =0), [64,1,1])
C_3yi = tf.tile(tf.expand_dims(tf.matmul(X_cal_3yi,r_threeyi), axis =0), [64,1,1])


##############################################################################


x_filter1yi =  tf.get_variable("a_xfilter1yi", [64,150,1])
x_filter2yi =  tf.get_variable("a_xfilter2yi", [64,150,1])
x_filter3yi =  tf.get_variable("a_xfilter3yi", [64, 150,1])


C_1fyi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_1yi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter1yi)
C_2fyi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_2yi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter2yi)
C_3fyi = tf.matmul(tf.tile(tf.expand_dims(tf.slice(X_cal_3yi,[0,0],[44,150]),axis =0), [64,1,1]), x_filter3yi)

f_mapx1yi = tf.matmul(C_1yi, tf.transpose(C_1fyi, [0,2,1]))
f_mapx2yi = tf.matmul(C_2yi, tf.transpose(C_2fyi, [0,2,1]))
f_mapx3yi = tf.matmul(C_3yi, tf.transpose(C_3fyi, [0,2,1]))

f_mapxyi = tf.nn.dropout(tf.nn.relu(f_mapx1yi+f_mapx2yi+f_mapx3yi),keep_prob = keep_prob)



print("##################!!!!!!!!!!!!!!!!!!!!!")
print(f_mapxyi)















U_4 = tf.concat([u_0_4, u_1_4, u_2_4, u_3_4, u_4_4] ,  axis=1)
V_4 = tf.concat([v_0_4,v_1_4,v_2_4,v_3_4,v_4_4] ,  axis=1)

X_4 = tf.concat([U_4,V_4] ,  axis=1)

X_41 = tf.slice(X_4, [0,0], [tf.shape(r_one)[0], -1])
X_42 = tf.slice(X_4, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X_43 = tf.slice(X_4, [tf.shape(r_two)[0] + tf.shape(r_one)[0], 0], [tf.shape(r_three)[0], -1])




f_411 = safe_norm(tf.matmul(X_41,tf.slice(C_1,[20,0],[10,-1])),axis = None, keep_dims = True)
f_412 = safe_norm(tf.matmul(X_42,tf.slice(C_2,[20,0],[10,-1])),axis = None, keep_dims = True)
f_413 = safe_norm(tf.matmul(X_43,tf.slice(C_3,[20,0],[10,-1])),axis = None, keep_dims = True)


_, f_421 = tf.nn.moments(tf.matmul(X_41,tf.slice(C_1,[20,0],[10,-1])),axes = [0],  keep_dims = True)
_, f_422 = tf.nn.moments(tf.matmul(X_42,tf.slice(C_2,[20,0],[10,-1])),axes = [0],  keep_dims = True)
_, f_423 = tf.nn.moments(tf.matmul(X_43,tf.slice(C_3,[20,0],[10,-1])),axes = [0],  keep_dims = True)

   
#############################################################################################

U_3 = tf.concat([u_0_3, u_1_3, u_2_3, u_3_3] ,  axis=1)
V_3 = tf.concat([v_0_3,v_1_3,v_2_3,v_3_3] ,  axis=1)

X_3 = tf.concat([U_3,V_3] ,  axis=1)

X_31 = tf.slice(X_3, [0,0], [tf.shape(r_one)[0], -1])
X_32 = tf.slice(X_3, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X_33 = tf.slice(X_3, [tf.shape(r_two)[0] + tf.shape(r_one)[0], 0], [tf.shape(r_three)[0], -1])


f_311 = safe_norm(tf.matmul(X_31,tf.slice(C_1,[12,0],[8,-1])),axis = None, keep_dims = True)
f_312 = safe_norm(tf.matmul(X_32,tf.slice(C_2,[12,0],[8,-1])),axis = None, keep_dims = True)
f_313 = safe_norm(tf.matmul(X_33,tf.slice(C_3,[12,0],[8,-1])),axis = None, keep_dims = True)


_, f_321 = tf.nn.moments(tf.matmul(X_31,tf.slice(C_1,[12,0],[8,-1])),axes = [0],  keep_dims = True)
_, f_322 = tf.nn.moments(tf.matmul(X_32,tf.slice(C_2,[12,0],[8,-1])),axes = [0],  keep_dims = True)
_, f_323 = tf.nn.moments(tf.matmul(X_33,tf.slice(C_3,[12,0],[8,-1])),axes = [0],  keep_dims = True)


#################################################################################################################
U_2 = tf.concat([u_0_2, u_1_2, u_2_2] ,  axis=1)
V_2 = tf.concat([v_0_2,v_1_2,v_2_2] ,  axis=1)

X_2 = tf.concat([U_2,V_2] ,  axis=1)

X_21 = tf.slice(X_2, [0,0], [tf.shape(r_one)[0], -1])
X_22 = tf.slice(X_2, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X_23 = tf.slice(X_2, [tf.shape(r_two)[0] + tf.shape(r_one)[0], 0], [tf.shape(r_three)[0], -1])



f_211 = safe_norm(tf.matmul(X_21,tf.slice(C_1,[6,0],[6,-1])),axis = None, keep_dims = True)
f_212 = safe_norm(tf.matmul(X_22,tf.slice(C_2,[6,0],[6,-1])),axis = None, keep_dims = True)
f_213 = safe_norm(tf.matmul(X_23,tf.slice(C_3,[6,0],[6,-1])),axis = None, keep_dims = True)


_, f_221 = tf.nn.moments(tf.matmul(X_21,tf.slice(C_1,[6,0],[6,-1])),axes = [0],  keep_dims = True)
_, f_222 = tf.nn.moments(tf.matmul(X_22,tf.slice(C_2,[6,0],[6,-1])),axes = [0],  keep_dims = True)
_, f_223 = tf.nn.moments(tf.matmul(X_23,tf.slice(C_3,[6,0],[6,-1])),axes = [0],  keep_dims = True)

#########################################################################################################################

U_1 = tf.concat([u_0_1, u_1_1] ,  axis=1)
V_1 = tf.concat([v_0_1,v_1_1] ,  axis=1)

X_1 = tf.concat([U_1,V_1] ,  axis=1)

X_11 = tf.slice(X_1, [0,0], [tf.shape(r_one)[0], -1])
X_12 = tf.slice(X_1, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X_13 = tf.slice(X_1, [tf.shape(r_two)[0] + tf.shape(r_one)[0], 0], [tf.shape(r_three)[0], -1])

f_111 = safe_norm(tf.matmul(X_11,tf.slice(C_1,[2,0],[4,-1])),axis = None, keep_dims = True)
f_112 = safe_norm(tf.matmul(X_12,tf.slice(C_2,[2,0],[4,-1])),axis = None, keep_dims = True)
f_113 = safe_norm(tf.matmul(X_13,tf.slice(C_3,[2,0],[4,-1])),axis = None, keep_dims = True)


_, f_121 = tf.nn.moments(tf.matmul(X_11,tf.slice(C_1,[2,0],[4,-1])),axes = [0],  keep_dims = True)
_, f_122 = tf.nn.moments(tf.matmul(X_12,tf.slice(C_2,[2,0],[4,-1])),axes = [0],  keep_dims = True)
_, f_123 = tf.nn.moments(tf.matmul(X_13,tf.slice(C_3,[2,0],[4,-1])),axes = [0],  keep_dims = True)


##############################################################################################


U_0 = tf.concat([u_0_0] ,  axis=1)
V_0 = tf.concat([v_0_0] ,  axis=1)

X_0 = tf.concat([U_0,V_0] ,  axis=1)

X_01 = tf.slice(X_0, [0,0], [tf.shape(r_one)[0], -1])
X_02 = tf.slice(X_0, [tf.shape(r_one)[0], 0], [tf.shape(r_two)[0], -1])
X_03 = tf.slice(X_0, [tf.shape(r_two)[0] + tf.shape(r_one)[0] ,0] , [tf.shape(r_three)[0], -1])



f_011 = safe_norm(tf.matmul(X_01,tf.slice(C_1,[0,0],[2,-1])),axis = None, keep_dims = True)
f_012 = safe_norm(tf.matmul(X_02,tf.slice(C_2,[0,0],[2,-1])),axis = None, keep_dims = True)
f_013 = safe_norm(tf.matmul(X_03,tf.slice(C_3,[0,0],[2,-1])),axis = None, keep_dims = True)


_, f_021 = tf.nn.moments(tf.matmul(X_01,tf.slice(C_1,[0,0],[2,-1])),axes = [0],  keep_dims = True)
_, f_022 = tf.nn.moments(tf.matmul(X_02,tf.slice(C_2,[0,0],[2,-1])),axes = [0],  keep_dims = True)
_, f_023 = tf.nn.moments(tf.matmul(X_03,tf.slice(C_3,[0,0],[2,-1])),axes = [0],  keep_dims = True)

#####################################################################################################3

l = tf.greater(theta_one[:,0],0)

theta_1 = tf.boolean_mask(theta_one, l)
phi_1 = tf.boolean_mask(phi_one, l)
r_1 = tf.boolean_mask(r_one, l)




r_pow = tf.transpose(tf.pow(r_1, 3), [1,0])
radial_real = tf.multiply(radial_poly(r_1,3,1), tf.cos(theta_1 * 3.0 + phi_1))
radial_imag = tf.multiply(radial_poly(r_1,3,1), tf.sin(theta_1 * 3.0 + phi_1))

real_term = tf.square(tf.matmul(r_pow,radial_imag))
imag_term = tf.square(tf.matmul(r_pow,radial_imag))

moment_31 = -0.277 * tf.sqrt(real_term + imag_term+ 0.0001)



radial_real2 = tf.multiply(radial_poly(r_1,2,1), tf.cos(theta_1 * 2.0 + phi_1))
radial_imag2 = tf.multiply(radial_poly(r_1,2,1), tf.sin(theta_1 * 2.0 + phi_1))

real_term2 = tf.square(tf.matmul(r_pow,radial_imag2))
imag_term2 = tf.square(tf.matmul(r_pow,radial_imag2))

moment_21 = 0.207 * tf.sqrt(real_term2 + imag_term2)

C_1_softmax = tf.nn.softmax(C_1, axis = 0)
C_2_softmax = tf.nn.softmax(C_2, axis = 0)
C_3_softmax = tf.nn.softmax(C_3, axis = 0)

f_1_softmax = tf.concat([ f_011, f_021, f_111, f_121,f_211, f_221, f_311, f_321,f_411, f_421 ], axis =0)
f_2_softmax = tf.concat([  f_012, f_022, f_112, f_122, f_212, f_222, f_312, f_322, f_412, f_422  ], axis = 0)
f_3_softmax = tf.concat([f_013, f_023, f_113, f_123, f_213, f_223,  f_313, f_323, f_413, f_423 ], axis = 0)

m1_softmax =  tf.concat([ moment_31, moment_21  ], axis = 0)
m2_softmax =  tf.concat([  moment_31, moment_21  ], axis = 0)
m3_softmax =  tf.concat([  moment_31, moment_21  ], axis = 0) 



#B_1 = tf.concat([C_1, f_011, f_021, f_111, f_121,f_211, f_221, f_311, f_321,f_411, f_421,moment_31,moment_21  ], axis =0)
#B_2 = tf.concat([C_2,  f_012, f_022, f_112, f_122, f_212, f_222, f_312, f_322, f_412, f_422, moment_31, moment_21   ], axis = 0)
#B_3 = tf.concat([C_3, f_013, f_023, f_113, f_123, f_213, f_223,  f_313, f_323, f_413, f_423, moment_31, moment_21  ], axis = 0)

B_1 =  tf.concat([C_1,C_1i, C_1x, C_1xi, C_1y, C_1yi, f_1_softmax,m1_softmax  ], axis =0)
B_2 =  tf.concat([C_2, C_2i,  C_2x, C_2xi, C_2y, C_2yi, f_2_softmax,m2_softmax  ], axis =0)
B_3 =  tf.concat([C_3, C_3i,  C_3x, C_3xi, C_3y, C_3yi, f_3_softmax,m3_softmax  ], axis =0)


z_reshape = tf.expand_dims(tf.reshape(f_mapz, [64,1,1936]), axis = 1)
#z_reshape = tf.reshape(f_mapz, [1,8,8,1936])
zi_reshape = tf.expand_dims(tf.reshape(f_mapzi, [64,1,1936]), axis = 1)
#zi_reshape = tf.reshape(f_mapzi, [1,8,8,1936])

x_reshape = tf.expand_dims(tf.reshape(f_mapx, [64,1,1936]), axis = 1)
#x_reshape = tf.reshape(f_mapx, [1,8,8,1936])
xi_reshape = tf.expand_dims(tf.reshape(f_mapxxi, [64,1,1936]), axis = 1)
#xi_reshape = tf.reshape(f_mapxxi, [1,8,8,1936])

y_reshape = tf.expand_dims(tf.reshape(f_mapxy, [64,1,1936]), axis = 1)
#y_reshape = tf.reshape(f_mapxy, [1,8,8,1936])
yi_reshape = tf.expand_dims(tf.reshape(f_mapxyi, [64,1,1936]), axis = 1)
#yi_reshape = tf.reshape(f_mapxyi, [1,8,8,1936])

#z_compact = tf.reshape(compact_bilinear_pooling_layer(z_reshape, zi_reshape, 1000 , sum_pool=False, sequential=False),[1,8,8,1000])
#x_compact =  tf.reshape(compact_bilinear_pooling_layer(x_reshape, xi_reshape, 1000 , sum_pool=False, sequential=False),[1,8,8,1000])
#y_compact =  tf.reshape(compact_bilinear_pooling_layer(y_reshape, yi_reshape, 1000 , sum_pool=False, sequential=False),[1,8,8,1000])


#compact1 = tf.reshape(compact_bilinear_pooling_layer(z_compact, x_compact, 5000 , sum_pool=False, sequential=False),[1,8,8,5000])



#feature_map1 = tf.nn.max_pool(tf.reshape(tf.concat([z_reshape, x_reshape, y_reshape], axis = 1), [1,44,44*3,64]),ksize = (1,8,8,1), strides = (1,6,6,1), padding = "SAME")
feature_map1 = tf.reshape(tf.concat([z_reshape, x_reshape, y_reshape], axis = 1), [1,8,8,1936*3])

feature_map2 = tf.reshape(tf.concat([zi_reshape, xi_reshape, yi_reshape], axis = 1), [1,8,8,1936*3])

#feature_map2 = tf.nn.max_pool(tf.reshape(tf.concat([zi_reshape, xi_reshape, yi_reshape], axis = 1), [1,44,44*3,64]),ksize = (1,8,8,1), strides = (1,6,6,1), padding = "SAME")


B = tf.concat([f_mapz,f_mapzi, f_mapx, f_mapxxi,f_mapxy, f_mapxyi ], axis =0)
#top = compact_bilinear_pooling_layer(feature_map1, feature_map2, 1000 , sum_pool=True, sequential=False)
#top = tf.nn.relu(compact_bilinear_pooling_layer(compact1, y_compact, 10000 , sum_pool=False, sequential=False))


#B_1 = tf.concat([C_1,moment_31,moment_21 ], axis =0)
#B_2 = tf.concat([C_2, moment_31, moment_21  ], axis = 0)
#B_3 = tf.concat([C_3, moment_31, moment_21 ], axis = 0)






#test10 = tf.norm(tf.matmul(X_1,B_1))
#test_11 = tf.reduce_max(tf.matmul(X_1,B_1))

#  = tf.concat([B_1, B_2, B_3], axis = 0)
def dense_batch_relu(x, phase, scope,units):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, units, 
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, 
                                          center=True, scale=True, 
                                          is_training=phase,
                                          scope='bn')
        return tf.nn.relu(h2, 'relu')


phase = tf.placeholder(tf.bool, name='phase')


# print(B)
#estimate_1 = tf.matmul(tf.transpose(u, perm=[0, 2, 1]),r_temp)
init_sigma = 0.01

# A_init = tf.random_normal(
# 		  shape=(3, 10,1),
# 		  stddev=init_sigma, dtype=tf.float32, name="cn_W_init")
# A = tf.Variable(A_init, name="cn_W")
"""
output_map =1.0 *  tf.div(
   tf.subtract(
      output_map ,
      tf.reduce_min(output_map , keepdims = True)
   ),
   tf.subtract(
      tf.reduce_max(output_map , keepdims = True),
      tf.reduce_min(output_map , keepdims = True)
   )
)

output_map2 =1.0 *  tf.div(
   tf.subtract(
      output_map2 ,
      tf.reduce_min(output_map2 , keepdims = True)
   ),
   tf.subtract(
      tf.reduce_max(output_map2 , keepdims = True),
      tf.reduce_min(output_map2 , keepdims = True)
   )
)
"""

def g_norm(inputs,
               groups=32,
               channels_axis=-1,
               reduction_axes=(-3, -2),
               center=True,
               scale=True,
               epsilon=1e-6,
               activation_fn=None,
               param_initializers=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None,
               mean_close_to_zero=False):
  
  # TODO(shlens): Support partially defined shapes for the inputs.
  inputs = ops.convert_to_tensor(inputs)
  original_shape = inputs.shape

  if inputs.shape.ndims is None:
    raise ValueError('Inputs %s has undefined rank.' % inputs.name)
  if channels_axis > (inputs.shape.ndims - 1):
    raise ValueError('Axis is out of bounds.')

  # Standardize the channels_axis to be positive and identify # of channels.
  if channels_axis < 0:
    channels_axis = inputs.shape.ndims + channels_axis
  channels = inputs.shape[channels_axis].value

  if channels is None:
    raise ValueError('Inputs %s has undefined channel dimension: %d.' % (
        inputs.name, channels_axis))

  # Standardize the reduction_axes to be positive.
  reduction_axes = list(reduction_axes)
  for i in range(len(reduction_axes)):
    if reduction_axes[i] < 0:
      reduction_axes[i] += inputs.shape.ndims

  for a in reduction_axes:
    if a > inputs.shape.ndims:
      raise ValueError('Axis is out of bounds.')
    if inputs.shape[a].value is None:
      raise ValueError('Inputs %s has undefined dimensions %d.' % (
          inputs.name, a))
    if channels_axis == a:
      raise ValueError('reduction_axis must be mutually exclusive '
                       'with channels_axis')
  if groups > channels:
    raise ValueError('Invalid groups %d for %d channels.' % (groups, channels))
  if channels % groups != 0:
    raise ValueError('%d channels is not commensurate with %d groups.' %
                     (channels, groups))

  # Determine axes before channels. Some examples of common image formats:
  #  'NCHW': before = [N], after = [HW]
  #  'NHWC': before = [NHW], after = []
  axes_before_channels = inputs.shape.as_list()[:channels_axis]
  axes_after_channels = inputs.shape.as_list()[channels_axis+1:]

  # Manually broadcast the parameters to conform to the number of groups.
  params_shape_broadcast = ([1] * len(axes_before_channels) +
                            [groups, channels // groups] +
                            [1] * len(axes_after_channels))

  # Reshape the input by the group within the channel dimension.
  inputs_shape = (axes_before_channels + [groups, channels // groups] +
                  axes_after_channels)
  inputs = array_ops.reshape(inputs, inputs_shape)

  # Determine the dimensions across which moments are calculated.
  moments_axes = [channels_axis + 1]
  for a in reduction_axes:
    if a > channels_axis:
      moments_axes.append(a + 1)
    else:
      moments_axes.append(a)

  with variable_scope.variable_scope(
      scope, 'GroupNorm', [inputs], reuse=reuse) as sc:
    # Note that the params_shape is the number of channels always.
    params_shape = [channels]

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    dtype = inputs.dtype.base_dtype
    if param_initializers is None:
      param_initializers = {}
    if center:
      beta_collections = utils.get_variable_collections(
          variables_collections, 'beta')
      beta_initializer = param_initializers.get(
          'beta', init_ops.zeros_initializer())
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=beta_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
      beta = array_ops.reshape(beta, params_shape_broadcast)

    if scale:
      gamma_collections = utils.get_variable_collections(
          variables_collections, 'gamma')
      gamma_initializer = param_initializers.get(
          'gamma', init_ops.ones_initializer())
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=gamma_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)
      gamma = array_ops.reshape(gamma, params_shape_broadcast)

    # Calculate the moments.
    if mean_close_to_zero:
      # One pass algorithm returns better result when mean is close to zero.
      counts, means_ss, variance_ss, _ = nn.sufficient_statistics(
          inputs, moments_axes, keep_dims=True)
      mean, variance = nn.normalize_moments(
          counts, means_ss, variance_ss, shift=None)
    else:
      mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)

    # Compute normalization.
    # TODO(shlens): Fix nn.batch_normalization to handle the 5-D Tensor
    # appropriately so that this operation may be faster.
    gain = math_ops.rsqrt(variance + epsilon)
    offset = -mean * gain
    if gamma is not None:
      gain *= gamma
      offset *= gamma
    if beta is not None:
      offset += beta
    outputs = inputs * gain + offset

    # Collapse the groups into the channel dimension.
    outputs = array_ops.reshape(outputs, original_shape)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)



#T_ = tf.concat([output_map,output_map2], axis = 1)

#layer_2_output = tf.matmul(layer_1,W_norm)
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
T = tf.one_hot(y, depth=10, name="T")

#T_ = T * -2.0 + 1.0

#layer_2_output =  tf.multiply(T_ ,layer_2)

#y_pred = tf.squeeze(tf.argmax(layer_2_output,axis = 1))

#y_pred = tf.argmax(layer_3_output, axis=1)


#y = tf.placeholder(shape=[None], dtype=tf.int16, name="y")
#T = tf.one_hot(y, depth=10, name="T")


#print(trainable_vars)

#names = [n.name for n in tf.get_default_graph().as_graph_def().node]


batch = tf.Variable(0, trainable = False)


assign_op = batch.assign(0)

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count



optimizer_2 =  tf.train.GradientDescentOptimizer(learning_rate=0.1)
#grads_2 = optimizer_2.compute_gradients(rot_loss, var_list = rot_vars)
#training_op_2 = optimizer_2.minimize(rot_loss, name="training_op_2", var_list = rot_vars)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
no_opt = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0,
                             do_common_subexpression_elimination=False,
                             do_function_inlining=False,
                             do_constant_folding=False)
config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=no_opt),
                        log_device_placement=True, allow_soft_placement=False,
                        device_count={"GPU": 0},
                        inter_op_parallelism_threads=3,
                        intra_op_parallelism_threads=1)

print('before_sess')
sess = tf.Session(config=config)
print('after_sess')

def condition(x, i, index, axis):
    return tf.logical_and(tf.equal(x[0,i,0], index), tf.equal(x[0,i,2], axis))


#correct = tf.equal(y, y_pred, name="correct")
#accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")


# block_hankel = tf.slice(calibrated_points_one_corrected_shape, [0, 0, 0], [-1,10,-1])
#sess.run(tf.global_variables_initializer())

def as_spherical(xyz):
    ptsnew = np.zeros(xyz.shape)
    max_r = np.max(np.linalg.norm(xyz,axis = 1))
    xy = (xyz[:,0]**2 + xyz[:,1]**2)
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)/max_r
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])+math.pi
    return ptsnew

def find_index(grid, array,index):
#	print("grid")
#	print(grid)
#	print("array")
#	print(np.expand_dims(array[:,index],axis = 1))
#	print("diff")
#	print((np.abs(grid - np.expand_dims(array[:,index],axis = 1))))
#	print("res")

	res = (np.abs(grid - np.expand_dims(array[:,index],axis = 1))).argmin(axis = 1)
#	print(res)
	return res

def format_input(sph_in):
	r_grid = np.linspace(start = 0.5, stop = 1, num = 10)
	phi_grid = np.linspace(start = 0, stop = math.pi, num = 20)
	theta_grid = np.linspace(start =0, stop = 2*math.pi, num = 20) 
	
#	phi, theta = np.meshgrid(phi_grid, theta_grid)
#	f = sph_harm(1, 10, theta, phi)
#	print(f)
	input_grid = np.zeros((20,20,10))
	r_arr = np.expand_dims(find_index(r_grid,sph_in,0),axis = 0)
	phi_arr = np.expand_dims(find_index(phi_grid,sph_in,1),axis = 0)
	theta_arr = np.expand_dims(find_index(theta_grid,sph_in,2),axis = 0)
	
	index_arr = np.concatenate((theta_arr,phi_arr,r_arr), axis=0)
	input_grid[theta_arr,phi_arr,r_arr] = input_grid[theta_arr,phi_arr,r_arr] +  1
#	print(input_grid)
	input_grid_tf = np.expand_dims(np.expand_dims(input_grid, axis = 0),axis = 4)
	return input_grid_tf

def sph_harm_lm(l, m, phi, theta):
    """ Wrapper around scipy.special.sph_harm. Return spherical harmonic of degree l and order m. """
    phi, theta = np.meshgrid(phi, theta)
    f = sph_harm(m, l, theta, phi)
    f_grid = f[np.newaxis,:,:,np.newaxis,np.newaxis]
   # f_tiled = np.tile(f_grid,(1,1,1,51,1))
    f_tiled = np.tile(f_grid,(1,1,1,10,1))
   # print(f_tiled.shape)
   # print(f_tiled)
    return f_tiled

def lin_pols_nl(n, l, r):
    """ Wrapper around scipy.special.sph_harm. Return spherical harmonic of degree l and order m. """
    r_grid = r[np.newaxis,np.newaxis,:]
    r_grid_tiled = np.tile(r_grid,(20,20,1))
    f = tf.expand_dims(tf.expand_dims(radial_poly(r_grid_tiled, n,l), axis = 0), axis = 4)
  #  print(f.shape)
 #   print(f)
    return f

def e_r(r,n,l):
   #print(r)
    k = np.zeros(r.shape)
    for i in range(2,n+1):
        k += (((n-1+1)*(-r))**i)/math.factorial(i)
   #print(k)
    return np.exp((n-1+1)*(-r))

def e_nl(n, l, r):
    r_grid = r[np.newaxis,np.newaxis,:]
    r_grid_tiled = np.tile(r_grid,(20,20,1))
    f = np.expand_dims(np.expand_dims(e_r(r_grid_tiled, n,l), axis = 0), axis = 4)
    return f

def get_harmonics(theta_grid, phi_grid, n_max):
	harmonics = []
	#	harmonics = np.zeros((theta_dim, phi_dim, l_max,2*l_max+1))
	for n in range(n_max+1):
		l_row = []
		for l in range(n+1):
			minl = -l
			m_row = []
			for m in range(minl, l+1):
	#			harmonics[:,:, = sph_harm_lm(l,m,phi_grid,theta_grid)		
				m_row.append(sph_harm_lm(l,m,phi_grid,theta_grid))
			l_row.append(m_row)
		harmonics.append(l_row)
	return harmonics

def get_lin_pols(r_grid, n_max):
        pols = []

        #       harmonics = np.zeros((theta_dim, phi_dim, l_max,2*l_max+1))
        for n in range(n_max+1):
                l_row = []
                for l in range(n+1):
                        minl = -l
                        m_row = []
                        for m in range(minl, l+1):
        #                       harmonics[:,:, = sph_harm_lm(l,m,phi_grid,theta_grid)           
                                m_row.append(lin_pols_nl(n,l,r_grid))
                        l_row.append(m_row)
                pols.append(l_row)
        return pols
	
def get_e(r_grid, n_max):
        pols = []

        #       harmonics = np.zeros((theta_dim, phi_dim, l_max,2*l_max+1))
        for n in range(n_max+1):
                l_row = []
                for l in range(n+1):
                        minl = -l
                        m_row = []
                        for m in range(minl, l+1):
        #                       harmonics[:,:, = sph_harm_lm(l,m,phi_grid,theta_grid)         
		#		print(e_nl(n,l,r_grid))  
                                m_row.append(e_nl(n,l,r_grid))
                        l_row.append(m_row)
                pols.append(l_row)
        return pols


def poly_transform(in_grid, n_max, r_grid,phi_grid,theta_grid,harmonics,lin_pols):
    coeffs = []
    factor = 2.0
    
 #   harmonics = get_harmonics(theta_grid, phi_grid,n_max)
#    lin_pols = get_lin_pols(r_grid,n_max)
    for n in range(n_max+1):
	row_l = []
    	for l in range(n+1):
        	row_m = []
        	minl =  -l
        	for m in range(minl, l+1):
            		factor = 2*np.sqrt(np.pi)/(n_max+1)
            		row_m.append(tf.reduce_sum(factor * np.sqrt(2*np.pi)/n_max * tf.cast(in_grid,tf.float32) * tf.constant(np.conj(harmonics[n][l][m-minl]),dtype =tf.float32 )* lin_pols[n][l][m-minl] ,axis = [1,2,3]) )
		row_l.append(row_m)
    	coeffs.append(row_l)

    return coeffs	
	
def sph_conv(f, g, n_max):
    """ Spherical convolution f * g. """
    r_grid = np.linspace(start = 0.5, stop = 1, num = 10)
    phi_grid = np.linspace(start = 0, stop = math.pi, num = 20)
    theta_grid = np.linspace(start = 0, stop = 2*math.pi, num = 20)
    harmonics = get_harmonics(theta_grid, phi_grid,n_max)
    lin_pols = get_lin_pols(r_grid,n_max)
    
    
    
    cf, cg = [poly_transform(x,n_max,r_grid,phi_grid,theta_grid,harmonics,lin_pols) for x in [f, g]]
    e = get_e(r_grid, n_max)
  #  print(e)
    out_dims = tf.shape(cg[0][0][0])[1]
    cfg = tf.zeros( [1,20,20,10,out_dims],dtype=tf.float32)

    for n in range(n_max+1):
        for n_ in range(n+1):
            for l in range(n_+1):
                minl =  -l
                for m in range(minl, l+1):
			va = tf.shape(tf.constant(e[n][l][m-minl],dtype =tf.float32 ))
			annl = tf.get_variable("annl" + str(n)+ str(n_)+str(l)+str(m),[1,1,1,1,1])
		#	vari = tf.get_variable("test" + str(n)+str(n_)+ str(m),va,dtype=tf.float32 )
	#		print(e[n][l][m])
    	#		cfg += tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.matmul(cf[n][l][m-minl],cg[n_][l][m-minl]),axis = 0),axis = 0) ,axis = 0)  * (tf.constant(e[n][l][m-minl],dtype =tf.float32 )-tf.constant(e[n_][l][m-minl],dtype =tf.float32 )) * tf.constant(harmonics[n_][l][m-minl],dtype =tf.float32 )
    			cfg += annl * tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.matmul(cf[n][l][m-minl],cg[n_][l][m-minl]),axis = 0),axis = 0) ,axis = 0)*tf.constant(e[n][l][m-minl],dtype =tf.float32 )  * tf.constant(harmonics[n_][l][m-minl],dtype =tf.float32 ) 
    cfg_norm = 2*np.pi*np.sqrt(4*np.pi / (n_max+1))/(10*20*20) * cfg
    return cfg_norm, cf, cg

def sphconv(inputs, input_dims, output_dims, n_max,scope, use_bias=True):
    
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
	std = 2./(2 * np.pi * np.sqrt((n_max // 2) * (output_dims)))
        with tf.device("cpu:0"):
		weights = tf.get_variable('W',
                                      trainable=True,
                                      initializer=tf.truncated_normal( [input_dims, 20, 20,10, output_dims],stddev=std),
					regularizer=tf.contrib.layers.l2_regularizer(0.000001))
	
	bias = tf.get_variable('b',
                                   trainable=True, initializer=tf.zeros([1, 1, 1, 1,output_dims], dtype=tf.float32))
	
	conv, cf,cg = sph_conv(inputs,weights,n_max)
	conv_ = group_norm(conv,groups = 1,reduction_axes=(1, 2,3))

        return conv_ + bias


input_data = tf.placeholder(tf.float32, [1,20,20,10,1])

conv1 = tf.nn.relu(sphconv(input_data, 1, 2, 5,'l1'))
conv2 = tf.nn.relu(sphconv(conv1, 2, 2, 5,'l2'))
#onv3 = tf.nn.relu(sphconv(conv2, 8, 4, 5,'l3'))
#conv4 = tf.nn.relu(sphconv(conv3, 4, 2, 5,'l4'))
#conv5 = tf.nn.relu(sphconv(conv4, 2, 2, 5,'l5'))
#conv6 = tf.nn.relu(sphconv(conv5, 64, 64, 5,'l6'))
#conv7 = tf.nn.relu(sphconv(conv5, 64, 128, 5,'l7'))
#conv8_ = tf.nn.relu(sphconv(conv6, 128, 128, 5,'l8'))


conv8 = tf.nn.relu(tf.reshape(conv1,[1,-1]))
conv8_ = tf.concat([tf.imag(conv8), tf.real(conv8)], axis = 1)
#with tf.device("gpu:0"):
dense = tf.layers.dense(conv8_,10)
#dense = tf.layers.dense(dense_,10)

outp  =1.0 *  tf.div(
   tf.subtract(
      dense  ,
      tf.reduce_min(dense  , keepdims = True)
   ),
   tf.subtract(
      tf.reduce_max(dense  , keepdims = True),
      tf.reduce_min(dense  , keepdims = True)
   )
)



loss = tf.losses.softmax_cross_entropy(T, dense)
#loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
#with tf.device("gpu:0"):
y_pred = tf.squeeze(tf.argmax(dense,axis = 1),name="output")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(y_pred)

sess = tf.Session()
#run_metadata = tf.RunMetadata()

#with tf.device("cpu:0"):
#                weights1 = tf.get_variable('W',
#                                      trainable=True,
#                                      initializer=tf.truncated_normal( [1, 20, 20,10, 16],stddev=0.1),
#                                        regularizer=tf.contrib.layers.l2_regularizer(0.000001))

#sess.run(weights1, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), run_metadata=run_metadata)
#with open("/home/ram095/sameera/3d_obj/code/3D_object_recognition/run2.txt", "w") as out:
 #	out.write(str(run_metadata))

correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
trainable_vars = tf.trainable_variables()

rot_vars = [var for var in trainable_vars if 'a_' in var.name]
caps_vars =  [var for var in trainable_vars if 'a_' not  in var.name]

#print(trainable_vars)

#names = [n.name for n in tf.get_default_graph().as_graph_def().node]


batch = tf.Variable(0, trainable = False)

learning_rate = tf.train.exponential_decay(
  0.0001,                # Base learning rate.
  batch,  # Current index into the dataset.
  500,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)


#assign_op = batch.assign(0)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-08)
#grads = optimizer.compute_gradients(loss, var_list = rot_vars)
training_op = optimizer.minimize(loss, name="training_op", global_step=batch) #,  var_list = caps_vars)
sess.run(tf.global_variables_initializer())
#sess.run(weights1, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), run_metadata=run_metadata)
#with open("/home/ram095/sameera/3d_obj/code/3D_object_recognition/run2.txt", "w") as out:
#        out.write(str(run_metadata))
	
#	print(r_arr)
#	print(theta_arr)
#	print(phi_arr)
#graph = tf.get_default_graph()
#mat_t1= graph.get_tensor_by_name("pool1/transfrom:0")
#mat_t2= graph.get_tensor_by_name("pool1/transfrom2:0")

#mat_t1p2= graph.get_tensor_by_name("pool4/transfrom:0")
#mat_t2p2= graph.get_tensor_by_name("pool4/transfrom2:0")

def read_datapoint(data, filename):
  points = np.array([[0, 0, 0]])

  for line in data:
    if 'OFF' != line.strip() and len([s for s in line.strip().split(' ')]) == 3:
        if 'bathtub' in filename:
           y = [0]
        if 'bed' in filename:
           y=[1]
        if 'chair' in filename:
           y=[2]
        if 'desk' in filename:
           y=[3]
        if 'dresser' in filename:
           y=[4]
        if 'monitor' in filename:
           y=[5]
        if 'night_stand' in filename:
           y=[6]
        if 'sofa' in filename:
           y=[7]
        if "table" in filename:
           y=[8]
        if 'toilet' in filename:
           y=[9]
        points_list = [float(s) for s in line.strip().split(' ')]
        points = np.append(points,np.expand_dims(np.array(points_list), axis=0), axis = 0)



  return points[2:], y

saver = tf.train.Saver()

training_files = '/home/ram095/sameera/3d_obj/training_files/'
testing_files = '/home/ram095/sameera/3d_obj/testing_files/'


#tf.saved_model.simple_save(sess,
#            "./",
#            inputs={"myInput": input_data},
#            outputs={"myOutput": y})
#saver.restore(sess, "./model.ckpt")
#g = tf.Graph()
#sess = tf.Session(graph=g)
output_graph_def = graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['output'])
with tf.gfile.GFile('graph.pb', "wb") as f:
    f.write(output_graph_def.SerializeToString())


def load_pb(pb_model):
    with tf.gfile.GFile(pb_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def estimate_flops(pb_model):
    graph = load_pb(pb_model)
    with graph.as_default():
        # placeholder input would result in incomplete shape. So replace it with constant during model frozen.
        flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('Model {} needs {} FLOPS after freezing'.format(pb_model, flops.total_float_ops))

#logger = logging.getLogger()    # initialize logging class
#logger.setLevel(logging.DEBUG)  # default log level
#format_ = logging.Formatter("%(asctime)s - %(message)s")    # output format 
#sh = logging.StreamHandler(stream=sys.stdout)    # output to standard output
#sh.setFormatter(format_)
#logger.addHandler(sh)

#estimate_flops("graph.pb")

loss_train_vals = []

loss_vals = []
acc_vals = []
file_names = []
main_itr = 0
i = 0
grad_vals = []
gradients = np.array([])
#np.set_printoptions(threshold=np.nan)
file_list = glob.glob(os.path.join(training_files, '*.off'))
pre_acc_val = 0.0
sess.run(assign_op)
#sess.run(assign_op)
#sess.run(zero_ops)
for j in range(10):
        loss_train_vals = []
        random.shuffle(file_list)
        npoints = 0
        for filename in file_list:

                main_itr = main_itr+1
        #       print(filename)
                f = open(filename, 'r')

                print(filename)
                points_raw, y_annot = read_datapoint(f, filename)
		npoints = npoints + points_raw.shape[0]
		print(points_raw.shape[0])
                print(npoints/main_itr)
        #       print(y_annot)  
                #sorted_idx = np.lexsort(points_raw.T)
                #sorted_data =  point6s_raw[sorted_idx,:
         #       points_raw =np.vstack(set(map(tuple, points_raw_)))
		spherical_points = as_spherical(points_raw)
		print("test1")
		format_in = format_input(spherical_points)
		print(main_itr)
#		conv1 = tf.nn.relu(sphconv(format_in, 1, 64, 5))
	#	test1 = sess.run([dense], feed_dict = {y:y_annot, input_data:format_in})
	#	dense = tf.layers.dense(conv1,10)
	#	print(test1)		
		loss_train,_ = sess.run([loss,training_op], feed_dict = {y:y_annot, input_data:format_in})
                print(j)
                start_time = time.time()
                _ = sess.run(y_pred, feed_dict = {y:y_annot, input_data:format_in})
                end_time = time.time() - start_time
                print(start_time)
                print(end_time)
                loss_train_vals.append(loss_train)
                print(np.mean(loss_train_vals))	
	 	                  

                if main_itr % 15000 == 0:
                      # saver.save(sess, "./model.ckpt")
                        for filename in glob.glob(os.path.join(testing_files, '*.off')):
					f = open(filename, 'r')	
                			points_raw, y_annot = read_datapoint(f, filename)
					spherical_points = as_spherical(points_raw)
					format_in = format_input(spherical_points)
					               
                                        loss_val, acc_val, pred,y_,T_ = sess.run(
                                        [loss, accuracy,y_pred,y,T], feed_dict = {y:y_annot, input_data:format_in})
                                        loss_vals.append(loss_val)
                                        start_time = time.time()
                                        _ = sess.run(y_pred)
                                        end_time = time.time() - start_time
                                        print(start_time)
				        print(end_time)
                                        acc_vals.append(acc_val)
					print(pred)
					print(y_annot)
					print(y_)
					print(T_)
                                        print("validation")
                                        print(acc_val)

                                        acc_val = np.mean(acc_vals)
                                        print(acc_val)
                        if acc_val > pre_acc_val:
                                print("saving best model")
                                saver.save(sess, "./model.ckpt")
                                pre_acc_val = acc_val
                        print(pre_acc_val)
                        loss_vals = []
                        acc_vals = []

                	#saver.save(sess, "./model.ckpt")
                        file_names = []


















