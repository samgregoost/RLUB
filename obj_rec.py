import numpy as np
import math
import os
import tensorflow as tf
import glob
import random
import trimesh



r = tf.placeholder(tf.float32, shape=[None, 1]name="r")
theta = tf.placeholder(tf.float32, shape=[None, 1], name="theta")
phi = tf.placeholder(tf.float32, shape=[None, 1], name="phi")

def spherical_harmonic(m,l):
        return math.pow(-1.0,m) * math.sqrt(((2.0*l + 1.0)*7.0/88.0) * (math.factorial(l-m)*1.0/(math.factorial(l+m)*1.0)))

def radial_poly(rho, m, n):
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

def uv_gen(r, theta, phi, scope):
    with tf.variable_scope(scope):
        r = tf.clip_by_value(r,0,1.0)        

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

        u_0_0_0 = tf.multiply(y_0_0, tf.cos(0.0)) * radial_poly(r,0,0)
        u_0_1_1 = tf.multiply(y_0_1, tf.cos(0.0)) * radial_poly(r,1,1)
        u_1_1_1 = tf.multiply(y_1_1, tf.cos(phi)) * radial_poly(r,1,1)
        u_0_0_2 = tf.multiply(y_0_2, tf.cos(0.0)) * radial_poly(r,0,2)
        u_0_2_2 = tf.multiply(y_0_2, tf.cos(0.0)) * radial_poly(r,2,2)
        u_1_2_2 = tf.multiply(y_1_2, tf.cos(phi)) * radial_poly(r,2,2)
        u_2_2_2 = tf.multiply(y_2_2, tf.cos(2.0*phi)) * radial_poly(r,2,2)
        u_0_1_3 = tf.multiply(y_0_1, tf.cos(0.0)) * radial_poly(r,1,3)
        u_1_1_3 = tf.multiply(y_1_1, tf.cos(phi)) * radial_poly(r,1,3)
        u_0_3_3 = tf.multiply(y_0_3, tf.cos(0.0)) * radial_poly(r,3,3)
        u_1_3_3 = tf.multiply(y_1_3, tf.cos(phi)) * radial_poly(r,3,3)
        u_2_3_3 = tf.multiply(y_2_3, tf.cos(2.0*phi)) * radial_poly(r,3,3)
        u_3_3_3 = tf.multiply(y_3_3, tf.cos(3.0*phi)) * radial_poly(r,3,3)
        u_0_0_4 = tf.multiply(y_0_0, tf.cos(0.0)) * radial_poly(r,0,4)
        u_0_2_4 = tf.multiply(y_0_2, tf.cos(0.0)) * radial_poly(r,2,4)
        u_1_2_4 = tf.multiply(y_1_2, tf.cos(phi)) * radial_poly(r,2,4)
        u_2_2_4 = tf.multiply(y_2_2, tf.cos(2.0*phi)) * radial_poly(r,2,4)
        u_0_4_4 = tf.multiply(y_0_4, tf.cos(0.0))* radial_poly(r,4,4)
        u_1_4_4 = tf.multiply(y_1_4, tf.cos(phi)) * radial_poly(r,4,4)
        u_2_4_4 = tf.multiply(y_2_4, tf.cos(2.0 * phi))* radial_poly(r,4,4)
        u_3_4_4 = tf.multiply(y_3_4, tf.cos(3.0 * phi))* radial_poly(r,4,4)
        u_4_4_4 = tf.multiply(y_4_4, tf.cos(4.0 * phi))* radial_poly(r,4,4)
	u_0_1_5 = tf.multiply(y_0_1, tf.cos(0.0 * phi))* radial_poly(r,1,5)
	u_1_1_5 = tf.multiply(y_1_1, tf.cos(1.0 * phi))* radial_poly(r,1,5)
	u_0_3_5 = tf.multiply(y_0_3, tf.cos(0.0 * phi))* radial_poly(r,3,5)
	u_1_3_5 = tf.multiply(y_1_3, tf.cos(1.0 * phi))* radial_poly(r,3,5)
	u_2_3_5 = tf.multiply(y_2_3, tf.cos(2.0 * phi))* radial_poly(r,3,5)
	u_3_3_5 = tf.multiply(y_3_3, tf.cos(3.0 * phi))* radial_poly(r,3,5)
	u_0_5_5 = tf.multiply(y_0_5, tf.cos(0.0 * phi))* radial_poly(r,5,5)
	u_1_5_5 = tf.multiply(y_1_5, tf.cos(1.0 * phi))* radial_poly(r,5,5)
	u_2_5_5 = tf.multiply(y_2_5, tf.cos(2.0 * phi))* radial_poly(r,5,5)
	u_3_5_5 = tf.multiply(y_3_5, tf.cos(3.0 * phi))* radial_poly(r,5,5)
	u_4_5_5 = tf.multiply(y_4_5, tf.cos(4.0 * phi))* radial_poly(r,5,5)
	u_5_5_5 = tf.multiply(y_5_5, tf.cos(5.0 * phi))* radial_poly(r,5,5)
	u_0_0_6 = tf.multiply(y_0_0, tf.cos(0.0 * phi))* radial_poly(r,0,6)
	u_0_2_6 = tf.multiply(y_0_2, tf.cos(0.0 * phi))* radial_poly(r,2,6)
	u_1_2_6 = tf.multiply(y_1_2, tf.cos(1.0 * phi))* radial_poly(r,2,6)
	u_2_2_6 = tf.multiply(y_2_2, tf.cos(2.0 * phi))* radial_poly(r,2,6)
	u_0_4_6 = tf.multiply(y_0_4, tf.cos(0.0 * phi))* radial_poly(r,4,6)
	u_1_4_6 = tf.multiply(y_1_4, tf.cos(1.0 * phi))* radial_poly(r,4,6)
	u_2_4_6 = tf.multiply(y_2_4, tf.cos(2.0 * phi))* radial_poly(r,4,6)
	u_3_4_6 = tf.multiply(y_3_4, tf.cos(3.0 * phi))* radial_poly(r,4,6)
	u_4_4_6 = tf.multiply(y_4_4, tf.cos(4.0 * phi))* radial_poly(r,4,6)
	u_0_6_6 = tf.multiply(y_0_6, tf.cos(0.0 * phi))* radial_poly(r,6,6)
	u_1_6_6 = tf.multiply(y_1_6, tf.cos(1.0 * phi))* radial_poly(r,6,6)
	u_2_6_6 = tf.multiply(y_2_6, tf.cos(2.0 * phi))* radial_poly(r,6,6)
	u_3_6_6 = tf.multiply(y_3_6, tf.cos(3.0 * phi))* radial_poly(r,6,6)
	u_4_6_6 = tf.multiply(y_4_6, tf.cos(4.0 * phi))* radial_poly(r,6,6)
	u_5_6_6 = tf.multiply(y_5_6, tf.cos(5.0 * phi))* radial_poly(r,6,6)
	u_6_6_6 = tf.multiply(y_6_6, tf.cos(6.0 * phi))* radial_poly(r,6,6)


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
        v_0_1_5 = tf.multiply(y_0_1, tf.sin(0.0 * phi))* radial_poly(r,1,5)
	v_1_1_5 = tf.multiply(y_1_1, tf.sin(1.0 * phi))* radial_poly(r,1,5)
	v_0_3_5 = tf.multiply(y_0_3, tf.sin(0.0 * phi))* radial_poly(r,3,5)
	v_1_3_5 = tf.multiply(y_1_3, tf.sin(1.0 * phi))* radial_poly(r,3,5)
	v_2_3_5 = tf.multiply(y_2_3, tf.sin(2.0 * phi))* radial_poly(r,3,5)
	v_3_3_5 = tf.multiply(y_3_3, tf.sin(3.0 * phi))* radial_poly(r,3,5)
	v_0_5_5 = tf.multiply(y_0_5, tf.sin(0.0 * phi))* radial_poly(r,5,5)
	v_1_5_5 = tf.multiply(y_1_5, tf.sin(1.0 * phi))* radial_poly(r,5,5)
	v_2_5_5 = tf.multiply(y_2_5, tf.sin(2.0 * phi))* radial_poly(r,5,5)
	v_3_5_5 = tf.multiply(y_3_5, tf.sin(3.0 * phi))* radial_poly(r,5,5)
	v_4_5_5 = tf.multiply(y_4_5, tf.sin(4.0 * phi))* radial_poly(r,5,5)
	v_5_5_5 = tf.multiply(y_5_5, tf.sin(5.0 * phi))* radial_poly(r,5,5)
	v_0_0_6 = tf.multiply(y_0_0, tf.sin(0.0 * phi))* radial_poly(r,0,6)
	v_0_2_6 = tf.multiply(y_0_2, tf.sin(0.0 * phi))* radial_poly(r,2,6)
	v_1_2_6 = tf.multiply(y_1_2, tf.sin(1.0 * phi))* radial_poly(r,2,6)
	v_2_2_6 = tf.multiply(y_2_2, tf.sin(2.0 * phi))* radial_poly(r,2,6)
	v_0_4_6 = tf.multiply(y_0_4, tf.sin(0.0 * phi))* radial_poly(r,4,6)
	v_1_4_6 = tf.multiply(y_1_4, tf.sin(1.0 * phi))* radial_poly(r,4,6)
	v_2_4_6 = tf.multiply(y_2_4, tf.sin(2.0 * phi))* radial_poly(r,4,6)
	v_3_4_6 = tf.multiply(y_3_4, tf.sin(3.0 * phi))* radial_poly(r,4,6)
	v_4_4_6 = tf.multiply(y_4_4, tf.sin(4.0 * phi))* radial_poly(r,4,6)
	v_0_6_6 = tf.multiply(y_0_6, tf.sin(0.0 * phi))* radial_poly(r,6,6)
	v_1_6_6 = tf.multiply(y_1_6, tf.sin(1.0 * phi))* radial_poly(r,6,6)
	v_2_6_6 = tf.multiply(y_2_6, tf.sin(2.0 * phi))* radial_poly(r,6,6)
	v_3_6_6 = tf.multiply(y_3_6, tf.sin(3.0 * phi))* radial_poly(r,6,6)
	v_4_6_6 = tf.multiply(y_4_6, tf.sin(4.0 * phi))* radial_poly(r,6,6)
	v_5_6_6 = tf.multiply(y_5_6, tf.sin(5.0 * phi))* radial_poly(r,6,6)
	v_6_6_6 = tf.multiply(y_6_6, tf.sin(6.0 * phi))* radial_poly(r,6,6)

       # V = tf.concat([v_0_0_0, v_0_1_1, v_1_1_1, v_0_0_2, v_0_2_2, v_1_2_2, v_2_2_2,
       #                 v_0_1_3, v_1_1_3, v_0_3_3, v_1_3_3, v_2_3_3, v_3_3_3,
       #                         v_0_0_4, v_0_2_4, v_1_2_4, v_2_2_4, v_0_4_4, v_1_4_4, v_2_4_4, v_3_4_4, v_4_4_4] ,  axis=1)
        
	V = tf.concat([v_0_0_0, v_0_1_1, v_1_1_1, v_0_0_2, v_0_2_2, v_1_2_2, v_2_2_2, 
			v_0_1_3, v_1_1_3, v_0_3_3, v_1_3_3, v_2_3_3, v_3_3_3, 
				v_0_0_4, v_0_2_4, v_1_2_4, v_2_2_4, v_0_4_4, v_1_4_4, v_2_4_4, v_3_4_4, v_4_4_4, v_0_1_5,v_1_1_5,v_0_3_5,v_1_3_5,v_2_3_5,v_3_3_5, v_0_5_5,v_1_5_5,v_2_5_5,v_3_5_5,v_4_5_5,v_5_5_5, v_0_0_6,v_0_2_6,v_1_2_6,v_2_2_6,v_0_4_6,v_1_4_6,v_2_4_6,v_3_4_6,v_4_4_6,v_0_6_6,v_1_6_6,v_2_6_6,v_3_6_6,v_4_6_6,v_5_6_6,v_6_6_6] ,  axis=1)

#	U = tf.concat([u_0_0_0, u_0_1_1, u_1_1_1, u_0_0_2, u_0_2_2, u_1_2_2, u_2_2_2,
#                        u_0_1_3, u_1_1_3, u_0_3_3, u_1_3_3, u_2_3_3, u_3_3_3,
#                                u_0_0_4, u_0_2_4, u_1_2_4, u_2_2_4, u_0_4_4, u_1_4_4, u_2_4_4, u_3_4_4, u_4_4_4] ,  axis=1)
 
	U = tf.concat([u_0_0_0, u_0_1_1, u_1_1_1, u_0_0_2, u_0_2_2, u_1_2_2, u_2_2_2, 
                        u_0_1_3, u_1_1_3, u_0_3_3, u_1_3_3, u_2_3_3, u_3_3_3, 
                                u_0_0_4, u_0_2_4, u_1_2_4, u_2_2_4, u_0_4_4, u_1_4_4, u_2_4_4, u_3_4_4, u_4_4_4, u_0_1_5,u_1_1_5,u_0_3_5,u_1_3_5,u_2_3_5,u_3_3_5, u_0_5_5,u_1_5_5,u_2_5_5,u_3_5_5,u_4_5_5,u_5_5_5, v_0_0_6,v_0_2_6,v_1_2_6,v_2_2_6,v_0_4_6,v_1_4_6,v_2_4_6,v_3_4_6,v_4_4_6,v_0_6_6,v_1_6_6,v_2_6_6,v_3_6_6,v_4_6_6,v_5_6_6,v_6_6_6] ,  axis=1)
       
        X_ = tf.concat([U,V] ,  axis=1)
        return X_

def enf_sym(c):
    c000 = tf.slice(c,[0,0,0],[-1,1,-1])
    c011 = tf.slice(c,[0,1,0],[-1,1,-1])



X_ = uv_gen(r, theta, phi, scope)

X_T  = 0.0001 * tf.transpose(X_,[1,0] )


X_cal_1, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,X_),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,X_),3.0*tf.eye(100)-tf.matmul(x,X_))),x),i+1)
    ,(X_T, 0))


X_var0 = tf.slice(X_,[0,0],[100,100])
X_var1 = tf.slice(uv_gen(r+0.1, theta, phi, "rv1"),[0,0],[100,100]) #(was 100,44)
X_var2 = tf.slice(uv_gen(r+0.2, theta, phi, "rv2"),[0,0],[100,100])
X_var3 = tf.slice(uv_gen(r+0.3, theta, phi, "rv3"),[0,0],[100,100])
X_var4 = tf.slice(uv_gen(r+0.4, theta, phi, "rv4"),[0,0],[100,100])
X_var5 = tf.slice(uv_gen(r+0.5, theta, phi, "rv5"),[0,0],[100,100])
X_var6 = tf.slice(uv_gen(r+0.6, theta, phi, "rv6"),[0,0],[100,100])
X_var7 = tf.slice(uv_gen(r+0.7, theta, phi, "rv7"),[0,0],[100,100])
X_var8 = tf.slice(uv_gen(r+0.8, theta, phi, "rv8"),[0,0],[100,100])
X_var9 = tf.slice(uv_gen(r+0.9, theta, phi, "rv9"),[0,0],[100,100])

X_var = tf.concat([X_var0,X_var1,X_var2,X_var3,X_var4,X_var5,X_var6,X_var7,X_var8,X_var9],axis = 0)


X_var0T  = 0.0001 * tf.transpose(X_var0,[1,0] )
X_var1T  = 0.0001 * tf.transpose(X_var1,[1,0] )
X_var2T  = 0.0001 * tf.transpose(X_var2,[1,0] )
X_var3T  = 0.0001 * tf.transpose(X_var3,[1,0] )
X_var4T  = 0.0001 * tf.transpose(X_var4,[1,0] )
X_var5T  = 0.0001 * tf.transpose(X_var5,[1,0] )
X_var6T  = 0.0001 * tf.transpose(X_var6,[1,0] )
X_var7T  = 0.0001 * tf.transpose(X_var7,[1,0] )
X_var8T  = 0.0001 * tf.transpose(X_var8,[1,0] )
X_var9T  = 0.0001 * tf.transpose(X_var9,[1,0] )



X_cal_var0, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,X_var0),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,X_var0),3.0*tf.eye(100)-tf.matmul(x,X_var0))),x),i+1)
    ,(X_varT, 0))#(44,100)

X_cal_var1, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,X_var1),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,X_var1),3.0*tf.eye(100)-tf.matmul(x,X_var1))),x),i+1)
    ,(X_varT, 0))#(44,100)

X_cal_var2, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,X_var2),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,X_var2),3.0*tf.eye(100)-tf.matmul(x,X_var2))),x),i+1)
    ,(X_varT2, 0))#(44,100)

X_cal_var3, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,X_var3),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,X_var3),3.0*tf.eye(100)-tf.matmul(x,X_var3))),x),i+1)
    ,(X_varT3, 0))#(44,100)

X_cal_var4, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,X_var4),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,X_var4),3.0*tf.eye(100)-tf.matmul(x,X_var4))),x),i+1)
    ,(X_varT4, 0))#(44,100)

X_cal_var5, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,X_var5),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,X_var5),3.0*tf.eye(100)-tf.matmul(x,X_var5))),x),i+1)
    ,(X_varT, 0))#(44,100)

X_cal_var6, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,X_var6),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,X_var6),3.0*tf.eye(100)-tf.matmul(x,X_var6))),x),i+1)
    ,(X_varT6, 0))#(44,100)

X_cal_var7, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,X_var7),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,X_var7),3.0*tf.eye(100)-tf.matmul(x,X_var7))),x),i+1)
    ,(X_varT7, 0))#(44,100)

X_cal_var8, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,X_var8),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,X_var8),3.0*tf.eye(100)-tf.matmul(x,X_var8))),x),i+1)
    ,(X_varT8, 0))#(44,100)

X_cal_var9, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(100) + 1.0/4.0 * tf.matmul(tf.eye(100)-tf.matmul(x,X_var9),tf.matmul(3.0*tf.eye(100)-tf.matmul(x,X_var9),3.0*tf.eye(100)-tf.matmul(x,X_var9))),x),i+1)
    ,(X_varT9, 0))#(44,100)

C = tf.tile(tf.expand_dims(tf.matmul(X_cal_1,r), axis =0), [16,1,1]) #(16,44,1)

c_format = tf.tile(tf.expand_dims(C,axis = 1),[1,10,1,1])#(16,10,44,1)

X_cal_var_tiled0 = tf.expand_dims(tf.tile(tf.expand_dims(X_cal_var0, axis = 0),[16,1,1]), axis = 1) (#16,1,44,100)
X_cal_var_tiled1 = tf.expand_dims(tf.tile(tf.expand_dims(X_cal_var1, axis = 0),[16,1,1]), axis = 1) (#16,1,44,100)
X_cal_var_tiled2 = tf.expand_dims(tf.tile(tf.expand_dims(X_cal_var2, axis = 0),[16,1,1]), axis = 1) (#16,1,44,100)
X_cal_var_tiled3 = tf.expand_dims(tf.tile(tf.expand_dims(X_cal_var3, axis = 0),[16,1,1]), axis = 1) (#16,1,44,100)
X_cal_var_tiled4 = tf.expand_dims(tf.tile(tf.expand_dims(X_cal_var4, axis = 0),[16,1,1]), axis = 1) (#16,1,44,100)
X_cal_var_tiled5 = tf.expand_dims(tf.tile(tf.expand_dims(X_cal_var5, axis = 0),[16,1,1]), axis = 1) (#16,1,44,100)
X_cal_var_tiled6 = tf.expand_dims(tf.tile(tf.expand_dims(X_cal_var6, axis = 0),[16,1,1]), axis = 1) (#16,1,44,100)
X_cal_var_tiled7 = tf.expand_dims(tf.tile(tf.expand_dims(X_cal_var7, axis = 0),[16,1,1]), axis = 1) (#16,1,44,100)
X_cal_var_tiled8 = tf.expand_dims(tf.tile(tf.expand_dims(X_cal_var8, axis = 0),[16,1,1]), axis = 1) (#16,1,44,100)
X_cal_var_tiled9 = tf.expand_dims(tf.tile(tf.expand_dims(X_cal_var9, axis = 0),[16,1,1]), axis = 1) (#16,1,44,100)

X_cal_var_tiled = tf.concat([X_cal_var_tiled0,X_cal_var_tiled1,X_cal_var_tiled2.X_cal_var_tiled3,X_cal_var_tiled4,X_cal_var_tiled5,X_cal_var_tiled6,X_cal_var_tiled7,X_cal_var_tiled8,X_cal_var_tiled9], axis = 1]) #(16,10,44,100)

surface_var_ = tf.random_uniform(
    [16,100,1],
    minval=0.0001,
    maxval=1.0,
    dtype=tf.float32,
    seed=None,
    name="a_surf"
)

surface_var = tf.variable(surface_var_, name = "surfvar")#(16,100,1)

surface_var_ed = tf.expand_dims(surface_var, axis =1) #(16,1,100,1)

surface_var_tiled = tf.tile(surface_var_ed,[1,10,1,1]) #(16,10,100,1)

Cf = tf.matmul(X_cal_var_tiled,surface_var_tiled) #(16,10,44,1)



expanded_f_map  = tf.matmul(c_format, tf.transpose(Cf, [0,1,3,2])) #(16,10,44,44)


weight_map_ = tf.random_uniform(
    [16,10,100,100],
    minval=0.0001,
    maxval=1.0,
    dtype=tf.float32,
    seed=None,
    name="a_weight"
)

weight_map =  tf.variable(weight_map_, name = "weightvar")

weighted_f_map = tf.multiply(expanded_f_map , weight_map)

weighted_f_row = tf.reduce_sum(weighted_f_map, axis = 2) #(16,10,44)

weighted_f_column = tf.reduce_sum(weighted_f_map, axis = 3) #(16,10,44) 

concat_vec = tf.reshape(tf.concat([weighted_f_column, weighted_f_row ],axis = 2),[1,-1])#(16,10,88)

output = tf.layers.dense(concat_vec, 10)


y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
T = tf.one_hot(y, depth=10, name="T")

loss = tf.losses.softmax_cross_entropy(T, dense)

y_pred = tf.squeeze(tf.argmax(dense,axis = 1),name="output")

sess = tf.Session()


correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
trainable_vars = tf.trainable_variables()

learning_rate = tf.train.exponential_decay(
  0.0001,                # Base learning rate.
  batch,  # Current index into the dataset.
  500,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-08)
#grads = optimizer.compute_gradients(loss, var_list = rot_vars)
training_op = optimizer.minimize(loss, name="training_op", global_step=batch) #,  var_list = caps_vars)
sess.run(tf.global_variables_initializer())
#sess.run(weights1, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), run_metadata=run_metadata)
#with open("/home/ram095/sameera/3d_obj/code/3D_object_recognition/run2.txt", "w") as out:
#        out.write(str(run_metadata))

#       print(r_arr)
#       print(theta_arr)
#       print(phi_arr)
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


def sample_points(file_name):
    mesh = trimesh.load(filename)
    mesh.remove_degenerate_faces()
    mesh.fix_normals()
    mesh.fill_holes()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    mesh.apply_translation(-mesh.centroid)
    r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
    mesh.apply_scale(1 / r)
    loc = render_model(mesh,sgrid)
#               print(filename)
    k = cart2sph(loc)

    return k

loss_train_vals = []

loss_vals = []
acc_vals = []
file_names = []
main_itr = 0
i = 0
grad_vals = []

sgrid = make_sgrid_(300)
file_list = glob.glob(os.path.join(training_files, '*.off'))
pre_acc_val = 0.0

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
         #       npoints = npoints + points_raw.shape[0]
		k = sample_points(filename)
		
		
            #    print(points_raw.shape[0])
            #    print(npoints/main_itr)
        #       print(y_annot)  
                #sorted_idx = np.lexsort(points_raw.T)
                #sorted_data =  point6s_raw[sorted_idx,:
         #       points_raw =np.vstack(set(map(tuple, points_raw_)))
             #   spherical_points = as_spherical(points_raw)
               # print("test1")
              #  format_in = format_input(spherical_points)
                print(main_itr)
#               conv1 = tf.nn.relu(sphconv(format_in, 1, 64, 5))
        #       test1 = sess.run([dense], feed_dict = {y:y_annot, input_data:format_in})
        #       dense = tf.layers.dense(conv1,10)
        #       print(test1)            
                loss_train,_ = sess.run([loss,training_op], feed_dict = {y:y_annot, r:k[:,2].reshape(-1,1), theta:k[:0].reshape(-1,1),phi:k[:,1].reshape(-1,1)})
                print(j)
                start_time = time.time()
                _ = sess.run(y_pred, feed_dict = {y:y_annot, input_data:format_in})
                end_time = time.time() - start_time
                print(start_time)
                print(end_time)
                loss_train_vals.append(loss_train)
                print(np.mean(loss_train_vals))

                if main_itr % 1500 == 0:
                      # saver.save(sess, "./model.ckpt")
                        for filename in glob.glob(os.path.join(testing_files, '*.off')):
                                        f = open(filename, 'r')
                                        points_raw, y_annot = read_datapoint(f, filename)
                                    #    spherical_points = as_spherical(points_raw)
                                    #    format_in = format_input(spherical_points)
					k = sample_points(filename)
                                        loss_val, acc_val, pred,y_,T_ = sess.run(
                                        [loss, accuracy,y_pred,y,T], feed_dict = {y:y_annot,r:k[:,2].reshape(-1,1), theta:k[:0].reshape(-1,1),phi:k[:,1].reshape(-1,1)})
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








	'''
theta_var_ = tf.random_uniform(
    [16,100,1],
    minval=0.0001,
    maxval=math.pi,
    dtype=tf.float32,
    seed=None,
    name="a_theta"
)


theta_var = tf.Variable(theta_var_, name="a_t")

phi_var_ = tf.random_uniform(
    [16,100,1],
    minval=0.0001,
    maxval=2*math.pi,
    dtype=tf.float32,
    seed=None,
    name="a_phi"
)

phi_var = tf.Variable(phi_var_, name="a_p")

r_var_ = tf.random_uniform(
    [16,100,1],
    minval=0.0001,
    maxval=1,
    dtype=tf.float32,
    seed=None,
    name="a_r"
)

r_var = tf.Variable(r_var_, name="a_rrvar", trainable = True)

X_var = uv_gen(theta_var_, phi_var_, scope)

X_Tvar  = 0.0001 * tf.transpose(X_var, [1,0,2])

X_cal_var, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 2 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(44) + 1.0/4.0 * tf.matmul(tf.eye(44)-tf.matmul(x,X_var),tf.matmul(3.0*tf.eye(44)-tf.matmul(x,X_var),3.0*tf.eye(44)-tf.matmul(x,X_var))),x),i+1)
        ,(X_Tvar, 0))






file_list = glob.glob(os.path.join(training_files, '*.npy'))

for j in range(10):
        loss_train_vals = []
        random.shuffle(file_list)
        for filename in file_list:
            points = 
            
'''
