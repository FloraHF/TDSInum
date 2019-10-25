import tensorflow as tf
import os
import numpy as np
from math import pi

from Config import Config
r = Config.CAP_RANGE
vd = Config.VD
vi = Config.VI

class ValueFunc(object):
    
    def __init__(self,
                 lr=Config.LEARNING_RATE,
                 layer_sizes=Config.LAYER_SIZES,
                 act_fns=Config.ACT_FUNCS, 
                 tau=Config.TAU, 
                 read_dir='',
                 save_dir=''):
        
        self.lr = lr
        self.layer_sizes = layer_sizes + [1]
        self.act_fns = act_fns + [None]
        self.tau = tau
        
        self.sess = tf.InteractiveSession()
        self.create_placeholders()
        self.fitting_graph()
        self.fitting_ops()
        self.bd_graph()
        self.learning_graph()
        self.learning_ops()
        self.pde_graph()
        self.pde_ops()
        self.target_update_ops()
        self.saver = tf.train.Saver()

        self.save_file = Config.MODEL_FILE
        if save_dir != '':
            self.save_dir = save_dir + 'value_fn/'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
        self.start_step = 0
        self.sess.run(tf.global_variables_initializer())

        if read_dir != '':
            self.start_step = self.restore_model(read_dir+'value_fn/')      
        
    def V_net(self, x, n, act_fns, name=''):
        
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i in range(len(n)):
                if i == 0:
                    w = tf.get_variable(name='w_'+str(i), shape=(3, n[i]),
                                        initializer=tf.contrib.layers.xavier_initializer())
                    out = tf.matmul(x, w)
                else:
                    w = tf.get_variable(name='w_'+str(i), shape=(n[i-1], n[i]),
                                        initializer=tf.contrib.layers.xavier_initializer()) # tf.random_normal_initializer() tf.contrib.layers.xavier_initializer()
                    out = tf.matmul(out, w)
                    
                b = tf.get_variable(name='b_'+str(i), shape=(n[i]),
                                    initializer=tf.zeros_initializer())
                out = tf.add(out, b)
                if act_fns[i] is not None:
                    out = act_fns[i](out)
        return out 
    
    def create_placeholders(self):
        self.s_ph = tf.placeholder(tf.float32, [None, 3], name='state_ph')
        self.b_ph = tf.placeholder(tf.float32, [None, 3], name='boundary_state_ph')
        self.v_ph = tf.placeholder(tf.float32, [None, 1], name='value_ph')        

    def fitting_graph(self):
        self.v = self.V_net(self.s_ph, self.layer_sizes, self.act_fns, name='value')
        self.n = tf.gradients(self.v, self.s_ph)[0]
        self.fitting_loss = tf.reduce_mean((self.v_ph - self.v)**2)
    
    def fitting_ops(self):
        self.fitter = tf.train.AdamOptimizer(learning_rate=self.lr, name='fit_V')
        self.fit_op = self.fitter.minimize(self.fitting_loss, 
                                           var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'value'))
        
    def fit(self, batch):
        feed_dict = {self.s_ph: batch['states'],
                     self.v_ph: batch['values']}
        self.sess.run(self.fit_op, feed_dict=feed_dict)

    def SS_policy(self, s, n):
        
#        reverse the sign of n when n[:,2] (theta) is negative
        # sgn = tf.less(n[:,2], 0)
        # sgn = (-1)*(tf.cast(sgn, tf.float32) - 0.5)/0.5
        # sgn = tf.stack((sgn, sgn, sgn), 1)
        # n = tf.multiply(n, sgn)
        
        cphi_1 = n[:,0]
        sphi_1 = - tf.divide(n[:,2], s[:,0])
        phi_1 = tf.atan2(sphi_1, cphi_1)
        
        cphi_2 = n[:,1]
        sphi_2 = tf.divide(n[:,2], s[:,1])
        phi_2 = tf.atan2(sphi_2, cphi_2)  
        
        rho_1 = tf.sqrt(tf.square(cphi_1) + tf.square(sphi_1))
        rho_2 = tf.sqrt(tf.square(cphi_2) + tf.square(sphi_2))
        
        cpsi_a = tf.multiply(rho_1, tf.cos(phi_1 - s[:,2]))
        cpsi_b = tf.multiply(rho_2, tf.cos(phi_2))
        cpsi = -(cpsi_a + cpsi_b)
        
        spsi_a = tf.multiply(rho_1, tf.sin(phi_1 - s[:,2]))
        spsi_b = tf.multiply(rho_2, tf.sin(phi_2))
        spsi = -(spsi_a + spsi_b)
        psi = tf.atan2(spsi, cpsi)
        
        phi = tf.stack([phi_1, phi_2], axis=1)
        
        return phi, psi
    
    def dx_model(self, x, phi, psi):

        dd1 = -(vi*tf.cos(x[:,2] + psi) + vd*tf.cos(phi[:,0]))

        dd2 = -(vi*tf.cos(psi)          + vd*tf.cos(phi[:,1]))
        
        dtht_a = vd*tf.sin(phi[:,0]) + vi*tf.sin(x[:,2] + psi)
        dtht_b = vd*tf.sin(phi[:,1]) + vi*tf.sin(psi)
        dtht = tf.divide(dtht_a, x[:,0]) - tf.divide(dtht_b, x[:,1])
        
        dx = tf.stack([dd1, dd2, dtht], 1)
        
        return dx
    
    def dyn_model(self, x, phi, psi, dt=Config.TIME_STEP):
        return x + self.dx_model(x, phi, psi)*dt
        
        
    def bd_graph(self):
        v_bd = self.V_net(self.b_ph, self.layer_sizes, self.act_fns, name='value')
        self.bd_err = tf.reduce_mean((v_bd - self.v_ph)**2)
        
    def learning_graph(self):
        
        self.v_target = self.V_net(self.s_ph, self.layer_sizes, self.act_fns, name='target_value')
        self.n_target = tf.gradients(self.v_target, self.s_ph)[0]
        self.phi, self.psi = self.SS_policy(self.s_ph, self.n_target)
        s_next = self.dyn_model(self.s_ph, self.phi, self.psi)
        self.v_next = self.V_net(s_next, self.layer_sizes, self.act_fns, name='target_value')
        self.td_err = tf.reduce_mean((self.v - self.v_next)**2)
        
        self.learning_loss = self.td_err + self.bd_err

    def learning_ops(self):
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.lr, name='train_V')
        self.train_op = self.trainer.minimize(self.learning_loss, 
                                                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'value'))
    
    def target_update_ops(self):
        source_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'value')
        target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_value')
        self.target_op = [tf.assign(target, (1 - self.tau) * target + self.tau * source)
                                     for target, source in zip(target_params, source_params)]
        
    def train(self, in_batch, bd_batch):
#        print(bd_batch['states'][0])
#        print(bd_batch['values'][0])
        feed_dict={self.s_ph: in_batch['states'],
                   self.b_ph: bd_batch['states'],
                   self.v_ph: bd_batch['values']}
        self.sess.run(self.train_op, feed_dict=feed_dict)
        self.sess.run(self.target_op)
    
    def pde_graph(self):
#        v1 = self.V_net(self.s_ph, self.layer_sizes, self.act_fns, name='target_value')
#        n1 = tf.gradients(v1, self.s_ph)[0]
#        phi, psi = self.SS_policy(self.s_ph, n1)
        dx = self.dx_model(self.s_ph, self.phi, self.psi)
        
#        v2 = self.V_net(self.s_ph, self.layer_sizes, self.act_fns, name='value')
        dv = tf.reduce_sum(tf.gradients(self.v, self.s_ph, grad_ys=dx), -1)
        
        self.pde_loss = tf.reduce_mean(tf.square(dv)) + self.bd_err
        
    def pde_ops(self):
        self.pder = tf.train.AdamOptimizer(learning_rate=self.lr, name='pde_V')
        self.pde_op = self.pder.minimize(self.pde_loss, 
                                                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'value'))
    
    def train_pde(self, in_batch, bd_batch):
        feed_dict={self.s_ph: in_batch['states'],
                   self.b_ph: bd_batch['states'],
                   self.v_ph: bd_batch['values']}
        self.sess.run(self.pde_op, feed_dict=feed_dict)
        self.sess.run(self.target_op)        
    
    def evaluate_td(self, batch):
        return self.sess.run(self.td_err, feed_dict={self.s_ph: batch['states']})

    def evaluate_bd(self, batch):
        feed_dict={self.b_ph: batch['states'],
                   self.v_ph: batch['values']}
        return self.sess.run(self.bd_err, feed_dict=feed_dict)
    
    def evaluate_fitting_err(self, batch):
        feed_dict = {self.s_ph: batch['states'],
                     self.v_ph: batch['values']}
        loss = self.sess.run(self.fitting_loss, feed_dict=feed_dict)
        return loss

    def evaluate_values(self, x):
        return self.sess.run(self.v, feed_dict={self.s_ph: x})[:,0]
    
    def evaluate_value(self, x):
        return self.sess.run(self.v, feed_dict={self.s_ph: x[None]})[0][0]

    def evaluate_target_value(self, x):
        return self.sess.run(self.v_target, feed_dict={self.s_ph: x[None]})[0][0]
    
    def evaluate_next_value(self, x):
        return self.sess.run(self.v_next, feed_dict={self.s_ph: x[None]})[0]
    
    def evaluate_gradient(self, x):
        return self.sess.run(self.n, feed_dict={self.s_ph: x[None]})[0]
    
    def evaluate_target_gradient(self, x):
        return self.sess.run(self.n_target, feed_dict={self.s_ph: x[None]})[0]
    
    def evaluate_policy(self, x):
        phi, psi = self.sess.run([self.phi, self.psi], feed_dict={self.s_ph: x[None]})
        return phi[0], psi[0]
        
    def save_model(self, step):
        self.saver.save(self.sess, self.save_dir+self.save_file, global_step=step)

    def restore_model(self, read_dir):
        print(read_dir)
        dirct = tf.train.latest_checkpoint(read_dir)
        episode = int(dirct.split('-')[-1])
        self.saver.restore(self.sess, dirct)
        return episode