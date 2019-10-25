def slope_mismatch(T, gmm=LB/2):
    
    x, _ = evelope_barrier_state(T, T=T, gmm=gmm)
    
    D1 = np.array([x[0], x[1]])
    D2 = np.array([x[2], x[3]])
    I = np.array([x[4], x[5]])
    D1_I = (D1 - I)/np.linalg.norm(D1 - I)
    D2_I = (D2 - I)/np.linalg.norm(D2 - I)
    
    s_I_D1 = D1_I[1]/D1_I[0]
    s_I_D2 = D2_I[1]/D2_I[0]
    
#    print('%.2f, %.2f, %.2f' %(s_I_D1, s_I_D2, (s_I_D1 - s_I_D2)**2))
    return (s_I_D1 - s_I_D2)**2

def get_max_T(gmm):

    def slope_mismatch_in(T):
        return slope_mismatch(T, gmm=gmm)
    
    assert gmm >= LB/2

    if gmm > (2*pi-(pi/2 - LB/2))/2:
        ub = (pi - LB/2)
    else:
        ub = 2*pi - LB
#    ub = 2*pi - LB
        
    if gmm == LB/2:
        Tmax = (2*pi-LB)/A
    else:
#        res = fmin(slope_mismatch, x0=(2*pi-2*gmm)/A)
#        print(res)
        res = minimize_scalar(slope_mismatch_in, bounds=[(2*pi-2*gmm)/A, ub/A], method='bounded')
        Tmax = res.x

    print('[%.2f, %.2f] gmm in [%.2f, %.2f]'%(gmm, Tmax/A, (2*pi-2*gmm)/A, ub/A))
        
    return Tmax



# originaly in class ValueFunc
    def SS_policy(self, s, n):
        
#        reverse the sign of n when n[:,2] (theta) is positive
        sgn = tf.greater(n[:,2], 0)
        sgn = (-1)*(tf.cast(sgn, tf.float32) - 0.5)/0.5
        sgn = tf.stack((sgn, sgn, sgn), 1)
        n = tf.multiply(n, sgn)
        
        cphi_1 = tf.multiply(n[:,0], tf.sin(s[:,0]))
        sphi_1 = tf.multiply(n[:,2], tf.cos(s[:,0]))
        phi_1 = tf.atan2(sphi_1, cphi_1)
        
        cphi_2 =  tf.multiply(n[:,1], tf.sin(s[:,1]))
        sphi_2 = -tf.multiply(n[:,2], tf.cos(s[:,1]))
        phi_2 = tf.atan2(sphi_2, cphi_2)  
        
        rho_1 = tf.sqrt(tf.square(cphi_1) + tf.square(sphi_1))
        rho_2 = tf.sqrt(tf.square(cphi_2) + tf.square(sphi_2))
        
        cpsi_a = tf.multiply(rho_1, tf.multiply(tf.tan(s[:,0]), tf.cos(phi_1 - s[:,2])))
        cpsi_b = tf.multiply(rho_2, tf.multiply(tf.tan(s[:,1]), tf.cos(phi_2)))
        cpsi = -(cpsi_a + cpsi_b)
        
        spsi_a = tf.multiply(rho_1, tf.multiply(tf.tan(s[:,0]), tf.sin(phi_1 - s[:,2])))
        spsi_b = tf.multiply(rho_2, tf.multiply(tf.tan(s[:,1]), tf.sin(phi_2)))
        spsi = -(spsi_a + spsi_b)
        psi = tf.atan2(spsi, cpsi)
        
        phi = tf.stack([phi_1, phi_2], axis=1)
        
        return phi, psi
    
    def dx_model(self, x, phi, psi):
        c_da1 = tf.divide(tf.square(tf.sin(x[:,0])), r*tf.cos(x[:,0]))
        b_da1 = vd*tf.cos(phi[:,0]) + vi*tf.cos(x[:,2] + psi)
        da1 = tf.multiply(c_da1, b_da1)
        
        c_da2 = tf.divide(tf.square(tf.sin(x[:,1])), r*tf.cos(x[:,1]))
        b_da2 = vd*tf.cos(phi[:,1]) + vi*tf.cos(psi)
        da2 = tf.multiply(c_da2, b_da2)
        
        dtht_1c = tf.sin(x[:,0])/r
        dtht_1b = vd*tf.sin(phi[:,0]) + vi*tf.sin(x[:,2] + psi)
        dtht_2c = tf.sin(x[:,1])/r
        dtht_2b = vd*tf.sin(phi[:,1]) + vi*tf.sin(psi)
        dtht = tf.multiply(dtht_1c, dtht_1b) - tf.multiply(dtht_2c, dtht_2b)
        
        dx = tf.stack([da1, da2, dtht], 1)
        
        return dx