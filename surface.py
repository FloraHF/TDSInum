import numpy as np

from Value import ValueFunc
from database import StateValueBuffer, InnerStateGenerator, BoundaryStateGenerator
from sampler import sample_SS
from Config import Config

def fit_nnV(Vfn=None, lr=1e-3, 
            n_steps=20000, batch_size=5000, 
            print_freq=50, save_freq=100, 
            read_dir='', save_dir=''):
    
    if Vfn is None:
        Vfn = ValueFunc(lr=lr, read_dir=read_dir, save_dir=save_dir)
    Database = StateValueBuffer()
#    bd_database = BoundaryCondPool()

    print('start fitting')
    for step in range(Vfn.start_step, Vfn.start_step+n_steps):
        
        if step == Vfn.start_step:
            v, x, s, d = sample_SS(gmm=Config.LB/2)
            Database.add_surface(v, d)
        elif (step - Vfn.start_step) % 500 == 0:
            print('more data')
            v, x, s, d = sample_SS()
            Database.add_surface(v, d)
            batch_size += 300
        
        batch = Database.random_batch(batch_size)
        Vfn.fit(batch)
        loss = Vfn.evaluate_fitting_err(batch)

        with open(save_dir+'loss.csv', 'a') as f:
            f.write(str(loss) + ',\n')

        if step % print_freq == 0 or step == 0:
            print('step %d loss: %.8f' % (step, loss))

        if step % save_freq == 0 or step == n_steps-1:
            Vfn.save_model(step)

    print('end of fitting')

def learn_nnV(n_steps=1200, batch_size=500, lr=1e-4,
              print_freq=50, save_freq=100, 
              fit_frac=1/100, read_dir='', save_dir=''):
    
    Vfn = ValueFunc(lr=lr, read_dir=read_dir, save_dir=save_dir)
    in_pool = InnerStateGenerator()
    bd_pool = BoundaryStateGenerator()
    fit_step = 0
        
    if read_dir == '' and fit_frac > 0:
        fit_step += int(n_steps*fit_frac)
        fit_nnV(Vfn=Vfn, lr=lr,
                n_steps=fit_step, batch_size=batch_size, 
                print_freq=print_freq, save_freq=save_freq, 
                read_dir=read_dir, save_dir=save_dir)
    
    print('start value iteration')
    for step in range(Vfn.start_step+fit_step+1, Vfn.start_step+n_steps):
        
        in_batch = in_pool.random_batch(batch_size)
        bd_batch = bd_pool.random_batch(int(batch_size/10))
        
        Vfn.train_pde(in_batch, bd_batch)
        
        in_loss = Vfn.evaluate_td(in_batch)
        bd_loss = Vfn.evaluate_bd(bd_batch)
        # print(bd_batch['states'][0], bd_batch['states'][0][-1], bd_batch['values'][0])
        
        loss = in_loss + bd_loss
        
        with open(save_dir+'loss.csv', 'a') as f:
            f.write(','.join(list(map(str, [loss, in_loss, bd_loss]))) + '\n')

        if step % print_freq == 0 or step == 0:
            print('step %d: td loss %.5f, bd loss %.5f' % (step, in_loss, bd_loss))   

        if step % save_freq == 0 or step == n_steps-1:
            Vfn.save_model(step)

    print('end of training')
             
if __name__ == '__main__':

#
    fit_nnV(lr=1e-2, read_dir='', save_dir='fitted_v/')
#    
    # learn_nnV(lr=1e-3, batch_size=10000, n_steps=25000, fit_frac=1/10000, read_dir='', save_dir='learned_v/')
#    
    
