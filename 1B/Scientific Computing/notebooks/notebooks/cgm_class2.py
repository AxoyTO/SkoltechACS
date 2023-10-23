import numpy as np 
import matplotlib.pyplot as plt
import time
import threading


import itertools
from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes
from multiprocessing import Lock
from multiprocessing import Process
import multiprocessing
import sharedmem 
import numexpr as ne
# import keyboard
# 

class PCGMs:
    def __init__(self, functional, *args):
        self.F = functional
        self.args   = args
        # Default arguments
        self.P              = None
        self.step_min       = 1e-7 
        self.step_max       = 1e7
        self.nproc          = 10
        self.initial_step   = 1e1
        self.freq           = 500
        self.csEPS          = 1e-4
        self.norm_grad      = self.norm
        print('Minimizer class initialized. Len(args[0]) = '+str(len(args[0])))

    def minimize_cs(self, N_iter, x0, initial_step):
        x= x0
        f, g = self.F(x, *self.args[0])
        f0 = f

        n=self.norm(g)

        print('CGM: Initial funcional value: '+str(f))
        print('CGM: Initial gradient norm: '+str(n))

        b=0; k=0

        start = time.time()
        step=initial_step

        while True:
            if N_iter != 0 and k > N_iter:
                print('CGM BREAK: maximum number of iterations reached')
                break


            if self.P is not None:
                g_try = self.P(g, *self.args[0])
            else:
                g_try = g
            x_try = ne.evaluate('x - step*g_try')
#             if self.P != 0:
#                 x_try = self.P(x_try, *self.args[0])
            f_try, g_try = self.F(x_try, *self.args[0])

            if f_try > f: 
                step = step/3
                g = 0
                print('CGM: Reducing the step to be '+str(step))

                if step < self.step_min:
                    print('CGM BREAK: step < step_min')
                    break
                continue

            x = x_try
            f = f_try

            n_try = self.norm_grad(g_try, *self.args[0])
            b = n_try**2/n**2
            g = g_try + b*g
            n = n_try

            k += 1
            if float(k)%self.freq == 0: 
                end = time.time()
                print('CGM: iteration '+str(k)+'; step='+str(step)+'; f='+str(f) + '; Time elapsed: '+str(end-start))
                start = time.time()

        print('Number of iterations: ............. '+str(k))
        print('Initial functional value: ......... '+str(f0))
        print('Last Functional value: .............'+str(f))
        print('_________________________________________________')
        return x, f


    def minimize(self, N_iter, x0, initial_step):
        x= x0
        st = time.time()
        f, g = self.F(x, *self.args[0])

        f0=f
        n_old=self.norm(g)

        f_old=f
        g_old=g
        n=0; b=0; k=0
        x_old=x
        print('CGM: Initial funcional value: '+str(f))
        print('CGM: Initial gradient norm: '+str(n_old))


        # Creating the graphs 
        vals=np.zeros(1)
        grads=np.zeros(1)
        vals[0]=f
        grads[0]=n_old

        start = time.time()
        step=initial_step
        while True:
            try: 
                if keyboard.is_pressed('alt + q'):
                    print('Exiting...')
                    break
            except:
                lala=1




            if N_iter != 0 and k > N_iter:
                print('CGM BREAK: maximum number of iterations reached')
                break

#             FIXME
#             FIXME
#             FIXME
#             FIXME
#             What the shit is here with projections? What did I mean? 
#             FIXME
            if self.step_mode=='Golden':
                step = self.calc_step_mp(x, g)
#             elif self.step_mode=='Constant':
#                 step = initial_step

            if self.P != 0:
                g = self.P(g, *self.args[0])
            x_try = ne.evaluate('x - step*g')
#             if self.P != 0:
#                 x_try = self.P(x_try, *self.args[0])
#             FIXME
#             FIXME
            f_try, g_try = self.F(x_try, *self.args[0])

            if f_try > f_old: 
                if self.step_mode=='Golden':
                    print('Unable to calculate the step')
                    print('f_try='+str(f_try)+' > f_old='+str(f_old))
                    break
                step = step/3
                g_old=0
                print('CGM: Reducing the step to be '+str(step))
                if step < self.step_min:
                    print('CGM BREAK: step < step_min')
                    break
                else:
                    continue

            k += 1
            x_old = x
#             print(g)
#             print(g_old)
#             print(b)
            x = x_try

            f_old = f
            f = f_try
            g_old = g
            g = g_try
            n = self.norm_grad(g, *self.args[0])
            b=n**2/n_old**2

            n_old=n
            g = g + b*g_old
#             print(g)
#             print(g_old)
#             print(b)
#             quit()


            if float(k)%self.freq == 0: 
                end = time.time()
                print('CGM: iteration '+str(k)+'; step='+str(step)+'; f='+str(f) + '; Time elapsed: '+str(end-start))
                start = time.time()

        print('Number of iterations: ............. '+str(k))
        print('Initial functional value: ......... '+str(f0))
        print('Last Functional value: .............'+str(f_old))
        print('_________________________________________________')
        return x_old, f_old

########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
# Auxiliary methods
    def norm(self, x, *args):
#         res = np.sqrt(np.sum(np.abs(x)**2)/x.size)
        res = np.linalg.norm(x)
        return res

    def set_norm(self, norm_grad):
        self.norm_grad = norm_grad

    def print_freq(self, freq=500):
        self.freq=freq

    def set_csEPS(self, eps):
        self.csEPS = eps

    def set_P(self, P):
        self.P = P

#     def set_min_step(self, step_min):
#         self.step_min = step_min
    def set_step_interval(self, step_min, step_max):
        self.step_min = step_min
        self.step_max = step_max
    



    def calc_step_mp_worker(self, x, g, a, vals, steps, block_size, lock, proc_num):
        b = a + block_size if a + block_size <= self.step_max else self.step_max

        phi=(1+np.sqrt(5))/2
        step1=b-(b-a)/phi
        step2=a+(b-a)/phi

        init=self.F(x, *self.args[0])
        k=0
        while True:
            k=k+1
            f1,_=self.F(x-step1*g, *self.args[0])
            f2,_=self.F(x-step2*g, *self.args[0])
            if f1>=f2: 
                a=step1
                step1=step2
                step2=a+(b-a)/phi
            else:
                b=step2
                step2=step1
                step1=b-(b-a)/phi
            mod=np.fabs(b-a) 
#             if mod <= block_size/1000 or b-a==0 or k>1000:
            if mod <= self.csEPS or b-a==0 or k>1000:
                break
        step=(a+b)/2
        val,_ = self.F(x-step*g, *self.args[0])

        vals[proc_num] = val
        steps[proc_num] = step
        return step


    def calc_step_mp(self, x, g):

        block_size = (self.step_max - self.step_min)/self.nproc
        point = sharedmem.copy(x)
        grad = sharedmem.copy(g)
        vals = np.zeros(self.nproc)
        steps = np.zeros(self.nproc)
        vals_mp = sharedmem.copy(vals)
        steps_mp = sharedmem.copy(steps)

        procs = []
        lock = Lock()
        for process in range(self.nproc):
            start = self.step_min + process*block_size
            proc = Process(target = self.calc_step_mp_worker, args=(point, grad, start, vals_mp, steps_mp, block_size, lock, process))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()
        
        steps = sharedmem.copy(steps_mp)
        vals = sharedmem.copy(vals_mp)
        valmin = np.argmin(vals)
        step = steps[valmin]
        return step
    

