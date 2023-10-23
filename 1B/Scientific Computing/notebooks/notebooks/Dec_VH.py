import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from auxiliary import load_image, kernel
import scipy
from cgm_class2 import *

def blurring(image, kernel):
    res = scipy.fft.ifftn(scipy.fft.fftn(image)*scipy.fft.fftn(kernel))
    return res


def norm(z, *args):
    return np.linalg.norm(z)

def fe(z, epsilon = 1e-4):
    return np.sqrt(z**2 + (epsilon/z.shape[0]**2)**2)

def Phi(z, *p):
    K = p[0]
    K_conj = np.conjugate(K)
    U = p[1]
    
    N = U.shape[0]

    Z = scipy.fft.fftn(z)
    resid = K*Z - U
    phi = np.linalg.norm(resid)

    grad_Phi = 2*scipy.fft.ifftn(K_conj*resid)
    return phi, grad_Phi


def Omega(z):
    zz =z
    z_ip1j = np.roll(zz, -1, axis=0)
    z_ip1j[z_ip1j.shape[0]-1,:] = 0
    
    z_ijp1 = np.roll(zz, -1, axis=1)
    z_ijp1[:,z_ijp1.shape[1]-1] = 0
    
    z_ip1jp1 = np.roll(z_ip1j, -1, axis=1)
    z_ip1jp1[:,z_ip1jp1.shape[1]-1] = 0
    
    hz = z_ip1jp1 - z_ip1j - z_ijp1 + zz

    O = np.sum(fe(hz)) # The functional value

    hz_im1j = np.roll(hz, 1, axis=0)
    hz_im1j[0,:] = 0
    
    hz_ijm1 = np.roll(hz, 1, axis=1)
    hz_ijm1[:,0] = 0
    
    hz_im1jm1 = np.roll(hz_im1j, 1, axis=1)
    hz_im1jm1[:,0] = 0
    
    dO = hz/fe(hz) + hz_im1jm1/fe(hz_im1jm1) - hz_ijm1/fe(hz_ijm1) - hz_im1j/fe(hz_im1j) # The gradient
    return O, dO

def cost_functional(x, *p):
    alpha = p[2]
    phi, grad_phi = Phi(x,*p)
    O, dO = Omega(x)
    M = phi + alpha*O
    dM = grad_phi + alpha*dO
    return M, dM



def optimize(u, r, alpha, step, N_item):
    N = u.shape[0]

    k = kernel(r, N)

    K = scipy.fft.fftn(k)
    U = scipy.fft.fftn(u)

    z0 = np.zeros(u.shape)

    args = [K,U,alpha]

    opt = PCGMs(cost_functional, args)
    opt.set_norm(norm)
    opt.set_step_interval(1e-10, 1e5)
    opt.step_mode = 'Constant'
    opt.print_freq(1)
    opt.set_csEPS(1e1)
#         opt.set_P(P)

    z_approx, cost = opt.minimize_cs(N_iter, z0, step)


    np.save('z_approx.npy', z_approx)
    coeff = 0.3

    z_approx = np.clip(z_approx, coeff*np.max(z_approx), (1-coeff)*np.max(z_approx))


    plt.imshow(scipy.fft.fftshift(np.real(z_approx)))
    plt.show()
    return z_approx



N = 512
# r = 0.1
r = 0.077
# z = load_image('test.png', N)
u = load_image('test3.png', N)

k = kernel(r, N)

# u = blurring(z, k)

alpha = 1e1
step = 1e-4
N_iter = 1000


z_approx = optimize(u, r, alpha, step, N_iter)

