import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import scipy
import numpy as np


def load_image(filename, N):
    img = plt.imread(filename)
    img = resize(img, (N,N))[:,:,0:3]

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    res = np.sqrt(R**2 + G**2 + B**2)
    res = res/np.max(res)
    return res


def kernel(r, N):
    print('N='+str(N))
    x, y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    k = np.zeros((N, N))
    k[np.where(x**2 + y**2 <= r**2)] = 1.0/(N*np.pi*r**2)
    return k


def blurring(image, kernel):
    Z = scipy.fft.fftn(image)
    K = scipy.fft.fftn(kernel)
    u = scipy.fft.fftshift(np.real(scipy.fft.ifftn(Z*K)))
    return u

def add_noise(u, noise_level):
    N = u.shape[0]
    u_noised = u + noise_level*np.max(u)*np.random.randn(N,N)
    return u_noised
