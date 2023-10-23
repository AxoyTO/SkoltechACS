import time
import numpy as np
import matplotlib.pyplot as plt

def imagesc(*args):
    N=len(args)/2
    fig=plt.figure()
    if N>3: 
        lines=int(N/3)+1
    else: lines=1
    columns=0
    if lines>1: columns=3
    else: columns=N
    N = int(N)
    for i in range(0,N):
        image = args[2*i]
        title=args[2*i+1]
        print('imagesc: i='+str(i)+'; title='+title)
        Ni,Nj=image.shape
        p=fig.add_subplot(lines,columns,i+1)
        im=p.imshow(np.transpose(image), origin='lower', aspect=float(Ni)/Nj, interpolation='bicubic', cmap='gray')#, extent=[0,1,0,1])
#         im=p.imshow(args[2*i], origin='lower')#, extent=[0,1,0,1])
        fig.colorbar(im)
        p.set_title(title)

    plt.show()


def image_table(*args):
    lines=len(args)
    columns=max( len(x) for x in args )/2
    print(columns)
    fig=plt.figure()
    number=1
    for line in range(0, lines):
        string=args[line]
        for column in range(0, columns):
            image=string[2*column]
            title=string[2*column+1]
            Ni,Nj=image.shape
            aspect=Ni/Nj
            p=fig.add_subplot(lines, columns, number)
            im=p.imshow(np.transpose(image), origin='lower', interpolation='bilinear', aspect=aspect)
            fig.colorbar(im)
            p.set_title(title)
            number=number+1
    plt.show()


    
    

def draw_graphs(*args):
    N=len(args)
    fig=plt.figure()
    if N>3: 
        lines=int(N/3)+1
    else: lines=1
    columns=0
    if lines>1: columns=3
    else: columns=N
    for i in range(0,N):
        curves=args[i]
        C=(len(curves)-1)/2
        p=fig.add_subplot(lines, columns, i+1)
        p.set_title(curves[0])
        for c in range(0,int(C)):
            p.plot(curves[2*c+1], label=curves[2*c+2])
        p.legend()
    plt.show()

def show_3d_array( u, step, axis=2):
    print('showing the 3d array')
    N = u.shape[axis]
    if axis==2: 
        for i in range(0, int(N/step)):
            plt.imshow(np.transpose(u[:,:,i*step]), origin='lower', interpolation='none')
            plt.colorbar()
            plt.title(str(step*i))
            plt.show()
    if axis==1:
        for i in range(0, int(N/step)):
            plt.imshow(np.transpose(u[:,i*step, :]), origin='lower', interpolation='none')
            plt.colorbar()
            plt.title(str(step*i))
            plt.show()
    if axis==0:
        for i in range(0, int(N/step)):
            plt.imshow(np.transpose(u[i*step, :, :]), origin='lower', interpolation='none')
            plt.colorbar()
            plt.title(str(step*i))
            plt.show()

    






# N=1000
# x=np.linspace(0,1,N)
# y1=np.exp(x)
# y2=x**2
# y3=np.sin(2*np.pi*x)
# y4=x
# 
# draw_graphs(('test1',y1,'y1'),('test2',y2,'y2'), ('test3',y3,'y3',y4,'y4'))
# 















# ########### PYQTGRAPH WRAPS ##################
# 
# def draw_graphs_pyqt(window_title, *args):
#     cols=('b','g','r','c','m','y','k','w')
#     N=len(args)
#     app = QtGui.QApplication([])
#     pg.setConfigOption('background', 'w')
#     pg.setConfigOption('foreground', 'k')
#     win1 = QtGui.QMainWindow()
#     win1.resize(600, 800)
#     win1.setWindowTitle(window_title)
#     plots=pg.GraphicsWindow(title=window_title)
#     win1.setCentralWidget(plots)
#     win1.show()
#     for i in range(0, N):
#         p=plots.addPlot(title=args[i][0])
#         p.addLegend()
#         curve_num=(len(args[i])-1)/2
#         for c in range(0,curve_num):
#             p.plot(args[i][2*c+1], pen=pg.mkPen(cols[c], width=2), name=args[i][2*c+2])
#         plots.nextRow()
# 
#     if __name__ == '__main__':
#         import sys
#         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#             QtGui.QApplication.instance().exec_()
# 
# from pyqtgraph.Qt import QtGui, QtCore
# import pyqtgraph as pg
# from pyqtgraph.ptime import time
# import time
# def show_u(u, step=5, delay=0):
#     Nk=u.shape[2]
#     app = QtGui.QApplication([])
#     win = QtGui.QMainWindow()
#     win.resize(800,800)
#     imv = pg.ImageView()
#     win.setCentralWidget(imv)
#     win.show()
#     win.setWindowTitle('The solution of the straight problem')
#     for k in range(0, int(Nk/step)):
#         usl=u[:,:,int(k*step)]
#         imv.setImage(usl)
#         if delay != 0:
#             time.sleep(delay)
#         app.processEvents()
# 
#     if __name__ == '__main__':
#         import sys
#         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#             QtGui.QApplication.instance().exec_()
# 
