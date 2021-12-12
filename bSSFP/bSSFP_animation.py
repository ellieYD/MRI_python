import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from cv2 import magnitude
from matplotlib import pyplot as plt
import numpy as np

def matmul(A,B):
    result = np.zeros([1,3])
    for i in range(3):
        result[0,i] = A[0]*B[i,0]+A[1]*B[i,1]+A[2]*B[i,2]
    return result

T1_white = 790
T2_white = 92
T1_CSF = 4000
T2_CSF = 2200

TR = 10
alpha = np.pi/2
rotation_p = np.array([
    [1,              0,             0],
    [0,  np.cos(alpha), np.sin(alpha)],
    [0, -np.sin(alpha), np.cos(alpha)]
])
rotation_n = np.array([
    [1,               0,              0],
    [0,  np.cos(-alpha), np.sin(-alpha)],
    [0, -np.sin(-alpha), np.cos(-alpha)]
])

excitation = 1600
white = np.zeros([excitation,3])
csf = np.zeros([excitation,3])
white[0,:] = [0, 0, 1]
white[0,:] = matmul(white[0,:],rotation_p) 
csf[0,:] = [0, 0, 1]
csf[0,:] = matmul(csf[0,:],rotation_p)

for i in range(1, excitation):
    white[i,0] = white[i-1,0]*np.exp(-TR/T2_white)
    white[i,1] = white[i-1,1]*np.exp(-TR/T2_white)
    white[i,2] = 1+(white[i-1,2]-1)*np.exp(-TR/T1_white)
    csf[i,0] = csf[i-1,0]*np.exp(-TR/T2_CSF)
    csf[i,1] = csf[i-1,1]*np.exp(-TR/T2_CSF)
    csf[i,2] = 1+(csf[i-1,2]-1)*np.exp(-TR/T1_CSF)
    
    if np.remainder(i+1,2) == 0:
        white[i,:] = matmul(white[i,:],rotation_n)
        csf[i,:] = matmul(csf[i,:],rotation_n)
    else:
        white[i,:] = matmul(white[i,:],rotation_p)
        csf[i,:] = matmul(csf[i,:],rotation_p)


x_axis = np.arange(0,excitation,1)
magnitude_white = np.sqrt(white[:,0]*white[:,0]+white[:,1]*white[:,1])
magnitude_csf = np.sqrt(csf[:,0]*csf[:,0]+csf[:,1]*csf[:,1])

plt.figure()
plt.subplot(211)
plt.plot(x_axis,magnitude_white)
plt.title("White matter signal")
plt.xlabel("Excitation number")
plt.ylabel("|Mxy|")
plt.subplot(212)
plt.plot(x_axis,magnitude_csf)
plt.title("CSF signal")
plt.xlabel("Excitation number")
plt.ylabel("|Mxy|")
plt.show()




fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

def get_arrow(theta):
    x = 0
    y = 0
    z = 0
    u = white[theta,0]
    v = white[theta,1]
    w = white[theta,2]
    return x,y,z,u,v,w

quiver = ax.quiver(*get_arrow(0))

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')


def update(theta):
    global quiver
    quiver.remove()
    quiver = ax.quiver(*get_arrow(theta))

ani = FuncAnimation(fig, update, frames=np.linspace(0,1600,num =1600).astype(int), interval=500)
plt.show()