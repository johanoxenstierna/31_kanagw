

import numpy as np
import random
random.seed(7)  # ONLY HERE
np.random.seed(7)
import time # so that one can profile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import imread
from matplotlib.patches import Rectangle
import P

'''Cant share these unfortunately so replace with your own'''
# heatmaps = np.load('./tutorials/heatmap/heatmaps.npy')
heatmaps = np.load('./O0s_info/stns_TZX.npy')
heatmaps = np.load('./O0s_info/TH.npy')
# heatmap = np.flipud(heatmaps[0, :, :])
# heatmap = np.flipud(heatmaps[:, :])
background_pic = imread('./tutorials/heatmap/background_pic.png')

FRAMES_START = 0
FRAMES_STOP = 2000
WRITE = 0
FPS = 40

Writer = animation.writers['ffmpeg']
writer = Writer(fps=FPS, bitrate=3600)

fig, ax_left = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

ax_left.set_title("Pick-frequency heatmap")
ax_left.axis("off")

im_ax = []  # list of drawables

# im_ax.append(ax_left.imshow(background_pic, extent=[0, 200, 0, 100], alpha=0.4, zorder=2))

# cmap = plt.cm.YlOrRd_r
cmap = plt.cm.PuBu_r
heatmap = np.flipud(heatmaps[0, :, :])
im_ax.append(ax_left.imshow(heatmap, extent=[0, P.NUM_X, 0, P.NUM_Z], alpha=0.99, cmap=cmap, zorder=1))

def init():
    return im_ax

def animate(i):

    '''Heatmap'''
    im_ax[-1].remove()  # Without this, the object doesnt get properly removed
    im_ax.pop()
    prints = "i: " + str(i) + " len_mi_ax: " + str(len(im_ax))
    heatmap = np.flipud(heatmaps[i, :, :])
    im_ax.append(ax_left.imshow(heatmap, extent=[0, P.NUM_X, 0, P.NUM_Z],
                                alpha=0.99, cmap=cmap, zorder=1))

    print(prints)

    return im_ax

ani = animation.FuncAnimation(fig, animate, frames=range(FRAMES_START, FRAMES_STOP),
                              blit=True, interval=1, init_func=init,
                              repeat=False)

# if WRITE == 0:
plt.show()
# else:
#     ani.save('./vids/vid_' + str(WRITE) + '.mp4', writer=writer)



