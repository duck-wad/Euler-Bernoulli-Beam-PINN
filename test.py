import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation

''' LOAD DATA '''

mat = scipy.io.loadmat('./data/Burgers.mat')

# 100 time points from 0 to 1
# 256 x points in range [-1, 1]
# for each time point, there is u solution at each x
t = np.array(mat['t']).flatten()
x = np.array(mat['x']).flatten()
u = np.array(mat['usol']).T

''' PLOT U EVOLUTION OVER TIME '''

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(x[0], x[-1])
ax.set_ylim(np.min(u), np.max(u))
ax.set_xlabel('x')
ax.set_ylabel('u')
title = ax.set_title('')
plt.grid(axis='x', linestyle='--')
plt.grid(axis='y', linestyle='--')

def init():
    line.set_data([], [])
    return line, title

def update(frame):
    line.set_data(x, u[frame])
    title.set_text(f'Time = {t[frame]}')
    return line, title

animation = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=100)

#plt.show()
