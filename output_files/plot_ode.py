# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:23:26 2021

@author: chaos2020
"""
import numpy as np
import matplotlib.pyplot as plt




ode_read = np.load('save_Lorenz_0.npz')

u_set = ode_read['u_set']
para_set = ode_read['para_set']

tick_plot_size = 25
label_plot_size = 30

plt.figure(figsize=(10,6))
plt.rc('xtick', labelsize=tick_plot_size) 
plt.rc('ytick', labelsize=tick_plot_size) 
plt.plot(u_set[2,0,:])    
plt.xlabel('steps', fontsize=label_plot_size)
plt.ylabel('x', fontsize=label_plot_size)
plt.show()

