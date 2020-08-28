#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:02:43 2020

@author: ToniPanzera
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.table as table
import seaborn as sns

sns.set()
sns.set_style("white")
sns.set_context("paper")
sns.set_style("ticks")

'''A set of functions to plot various line and continuum behaviours of Wolf-Rayet Binary Stars'''

#Function to convert q to percentage
def get_percent_q(q):
    q_percent = q*100
    return q_percent

def get_percent_u(u):
    u_percent = u*100
    return u_percent

#Function to calculate the position angle given q and u values. Returns a degree value
def get_pa(q,u, name='PA'):
    pa = np.rad2deg(0.5*np.arctan2(u,q))
    i = 0
    for angle in pa:
        if angle < 0:
            pa[i] = angle + 180
            i += 1
    return pa

#Function to calculate position angle error
def calc_pa_err(q, u, qerr, uerr, name='PAerr'):
    p = np.sqrt(q**2 + u **2)
    return table.column((1 / p**2) * np.sqrt((q * qerr)**2 + (u * uerr)**2), name=name)

#Function to calculate total polarisation (p) given q and u
def get_p(q,u, name='P'):
    p = np.sqrt(q**2+u**2)
    return p

#Wraps the phase so that it runs from -0.2-1.2 instead of 0-1
def phase_wrap(a):
    return np.concatenate((a-1,a,a+1))

#Wraps data to that it is in line with the phase wrapping
def wrap(c):
    return np.concatenate((c,c,c))

#Function to calculate the BME fit
def BME_func_full(phase, q0, q3, q4):
    return q0 + q3*np.cos(4*np.pi*phase) + q4*np.sin(4*np.pi*phase)

#Function to plot the BME fit
def BME_fit_plot(BME_func_full, phase, fit_data, fit_data_error):
    fit, cov = curve_fit(BME_func_full, phase, fit_data, 
                sigma = np.ones(len(fit_data))*fit_data_error, absolute_sigma = True)
    phase_range = np.linspace(-0.2, 1.2)
    fit_result = BME_func_full(phase_range, *fit)
    return plt.plot(phase_range, fit_result, color='gray', label = 'BME FIT')

#Function to rotate q by the desired position angle
def q_rot(q,u,pa):
    return q*np.cos(np.deg2rad(2*int(pa)))+u*np.sin(np.deg2rad(2*int(pa)))

#Function to rotate u by the desired position angle
def u_rot(q,u,pa):
    return -q*np.sin(np.deg2rad(2*int(pa)))+u*np.cos(np.deg2rad(2*int(pa)))

#Function to calculate angle from a regression line
def get_reg_angle(slope):
    a = np.rad2deg(0.5*(np.pi+np.tan(slope)))
    return a

#Convert wavelength into velocity
def get_rv(lamda, lamda0):
    c = 299792.458
    lamda0 = np.ones(len(lamda))*lamda0
    v = ((lamda-lamda0)/lamda0)*np.ones(len(lamda))*c
    return v
        
    