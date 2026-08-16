#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:32:06 2022

@author: qopt
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import glob
import os
import shutil
from matplotlib import gridspec
import math
from scipy.stats import norm,poisson
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import chi2
import matplotlib
import inspect

# directorya = "/mnt/N-drive/SCI-NBI-quantop-data/data/LAQS/2024/20240829/tweezerImages"
# directoryb = "/home/qopt/Desktop/data/20240829/"
#
# files = [file for file in os.listdir(directorya) if os.path.isfile(os.path.join(directorya, file))]
# for file in files:
#     if not os.path.exists(os.path.join(directoryb, file)):
#         shutil.copy(os.path.join(directorya, file), directoryb)
##            
def subtract_img(img, sign = 0 ):
    im = img[0].astype(np.int32)-img[1].astype(np.int32)
    return (-1)**sign * im


plt.close('all')


def twoD_Gaussian(xs, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x = xs[0]
    y = xs[1]
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def counts_to_photons(data,size):
    ph = (data - 200*size**2)*0.1
    return ph

class camera_stats():
    def __init__(self, img_array, size, atom_threshold, binning = 2, threshold = 80, mask = None, name = None, location = None, params = None, info = None):
        self.name = name
        self.img_array = counts_to_photons(img_array.astype(np.int32),binning)
        self.size = size
        self.atom_threshold = atom_threshold
        self.location = location
        self.mean_img = np.mean(self.img_array,0)
        self.params = params
        self.info = info
        self.threshold = threshold
#        plt.figure()
#        plt.imshow(self.mean_img)
        self.mask = mask
        if location is None:
            self.location = self.get_location(self.mean_img)
        if mask is None:
            self.mask = self.get_mask()
        
        self.counts = self.get_result(self.img_array)
        self.countsComb = self.counts.flatten()

    
    def __repr__(self):
        return repr(f'Camera stats-- {self.name} -- {len(self.countsComb)/len(self.counts)} Files')
    
    def get_location(self, img, contourArea = 2):
        """
        Function for locating spots in an image
        contourArea : lower bound for spot sizes detected        
        """
        im = img
        
        im -= np.min(im)
        im = im/np.max(im)*255
        im = im.astype(np.uint8)
        im = cv2.GaussianBlur(im,(3,3),0)
        ret, thresh = cv2.threshold(im,self.threshold, 255, cv2.THRESH_BINARY)
        plt.figure()
        plt.imshow(thresh)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        x = []
        y = []
        for c in contours:
            if cv2.contourArea(c)>contourArea:
                M = cv2.moments(c)
                xloc = int(M['m10']/M['m00'])
                yloc = int(M['m01']/M['m00'])
                if (yloc > 5) & (yloc<(im.shape[1]-5)):
                    x.append(xloc)
                    y.append(yloc)
                plt.plot(xloc, yloc, '.')
        print('Number of spots detected: ' + str(len(x)))
        return np.array([x,y]).T
    
    def get_mask(self):
        """
        Function for producing the image mask. Using the locations found by the 
        get_location function, we step through the array to produce a mask. At 
        each location we fit a 2D Gaussian function to the image and produce a 
        mask. At the moment the 'active' pixels are chosen as the pixels above 
        0.3*A + k, that is the amplitude of the gaussian A and the offset k. 
        """
        img = self.mean_img
        x = np.arange(img.shape[0])
        y = np.arange(img.shape[1])

        X, Y = np.meshgrid(x, y)
        self.popt = []
        mask = np.zeros(img.shape,dtype=bool)
        for i,l in enumerate(self.location):
            ly = int(l[0]-self.size/2)
            lx = int(l[1]-self.size/2)
            
            initial_guess = (3,l[1],l[0],5,5,1,0.1)
            
            xf, yf = np.meshgrid(x[lx:lx+self.size], y[ly:ly+self.size])
            popt, pcov = curve_fit(twoD_Gaussian, (xf, yf), 
                                   img[lx:lx+self.size, ly:ly+self.size].ravel(order='F'),p0=initial_guess)
            
            mask |= twoD_Gaussian((X,Y),*popt).reshape(img.shape, order='F') > popt[0]*0.3 + popt[-1] 
            self.popt.append(popt)

        return np.logical_not(mask)

    def get_result(self,img_array):
        """
        Function for calculating pixel sums. Function steps through the locations
        and calculates the pixel sum within self.size. The image is masked using 
        self.mask.
        """ 
        counts = np.zeros([len(self.location),len(img_array)])
    
        for j,img in enumerate(np.copy(img_array)): 
            img[self.mask] = 0
            for i,b in enumerate(self.location):
                bx = int(b[0]-self.size/2)
                by = int(b[1]-self.size/2)
                
                im = img[by: by + self.size + 1,bx: bx + self.size + 1]
                counts[i,j] = np.sum(im) 

                im = None
        return counts
    
    def plot_ROIs(self):
        """
        Function for plotting the ROIs on the mean img.
        """
        fig,ax = plt.subplots()
        ax.imshow(self.mean_img)
        if self.name:
            fig.suptitle(self.name)
        
        for i,b in enumerate(self.location):    
            # print(b)
            ax.plot(b[0],b[1],'r.')
            ax.text(b[0],b[1],str(i),color = 'r')
            # 
            bx = int(b[0]-self.size/2)
            by = int(b[1]-self.size/2)

            rect = patches.Rectangle((bx, by), self.size, self.size, 
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
    def plot_histogram(self): 
        N = len(self.counts)
        cols = 8
        rows = int(math.ceil(N / cols))
        
        gs = gridspec.GridSpec(rows, cols)
        fig = plt.figure()
        x_max = np.max(self.counts)
        for n in range(N):

            ax = fig.add_subplot(gs[n])
            ax.hist(self.counts[n],bins = 20)
            ax.set_xlim([-5,x_max])
        if self.name:
            fig.suptitle(self.name)



    def fit_histogram(self):
        hist = self.countsComb
        bins = np.arange(np.min(hist),np.max(hist) + 2,1)-0.5
        entries, bin_edges = np.histogram(hist, bins = bins, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        fitbins = np.linspace(np.min(hist),np.max(hist) + 2, 40) -0.5
        entriesF, bin_edgesF = np.histogram(hist, bins = fitbins, density=True)

        peaks, _ = find_peaks(entriesF)
        peak1 = np.max(bin_centers)/5
        peak2 = np.max(bin_centers)*3/4
        parameters, cov_matrix = curve_fit(self.fit_function, bin_centers, entries,p0=[0.55,
                                                                                       np.sqrt(peak1),
                                                                                       peak1,
                                                                                       np.sqrt(peak2),
                                                                                       peak2])
        self.fit_parameters = parameters
        self.cov_matrix = cov_matrix
        return parameters,np.sqrt(np.diag(cov_matrix))
    
    def fit_function(self,k,A, lamb1, lamb2,sc1,sc2):
        '''poisson function, parameter lamb is the fit parameter'''
        return (1-A)*norm.pdf(k, lamb1,sc1) + A*norm.pdf(k, lamb2,sc2)

    def fit_function_p(self,k,A, lamb1, lamb2):
        '''poisson function, parameter lamb is the fit parameter'''
        return (1-A)*poisson.pmf(k, lamb1) + A*poisson.pmf(k, lamb2)
    
    def make_list(self, name, value):
        if not hasattr(self, name):
            setattr(self,  name, value)
        return getattr(self, name)
    
def make_scatterplot(array1, array2,thresh,name=None):
#    thresh = 40
    fig, ax = plt.subplots()
    a = array1<thresh
    b = array2<thresh 
    c = array1>thresh
    d = array2>thresh
    
    background = (a & b)
    lost = (c & b)
    kept = (c & d)  
    #ax.scatter(d350_350_2.counts[0],d350_350_3.counts[0],s = 2)
    ax.scatter(array1[background],array2[background],
               s = 2,color = 'k',label = 'No atom: ' + str(np.sum(background)) )
    ax.scatter(array1[lost],array2[lost],s = 2,color = 'r'
               ,label = 'Lost atom: ' + str(np.sum(lost)))
    ax.scatter(array1[kept],array2[kept],s = 2,color = 'g'
               ,label = 'Kept atom: ' + str(np.sum(kept)))
    #ax.scatter()
    fig.suptitle(name)
    ax.set_aspect('equal')
#    ax.set_xlim([0,210])
#    ax.set_ylim([0,210])
    ax.axvline(thresh)
    ax.axhline(thresh)
    ax.legend()
    
    ax.set_xlabel('1st image')
    ax.set_ylabel('2nd image')
    
def make_scatterplot2(a1, a2,thresh,name=None):
#    thresh = 40
#    fig, ax = plt.subplots()
    N = len(a1)
    cols = 8
    rows = int(math.ceil(N / cols))
    
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()
    for n in range(N):
        array1 = a1[n]
        array2 = a2[n]
        
        ax = fig.add_subplot(gs[n])
#        ax.hist(self.counts[n],bins = 20)
        a = array1<thresh
        b = array2<thresh 
        c = array1>thresh
        d = array2>thresh
        
        background = (a & b)
        lost = (c & b)
        kept = (c & d)  
        #ax.scatter(d350_350_2.counts[0],d350_350_3.counts[0],s = 2)
        
        ax.scatter(array1[background],array2[background],
                   s = 2,color = 'k',label = 'No atom: ' + str(np.sum(background)) )
        ax.scatter(array1[lost],array2[lost],s = 2,color = 'r'
                   ,label = 'Lost atom: ' + str(np.sum(lost)))
        ax.scatter(array1[kept],array2[kept],s = 2,color = 'g'
                   ,label = 'Kept atom: ' + str(np.sum(kept)))
        ax.axvline(thresh)
        ax.axhline(thresh)
        ax.set_xlim(-5,300)
        ax.set_ylim(-5,300)
#        ax.legend()
    #ax.scatter()
        ax.set_aspect('equal')
    fig.tight_layout()
    #    ax.set_xlim([0,210])
    #    ax.set_ylim([0,210])
        
        
#    fig.xlabel('1st image')
#    fig.ylabel('2nd image')
    fig.suptitle(name)
    fig.tight_layout()


def routine(stats,save = False):
    stats.plot_ROIs()
    stats.plot_histogram()
    
def str_from_dict(dictionary):
    textstr = ''
    for i in dictionary.keys():
        textstr+= i + ' = ' + str(dictionary[i]) + '\n'
    return textstr
#    stats.plot_individual(save = save)v
#%%
class experiment_series():
    def __init__(self, name):
        self.name = name
        self.data = []
        
    def make_list(self, name, value):
        if not hasattr(self, name):
            setattr(self,  name, value)
        return getattr(self, name)
    
    def __repr__(self):
        return repr(f'Experiment series -- {self.name} -- {len(self.data)} points')
        # print
from dataclasses import dataclass

@dataclass
class datatuple:
    """Class for keeping track of an item in inventory."""
    params: dict
    data: list

    def append(self,dat):
        self.data.append(dat)


def load_data(experiment_list, directory = ''):

    for e in experiment_list:
        e['exp_data'] = []
        e['experiment_series'] = experiment_series(e['Name'])
        dats = []
        names = glob.glob(f'{directory}{e["sname"]}_*.npy')
        # print(f'{directory}/{e["sname"]}_*.npy')
        print('Found {:} files for '.format(len(names)) + e['Name'])
        
        paramsl = []
        for i,named in enumerate(names):
            try:
                dobj = np.load(named, allow_pickle = True)[()]
                dat    = dobj['Images']
                params = dobj['globalvariables']
                if params['param4'][0]==9192.3:
                        print(named)
                if params not in paramsl:
                    paramsl.append(params)
                    dats.append(datatuple(params,[dat]))
                    if params['param4'][0]==9192600:
                        print(named)
                else:
                   for d in dats:
                       if params == d.params:
                           d.append(dat)
            except:
                print(named)
        # print(d.params)        
        for d in dats:
            expsl = []
            if not e['mask']:
                refobj = camera_stats(np.array(d.data)[:,1], 
                                      e['size'], 
                                      e['Atom threshold'],
                                      name=e['Name'],
                                      mask = e['mask'], 
                                      location = e['loc'], 
                                      threshold = e['Threshold'])
                expsl.append(refobj)
                loc  = refobj.location
                loc = loc[np.lexsort((np.round((loc[:,0]-np.min(loc[:,0]))/e['size'])*e['size'],np.round((loc[:,1]-np.min(loc[:,1]))/e['size'])*e['size']))]
                mask = refobj.mask
                e['exp_data'] = expsl
                continue
        
            for i in range(e['Num imgs']):
                expsl.append(camera_stats(np.array(d.data)[:,i], 
                                          e['size'], 
                                          e['Atom threshold'], 
                                          name=e['Name'] + ' ' + str(i), 
                                          mask = mask, 
                                          location = loc, 
                                          params = d.params, 
                                          info = e['Info']))
            e['experiment_series'].data.append(expsl)#camera_stats(np.array(d.data)[:,i], e['size'], e['Atom threshold'], name=e['Name'] + ' ' + str(i), mask = mask, location = loc, params = d.params, info = e['Info']))
            e['exp_data'].append(expsl)
            
    for j,ser in enumerate(experiment_list):
        # print(ser['Name'])
        if not ser['Name']== 'Reference':
            for _,exps in enumerate(ser['exp_data']):
            
                for i, exp in enumerate(exps):
#                     print(exp)
#                     print(i)
#                    routine(exp)
                    if i == 1:
                        loading_array             = np.array([np.sum(e>exp.atom_threshold)/len(e) for e in exp.counts])
                        loading                   = np.sum(exp.countsComb>exp.atom_threshold)/len(exp.countsComb)
                        loading_array_individual  = (exp.counts>exp.atom_threshold)*1
        
                    if i == 2:
                        survival_array            = np.array([np.sum(e>exp.atom_threshold)/len(e) for e in exp.counts])/loading_array
                        survival                  = np.sum(exp.countsComb>exp.atom_threshold)/len(exp.countsComb)/loading
                        survival_array_individual = (exp.counts>exp.atom_threshold)*1
                        difference_array          = np.sum(((survival_array_individual-loading_array_individual)>0)*(survival_array_individual-loading_array_individual), axis = 1)
                        
                        ser['experiment_series'].make_list('loading_array_individual', []).append(loading_array_individual)
                        ser['experiment_series'].make_list('survival_array_individual', []).append(survival_array_individual)
                        ser['experiment_series'].make_list('difference_array', []).append(difference_array)
        
                        ser['experiment_series'].make_list('loading', []).append(loading)
                        ser['experiment_series'].make_list('survival', []).append(survival)
                        ser['experiment_series'].make_list('error_survival', []).append(np.sqrt(survival*(1-survival)/(loading*len(exp.countsComb))))
                        ser['experiment_series'].make_list('error_loading', []).append(np.sqrt(loading*(1-loading)/len(exp.countsComb)))
                        
                        ser['experiment_series'].make_list('loading_array', []).append(loading_array)
                        ser['experiment_series'].make_list('survival_array', []).append(survival_array)
                        ser['experiment_series'].make_list('error_loading_array', []).append(np.array([np.sqrt(f*(1-f)/(len(exp.counts[i]))) for i, f in enumerate(loading_array)]))                
                        ser['experiment_series'].make_list('error_survival_array', []).append(np.array([np.sqrt(p*(1-p)/(loading_array[i]*len(exp.counts[i]))) for i, p in enumerate(survival_array)]))
                         
                        ser['experiment_series'].make_list('parameters', []).append(exp.params)
                        ser['experiment_series'].make_list('mean_imgs', []).append(exp.mean_img)
                        ser['experiment_series'].make_list('countsComb',[]).append(exp.countsComb)
                        for p in exp.params.keys():
                            ser['experiment_series'].make_list(p,[]).append(exp.params[p][0])
                        # ser['experiment_series'<].make_list('loading_array_individual', []).append(loading_array_individual)
                        # print(exp.params)
                    
#%%
""" Here are some useful plot functions"""

def fit_function_single_site(k,A, lamb1, lamb2,sc1,sc2):
    '''poisson function, parameter lamb is the fit parameter'''
    return (1-A)*norm.pdf(k, lamb1,sc1) + A*norm.pdf(k, lamb2,sc2)
    
def fit_histogram_single_site(hist):
    b_tot = int(np.sqrt(len(hist)))
    bins = np.linspace(np.min(hist),np.max(hist),b_tot)-0.5
    entries, bin_edges = np.histogram(hist, bins = bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    fitbins = np.linspace(np.min(hist),np.max(hist) + 2, 40) -0.5
    entriesF, bin_edgesF = np.histogram(hist, bins = fitbins, density=True)

    peaks, _ = find_peaks(entriesF)
    peak1 = np.max(bin_centers)/5
    peak2 = np.max(bin_centers)*3/4
    parameters, cov_matrix = curve_fit(fit_function_single_site, bin_centers, entries,p0=[0.55,
                                                                                       peak1,
                                                                                       peak2,
                                                                                       np.sqrt(peak1),
                                                                                       np.sqrt(peak2),
                                                                                      ])
    return parameters,np.sqrt(np.diag(cov_matrix))

def plot_histogram_and_fit_single_site(experiment_list:list, EXPN:int, param:str):
    """
    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    EXPN : int
        DESCRIPTION.
    param : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for exp in experiment_list:

        exp_data_list = exp['experiment_series'].data
        rows = 8
        cols = 8
        sigma_vec = np.zeros((len(exp_data_list), cols*rows))
        mu_vec = np.zeros((len(exp_data_list), cols*rows))
        for iexp_data, exp_data_ in enumerate(exp_data_list):
            # print(exp_data)
            exp_data = exp_data_[EXPN]
            gs = gridspec.GridSpec(rows, cols)
            fig = plt.figure()
            fig.suptitle(exp_data.name + '-' + str(exp_data.params[param][0]) +'  '+ str(len(exp_data.countsComb)/64))
            fig.text(0.01,0.01, 'func name: ' + inspect.stack()[0][3], c = 'grey', fontsize='small')

            hist_vec = exp_data.counts
            for n in range(rows*cols):
                hist = hist_vec[n]
                b_tot = int(np.sqrt(len(hist)))#round((np.max(hist)-np.min(hist))/np.sqrt(len(hist)))
                #    print(np.max(hist)-np.min(hist))
                bins = np.linspace(np.min(hist),np.max(hist),b_tot)-0.5
                # print(b_tot)
                entries, bin_edges = np.histogram(hist, bins=bins, density=True)
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                
                parameters, sdparametrs = fit_histogram_single_site(hist)
                mu_vec[iexp_data, n] = parameters[2]
                sigma_vec[iexp_data, n] = parameters[4]
                ax = fig.add_subplot(gs[n])
                ax.hist(hist, bins=bins, density=True)
                ax.plot(bin_centers, fit_function_single_site(bin_centers, *parameters), label='µ = {:.0f} $\\sigma²$ = {:.0f}'.format(parameters[2],parameters[4]**2))
                ax.set_xlim(-14, 200)
                ax.legend()
                
        fig_1, ax_1 = plt.subplots(1,3,figsize = (10,4))
        for iexp_data, exp_data_ in enumerate(exp_data_list):
            fig_1.suptitle('Frequency scan of Piezo')
            ax_1[0].hist(mu_vec[iexp_data], bins = 20, alpha = 0.7, label = f'{100+400*iexp_data}')
            ax_1[1].hist(sigma_vec[iexp_data]**2, bins = 20, alpha = 0.7, label = f'{100+400*iexp_data}')
            ax_1[2].hist(sigma_vec[iexp_data]**2/mu_vec[iexp_data], bins = 20, alpha = 0.7, label = f'{100+400*iexp_data}')
        ax_1[0].set_title('μ distribution')
        ax_1[1].set_title('$\\sigma²$ distribution')
        ax_1[2].set_title('$\\sigma²/µ$ distribution')
        ax_1[0].legend()
        ax_1[1].legend()
        ax_1[2].legend()
        
            
def plot_histogram_and_fit(experiment_list:list, EXPN:int, param:str):
    """
    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    EXPN : int
        DESCRIPTION.
    param : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for exp in experiment_list:

        exp_data_list = exp['experiment_series'].data
        for exp_data_ in exp_data_list:
            # print(exp_data)
            exp_data = exp_data_[EXPN]
            fig, ax = plt.subplots()
            fig.suptitle(exp_data.name + '-' + str(exp_data.params[param][0]) +'  '+ str(len(exp_data.countsComb)/64))
            fig.text(0.01,0.01, 'func name: ' + inspect.stack()[0][3], c = 'grey', fontsize='small')

            hist = exp_data.countsComb
            b_tot = int(np.sqrt(len(hist)))#round((np.max(hist)-np.min(hist))/np.sqrt(len(hist)))
        #    print(np.max(hist)-np.min(hist))
            bins = np.linspace(np.min(hist),np.max(hist),b_tot)-0.5
            # print(b_tot)
            entries, bin_edges, _ = plt.hist(hist, bins=bins, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
            exp_data.fit_histogram()
            parameters = exp_data.fit_parameters
            ax.plot(bin_centers,exp_data.fit_function(bin_centers, *parameters), label='µ = {:.0f} $\\sigma$ = {:.0f}'.format(parameters[2],parameters[4]))
            ax.set_xlim(-14, 150)
            ax.legend()
            
def plot_scatterplot(experiment_list:list, param:str):
    """
    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    EXPN : int
        DESCRIPTION.
    param : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for exp in experiment_list:

        exp_data_list = exp['experiment_series'].data
        for exp_data_ in exp_data_list:
            # print(exp_data)
            exp_data_load = exp_data_[1]
            exp_data_surv = exp_data_[2]
            fig, ax = plt.subplots()
            fig.suptitle(exp_data_load.name + '-' + str(exp_data_load.params[param][0]) +'  '+ str(len(exp_data_load.countsComb)/64))
            fig.text(0.01,0.01, 'func name: ' + inspect.stack()[0][3], c = 'grey', fontsize='small')

            hist_load = exp_data_load.countsComb
            hist_surv = exp_data_surv.countsComb
#            b_tot = int(np.sqrt(len(hist)))#round((np.max(hist)-np.min(hist))/np.sqrt(len(hist)))
        #    print(np.max(hist)-np.min(hist))
#            bins = np.linspace(np.min(hist),np.max(hist),b_tot)-0.5
            # print(b_tot)
#            entries, bin_edges,_ = plt.hist(hist, bins=bins, density=True)
#            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
#            exp_data.fit_histogram()
#            parameters = exp_data.fit_parameters
            threshold_load = exp_data_surv.atom_threshold
            threshold_surv = exp_data_surv.atom_threshold
            a = hist_load<threshold_load
            b = hist_surv<threshold_surv
            c = hist_load>threshold_load
            d = hist_surv>threshold_surv
            
            background = (a & b)
            lost = (c & b)
            kept = (c & d)
            ghost = (a & d)
            size = 10
            ax.scatter(hist_load[background],hist_surv[background], alpha = 0.5,s = size, c = 'k', label = f'Background: {np.sum(background)}')
            ax.scatter(hist_load[lost],hist_surv[lost], alpha = 0.5,s = size, c = 'r', label = f'Lost: {np.sum(lost)}')
            ax.scatter(hist_load[kept],hist_surv[kept], alpha = 0.5,s = size, c = 'g', label = f'Kept: {np.sum(kept)}')
            ax.scatter(hist_load[ghost],hist_surv[ghost], alpha = 0.5,s = size, c = 'm', label = f'Ghost: {np.sum(ghost)}')

            ax.axvline(threshold_load, ls = 'dashed', c= 'k')
            ax.axhline(threshold_surv, ls = 'dashed', c= 'k')
            ax.axis('square')
            ax.legend()



def plot_stacked_histograms(experiment_list:list, EXPN:int, param:str, save = False):
    """
    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    EXPN : int
        DESCRIPTION.
    param : str
        DESCRIPTION.
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    for exp in experiment_list:
        exp_data = exp['experiment_series']
        fig, ax = plt.subplots()
#        fig.suptitle(f'{exp["experiment_series"].name} -- img #{EXPN}')
        fig.text(0.01,0.01, 'func name: ' + inspect.stack()[0][3], c = 'grey', fontsize='small')

        props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    
        ax.text(0., -0.1, str_from_dict(exp['Info']), transform=ax.transAxes, fontsize=8,
                ha='left',va = 'top', bbox=props)
        ax.set_xlabel('Photon counts')
        ax.set_ylabel('Frequency')
        
        delta_Y = 0.02
        exp_data_list = exp_data.data
        # np_ex = np.array(exp_data_list)
        x_array        = [e[param][0]   for e in exp_data.parameters]
        # parslist       = [exp['experiment_series'].parameters[param][0] for es in np_ex[:,1]]
        pars, expList= zip(*sorted(zip(x_array, exp_data_list)))
        
        for i,e in enumerate(expList):
            
            hist = e[EXPN].countsComb
            b_tot = 15
            bins = np.arange(np.min(hist),np.max(hist),b_tot)-0.5
            entries, bin_edges = np.histogram(hist, bins=bins, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
            e[EXPN].fit_histogram()
            parameters = e[EXPN].fit_parameters
            Y = entries
            baseline = min(Y)
        
            y_change = delta_Y * i - baseline
            Y = Y + y_change
            
            ax.fill_between(bin_centers, Y,np.ones(len(bin_centers)) * delta_Y * i ,alpha=0.7)        
            
            binlin = np.linspace(bin_centers[0],bin_centers[-1],1000)
            ax.plot(binlin,e[EXPN].fit_function(binlin, *parameters) +y_change)
            
            ax.axhline(y_change, ls = 'dashed')
            ax.text(parameters[2]+parameters[4], y_change + 0.005, str(e[EXPN].params[param][0]))
        fig.tight_layout()
        
def plot_image_series(experiment_list:list, EXPN:int, param:str, save = False):
    """
    

    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    EXPN : int
        DESCRIPTION.
    param : str
        DESCRIPTION.
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    for exp in experiment_list:
        exp_data = exp['experiment_series']
        
        exp_data_list = exp_data.data
        # np_ex = np.array(exp_data_list)
        x_array        = [e[param][0]   for e in exp_data.parameters]
        # parslist       = [exp['experiment_series'].parameters[param][0] for es in np_ex[:,1]]
        pars, expList= zip(*sorted(zip(x_array, exp_data_list)))
        ncols = len(x_array)
        fig, ax = plt.subplots(ncols = ncols, sharex=True, sharey=True)
        fig.suptitle(f'{exp["experiment_series"].name} -- img #{EXPN}')
        fig.text(0.01,0.01, 'func name: ' + inspect.stack()[0][3], c = 'grey', fontsize='small')


        # fig, ax = plt.subplots(ncols = ncols)
        if ncols>1:
            axf = ax.flatten()
        else:
            axf = [ax]
        for i,e in enumerate(expList):

            axf[i].imshow(e[EXPN].mean_img)
            axf[i].set_title(e[EXPN].params[param][0])
        # if save:
            # fig.savefig(exlist[0][EXPN].name)

def tuple_to_np(tpl):
    outp= []
    for t in tpl:
        outp.append(np.array(t))
    return outp

def lorentzian(x, A, b, x0, gamma):
    """
    Returns b - Lorentzian function values at x.

    Parameters:
    - x: array-like, input values
    - b: baseline shift
    - x0: center of the peak
    - gamma: half width at half max (HWHM)
    """
    return b - A*(gamma / (np.pi * ((x - x0)**2 + gamma**2)))


def gaussian(x, A, b, mu, sigma):
    """
    Returns b - Gaussian function values at x.

    Parameters:
    - x: array-like, input values
    - b: baseline shift
    - mu: mean of the Gaussian
    - sigma: standard deviation (width)
    """
    return b - A*(1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def chi2_pvalue(y_data, y_fit, y_err, num_params):
    residuals = y_data - y_fit
    chi2_stat = np.sum((residuals / y_data)**2)
    dof = len(y_data) - num_params
    p_value = 1 - chi2.cdf(chi2_stat, dof)
    return p_value

def plot_survival_stand(experiment_list:list, param:str, xlabel = None, ylabel = None):    
    """
    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    param : str
        DESCRIPTION.
    xlabel : TYPE, optional
        DESCRIPTION. The default is None.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    fig.text(0.01,0.01, 'func name: ' + inspect.stack()[0][3], c = 'grey', fontsize='small')

    for i, exp in enumerate(experiment_list):
        exp_data = exp['experiment_series']
        


        x_array        = [e[param][0]   for e in exp_data.parameters]
        
        x_f_np, y_f_np, z_f_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.loading, exp_data.error_loading))))
        x_s_np, y_s_np, z_s_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.survival,exp_data.error_survival))))
        
    
        ax.errorbar(x_s_np, y_s_np, z_s_np, label=exp['Name'] + '-Survival rate', fmt='-', marker = '.', capsize=3)
        
        ax.errorbar(x_f_np, y_f_np, z_f_np, label=exp['Name'] + '-Loading rate',  fmt='-', marker = '.', capsize=3)
        

    if xlabel:
        ax.set_xlabel(xlabel, fontsize = 18)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize = 18)
    fig.legend()

def plot_survival(experiment_list:list, param:str, xlabel = None, ylabel = None):    
    """
    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    param : str
        DESCRIPTION.
    xlabel : TYPE, optional
        DESCRIPTION. The default is None.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    fig.text(0.01,0.01, 'func name: ' + inspect.stack()[0][3], c = 'grey', fontsize='small')

    for i, exp in enumerate(experiment_list):
        exp_data = exp['experiment_series']
        


        x_array        = [e[param][0]   for e in exp_data.parameters]
        
        x_f_np, y_f_np, z_f_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.loading, exp_data.error_loading))))
        x_s_np, y_s_np, z_s_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.survival,exp_data.error_survival))))

        ax.errorbar((x_f_np - x_f_np[0])/10 + 125, y_f_np, z_f_np, label=exp['Name'] + '-Loading rate',  fmt='-', marker = '.', capsize=3)
        ax.errorbar((x_s_np - x_s_np[0])/10 + 125, y_s_np, z_s_np, label=exp['Name'] + '-Survival rate',  fmt='-', marker = '.', capsize=3)
        print(x_s_np)
#    np.save('temperature_8x8_20250829', (x_s_np, y_s_np, z_s_np))
    if xlabel:
        ax.set_xlabel(xlabel, fontsize = 10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize = 10)
    fig.legend()
            
            


def plot_survival_individual(experiment_list:list, param:str, shape:tuple, xlabel = None, ylabel = None):
    """
    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    param : str
        DESCRIPTION.
    shape : tuple
        DESCRIPTION.
    xlabel : TYPE, optional
        DESCRIPTION. The default is None.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(ncols = shape[1], nrows = shape[0], sharex=True, sharey=True)
    fig.text(0.01,0.01, 'func name: ' + inspect.stack()[0][3], c = 'grey', fontsize='small')
    axf = ax.flatten()
    for exp in experiment_list:
        exp_data = exp['experiment_series']
        
        x_array        = [e[param][0]   for e in exp_data.parameters]
        x_f_np, y_f_np, z_f_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.loading_array, exp_data.error_loading_array))))
        x_s_np, y_s_np, z_s_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.survival_array,exp_data.error_survival_array))))

        ys = y_s_np.T
        zs = z_s_np.T
        for i,y in enumerate(ys):
            axf[i].errorbar(x_s_np,y,zs[i],label=exp['Name'] + '-Survival rate', fmt='.', capsize = 3)
            axf[i].set_ylim([0,1])

def plot_survival_individual2(experiment_list:list, param:str, shape:tuple, xlabel = None, ylabel = None):
    """
    

    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    param : str
        DESCRIPTION.
    shape : tuple
        DESCRIPTION.
    xlabel : TYPE, optional
        DESCRIPTION. The default is None.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    xlist : TYPE
        DESCRIPTION.
    ylist : TYPE
        DESCRIPTION.

    """    
    xlist = []
    ylist = []
    for EXPN, exp in enumerate(experiment_list):
        exp_data = exp['experiment_series']
        x_array        = [e[param][0]   for e in exp_data.parameters]
        
        x_f_np, y_f_np, z_f_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.loading_array, exp_data.error_loading_array))))
        x_s_np, y_s_np, z_s_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.survival_array,exp_data.error_survival_array))))
        
        ncols = len(x_s_np)
        fig, ax = plt.subplots(ncols = ncols, sharex=True, sharey=True)
        fig.suptitle(f'{exp["experiment_series"].name} -- img #{EXPN}')
        fig.text(0.01,0.01, 'func name: ' + inspect.stack()[0][3], c = 'grey', fontsize='small')

        if ncols>1:
            axf = ax.flatten()
        else:
            axf = [ax]
        ys = y_s_np.reshape((len(x_s_np),shape[0],shape[1]))
        for i,xs in enumerate(x_s_np):
            axf[i].imshow(ys[i], vmin=0, vmax=1)
        xlist.append(x_s_np)
        
        ylist.append(ys)
    return xlist, ylist

def plot_loading_individual2(experiment_list:list, param:str, shape:tuple, xlabel = None, ylabel = None):
    """
    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    param : str
        DESCRIPTION.
    shape : tuple
        DESCRIPTION.
    xlabel : TYPE, optional
        DESCRIPTION. The default is None.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    xlist : TYPE
        DESCRIPTION.
    ylist : TYPE
        DESCRIPTION.
    """    
    xlist = []
    ylist = []
    for EXPN, exp in enumerate(experiment_list):
        exp_data = exp['experiment_series']
        x_array        = [e[param][0]   for e in exp_data.parameters]
        
        x_f_np, y_f_np, z_f_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.loading_array, exp_data.error_loading_array))))
        x_s_np, y_s_np, z_s_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.survival_array,exp_data.error_survival_array))))
        
        ncols = len(x_s_np)
        fig, ax = plt.subplots(ncols = ncols, sharex=True, sharey=True)
        fig.suptitle(f'{exp["experiment_series"].name} -- img #{EXPN}')
        fig.text(0.01,0.01, 'func name: ' + inspect.stack()[0][3], c = 'grey', fontsize='small')

        if ncols>1:
            axf = ax.flatten()
        else:
            axf = [ax]
        ys = y_f_np.reshape((len(x_f_np),shape[0],shape[1]))
        for i,xs in enumerate(x_f_np):
            axf[i].imshow(ys[i])#, vmin=0, vmax=1)
        xlist.append(x_f_np)
        
        ylist.append(ys)
    return xlist, ylist

def plot_survival_individual_rows(experiment_list:list, param:str, shape:tuple, xlabel = None, ylabel = None): 
    """
    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    param : str
        DESCRIPTION.
    shape : tuple
        DESCRIPTION.
    xlabel : TYPE, optional
        DESCRIPTION. The default is None.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(ncols = 1, nrows = shape[0], sharex=True, sharey=True)
    fig.text(0.01,0.01, 'func name: ' + inspect.stack()[0][3], c = 'grey', fontsize='small')
    axf = ax.flatten()
    for exp in experiment_list:
        exp_data = exp['experiment_series']
        x_array        = [e[param][0]   for e in exp_data.parameters]

        x_f_np, y_f_np, z_f_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.loading_array, exp_data.error_loading_array))))
        x_s_np, y_s_np, z_s_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.survival_array,exp_data.error_survival_array))))
              
        ys = y_s_np.T

        ys_r = ys.reshape([shape[0],shape[1],ys.shape[-1]])
        ys_collapsed = np.array([np.mean(y, axis = 0) for y in ys_r])
        
        for i,y in enumerate(ys_collapsed):

            axf[i].plot(x_s_np,ys_collapsed[i],label=exp['Name'] + '-Survival rate')
            axf[i].set_ylim([0,1])
        


def extract_data(experiment_list:list, param:str):    
    """
    Parameters
    ----------
    experiment_list : list
        DESCRIPTION.
    param : str
        DESCRIPTION.
    xlabel : TYPE, optional
        DESCRIPTION. The default is None.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    loading = []
    loading_err = []
    times = []
    for i, exp in enumerate(experiment_list):
        exp_data = exp['experiment_series']
        


        x_array        = [e[param][0]   for e in exp_data.parameters]
        
        x_f_np, y_f_np, z_f_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.loading, exp_data.error_loading))))
        x_s_np, y_s_np, z_s_np = tuple_to_np(zip(*sorted(zip(x_array, exp_data.survival,exp_data.error_survival))))


        
        loading.append(y_f_np)
        loading_err.append(z_f_np)
        times.append(x_f_np)
        
    return times[0], loading[0], loading_err[0]
#%%

directorya = "/mnt/N-drive/SCI-NBI-quantop-data/data/LAQS/2026/20260623/tweezerImages"
directoryb = "/home/qopt/Desktop/data/20260623/"
#
files = [file for file in os.listdir(directorya) if os.path.isfile(os.path.join(directorya, file))]
for file in files:
    if not os.path.exists(os.path.join(directoryb, file)):
        shutil.copy(os.path.join(directorya, file), directoryb)


#%%
el = []
size = 8
#el.append({'sname' : 'tweezerLoad8x8'    ,'mask': None,'loc': None, 'Num imgs' : 3, 'Name': 'Reference', 'Threshold': 50,'size':10, 'Atom threshold':30})
#
#el.append({'sname' : 'tweezerLoad8x8'    ,'mask': None,'loc': None, 'Num imgs' : 3, 'Name': 'Reference', 'Threshold': 70,'size':10, 'Atom threshold':50})
#el.append({'sname' : 'tweezerLoad8x8'    ,'mask': True,'loc': True,'Num imgs' : 3,'Name': 'Load MOPA','size':10, 'Atom threshold':30, 'Info':{'COL freq': 'Scanned', 'COL power': 0.1, 'IMG freq': 300, 'IMG power': 0.08, 'Trap Power':275}})
#
#el.append({'sname' : 'tweezerLoad8x8-MOPA'    ,'mask': None,'loc': None, 'Num imgs' : 3, 'Name': 'Reference', 'Threshold': 60,'size':10, 'Atom threshold':50})
#el.append({'sname' : 'tweezerLoad8x8-MOPA'    ,'mask': True,'loc': True,'Num imgs' : 3,'Name': 'Load MOPA-1','size':10, 'Atom threshold':30, 'Info':{'COL freq': 'Scanned', 'COL power': 0.1, 'IMG freq': 300, 'IMG power': 0.08, 'Trap Power':275}})
#
#el.append({'sname' : 'tweezerLoad8x8-TiSaph'    ,'mask': None,'loc': None, 'Num imgs' : 3, 'Name': 'Reference', 'Threshold': 70,'size':10, 'Atom threshold':50})
#el.append({'sname' : 'tweezerLoad8x8-TiSaph'    ,'mask': True,'loc': True,'Num imgs' : 3,'Name': 'Load TiSaph','size':10, 'Atom threshold':30, 'Info':{'COL freq': 'Scanned', 'COL power': 0.1, 'IMG freq': 300, 'IMG power': 0.08, 'Trap Power':275}})
#
#el.append({'sname' : 'tweezerLoad8x8-TiSaph-1'    ,'mask': None,'loc': None, 'Num imgs' : 3, 'Name': 'Reference', 'Threshold': 60,'size':10, 'Atom threshold':50})
#el.append({'sname' : 'tweezerLoad8x8-TiSaph-1'    ,'mask': True,'loc': True,'Num imgs' : 3,'Name': 'Load TiSaph-1','size':10, 'Atom threshold':30, 'Info':{'COL freq': 'Scanned', 'COL power': 0.1, 'IMG freq': 300, 'IMG power': 0.08, 'Trap Power':275}})

el.append({'sname' : 'tweezerLoad8x8-TiSaph-2'    ,'mask': None,'loc': None, 'Num imgs' : 3, 'Name': 'Reference', 'Threshold': 60,'size':10, 'Atom threshold':50})
el.append({'sname' : 'tweezerLoad8x8-TiSaph-2'    ,'mask': True,'loc': True,'Num imgs' : 3,'Name': 'Load TiSaph-2','size':10, 'Atom threshold':30, 'Info':{'COL freq': 'Scanned', 'COL power': 0.1, 'IMG freq': 300, 'IMG power': 0.08, 'Trap Power':275}})

el.append({'sname' : 'tweezerLoad8x8-MOPA-2'    ,'mask': None,'loc': None, 'Num imgs' : 3, 'Name': 'Reference', 'Threshold': 60,'size':10, 'Atom threshold':50})
el.append({'sname' : 'tweezerLoad8x8-MOPA-2'    ,'mask': True,'loc': True,'Num imgs' : 3,'Name': 'Load MOPA-2','size':10, 'Atom threshold':30, 'Info':{'COL freq': 'Scanned', 'COL power': 0.1, 'IMG freq': 300, 'IMG power': 0.08, 'Trap Power':275}})


load_data(el, directoryb)

survival = []
loading = []
error_survival = []
error_loading = []
interval_time = []
#
el_no_reference = [e for e in el if e['Name'] != 'Reference']

#
param = 'param1'
xlabel = 'Frequency [MHz]'
plot_histogram_and_fit(el_no_reference, 1, param)
#plot_histogram_and_fit(el_no_reference, 2, param)

#plot_histogram_and_fit_single_site(el_no_reference, 1, param)

plot_scatterplot(el_no_reference, param)

plot_survival(el_no_reference, param, xlabel = xlabel , ylabel = 'Survival')
#plot_survival_stand(el_no_reference,'param4', xlabel = 'Raman detuning [kHz]', ylabel = 'Loading')

#plot_survival_individual2(el_no_reference,'param4', (8,8) ,'Pulse Duration [µs]', ylabel = 'Survival')
#
#plot_survival_individual2(el_no_reference, 'param4', (8,8) ,xlabel = 'n', ylabel = 'Survival')
#plot_loading_individual2(el_no_reference, 'param4', (8,8) ,xlabel = 'n', ylabel = 'Survival')

#plot_survival_individual(el_no_reference,'param1', (8,8) ,xlabel = 'MW frequency [Hz]', ylabel = 'Survival')
#plot_survival_individual_rows(el_no_reference,'param1', (8,8) ,xlabel = 'MW frequency [Hz]', ylabel = 'Survival')
# plot_survival_individual_rows_no_BKG(el_no_reference,'param1', (8,8) ,xlabel = 'MW frequency [Hz]', ylabel = 'Survival')


#plot_histogram_and_fit(el_no_reference, 1,'param4')
#plot_histogram_and_fit(el_no_reference, 2,'param4')

#plot_stacked_histograms(el_no_reference, 1, 'param1')
plot_image_series(el_no_reference, 1, param)

#el_no_reference[0]["experiment_series"].data[0][1].plot_histogram()
plt.figure()
#plt.imshow(el_no_reference[0]['exp_data'][0][1].mask)
#%%
#times, loading, loading_err = extract_data(el_no_reference,'param4')
#
