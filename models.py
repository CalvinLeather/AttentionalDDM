import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline

class model_1():
    def __init__(self, item_high, item_low, item_ref, sigma_sq = 0.014/3):
        self.r_high = item_high #value of high item in bundle
        self.r_low = item_low #value of low item in bundle
        self.r_ref = item_ref #value of alternative
        self.duration = 2500 #other DDM constants, see Krajbich and Rangel's work
        self.d=.0002
        self.sigma_sq = sigma_sq
        self.theta = 1
        
    def sim_trial(self): #simulate teh outcome of a single trial, returns the accumulator
        self.att = self.make_att() #generate attentional array with markov chain
        self.eps = np.random.randn(3, self.duration)*self.sigma_sq #create noise for each item
        self.E_high_val = self.d*self.r_high + self.eps[0,:] #add value to noise for each item
        self.E_low_val = self.d*self.r_low + self.eps[1,:]
        self.E_reference = self.d*self.r_ref + self.eps[2,:]
        self.V = 2*(self.att*self.E_high_val + (1-self.att)*self.E_low_val) - self.E_reference #calculate momentary evidence
        self.acc = np.cumsum(self.V) #calculate accumulation of evidence
        try:
            self.acc = self.acc[:np.where(np.abs(self.acc)>1)[0][0]] #find where it crosses the boundary
        except:
            #if it goes past the end of the array without hitting a boundary, restart the stimulation
            #should should check for your paramters that this doesn't occur too much.
            print('bound reached, restarting')
            self.sim_trial()
        return self.acc
    
    def plot_trial(self):
        #Plots V over the course of the trial
        self.sim_trial()
        c = ['b' if x==1 else 'g' for x in self.att] #color plot based on attention
        plt.scatter(range(len(self.acc)), self.acc, c = c , marker='.')
        
    def make_att(self):
        #Markov process to create attention array
        p_switch = .005
        p_stay = .995
        att = np.zeros((1,self.duration))[0]
        att[0] = np.random.choice([0,1])
        for i in range(self.duration-1):
            if np.random.rand()<p_stay:
                att[i+1] = att[i]
            else:
                att[i+1] = att[i]*-1+1
        att[att==1] = self.theta  #make .3/.7 instead of 0/1 a la Krajbic & Rangel
        att[att==0] = 1-self.theta
        return att
    
class model_2(model_1):
    def __init__(self, item1, item2, item3, beta = .5, beta_i = .5, sigma_sq = 0.014):
        # same as model 1, but with additional paramters for softmaxes
        model_1.__init__(self, item1, item2, item3, sigma_sq)
        self.beta_j = beta_j #beta for p_stay
        self.beta_i = beta_i #beta for saliency softmax
        
    def sim_trial(self): #simulate a single trial, returns the accumulator
        #define noise, momentary evidence and attentional arrays
        self.eps = np.random.randn(3, self.duration)*self.sigma_sq
        self.E_high_val = self.d*self.r_high + self.eps[0,:] 
        self.E_low_val = self.d*self.r_low + self.eps[1,:]
        self.E_reference = self.d*self.r_ref + self.eps[2,:]
        self.att = np.zeros((1,self.duration))[0]
        self.att[0] = np.random.choice([0,1]) 
        for i in range(self.duration-1): 
            #calculate average value of the items up to the current time
            self.E_bar_high = np.mean(self.E_high_val[0:i+1]) 
            self.E_bar_low = np.mean(self.E_low_val[0:i+1])
            #calculate saliency of the items
            self.S_high = np.exp(self.E_bar_high/self.beta_i)/(np.exp(self.E_bar_high/self.beta_i)+np.exp(self.E_bar_low/self.beta_i))
            self.S_low =  1-self.S_high
            #calculate transitional probabilities for markov model
            self.p_stay_high = 1/(1+np.exp(-self.beta_j*self.S_high))
            self.p_stay_low = 1/(1+np.exp(-self.beta_j*self.S_low))
            #markov dynamics
            if self.att[i] == 1:
                if np.random.rand()<self.p_stay_high: 
                    self.att[i+1]=self.att[i]
                else:
                    self.att[i+1] = -1*self.att[i]+1
            else:
                if np.random.rand()<self.p_stay_low:
                    self.att[i+1]=self.att[i]
                else:
                    self.att[i+1] = -1*self.att[i]+1
        #calculate total momentary evdience and accumulation 
        self.V = 2*(self.att*self.E_high_val + (1-self.att)*self.E_low_val) - self.E_reference 
        self.acc = np.cumsum(self.V)
        #find the end point, if it doesn't exist, restart the simulation
        try:
            self.acc = self.acc[:np.where(np.abs(self.acc)>1)[0][0]]
        except:
            print('bound reached, restarting')
            self.sim_trial()
        return self.acc  

#peasant model, we won't be using this in the paper probably
class model_p():
    def __init__(self, item1, item2, item3, tau=.5, beta=.5):
        self.item1 = item1
        self.item2 = item2
        self.item3 = item3
        self.tau = tau
        self.beta = beta
        
    def sim_trial(self):
        partition = np.sum(np.exp(np.asarray([self.item1, self.item2])/self.tau))
        self.p_look_high = np.exp(self.item1/self.tau)/partition
        self.p_look_low = np.exp(self.item2/self.tau)/partition
        
        self.V_bundle = 2*(self.p_look_high*self.item1+self.p_look_low*self.item2)
        self.V_reference = self.item3
        
        partition = np.sum(np.exp(np.asarray([self.V_bundle, self.V_reference])/self.beta))
        self.p_choose_bundle = np.exp(self.V_bundle/self.beta)/partition
        self.p_choose_reference = np.exp(self.V_reference/self.beta)/partition
        
        if np.random.rand()<self.p_choose_bundle:
            return 1
        else:
            return -1
