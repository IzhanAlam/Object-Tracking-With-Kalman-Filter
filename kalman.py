# -*- coding: utf-8 -*-

import numpy as np


class kalman_filter:

    #Iniitilize the kalman filter manually
    def __init__(self, state_estimateX, covariance_P,  transition_F, control_B, noise_covariance_Q, measurement_Z, obs_model_H, obs_noise_covariance_R):
        """
         Initialize the filter manually to set up different parameters
        """
        self.state_estimateX = state_estimateX
        self.covariance_P = covariance_P
        self.transition_F = transition_F
        self.control_B = control_B
  
        self.noise_covariance_Q = noise_covariance_Q
        self.measurement_Z = measurement_Z
        self.obs_model_H = obs_model_H
        self.obs_noise_covariance_R = obs_noise_covariance_R
    
    #This function will initalize kalman filter with the given values below
    def __init__(self):
        
        #dt - sampling rate
        dt = 1
        #variance of position measurement errors
        Bx = 0.001
        #variance of velocity measurement errors
        Bv = 0.5

        #This model is for position measurements
        stateMatrix = np.zeros((4, 1), np.float32)
        self.state_estimateX = np.zeros((4, 1))
        self.covariance_P = np.eye(4)
        self.transition_F = np.array([[1, 0, dt, 0],[0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.noise_covariance_Q  = np.eye(4)

        
        self.measurement_Z = np.zeros((2, 1))
        self.obs_model_H = np.array([[1,0,0,0],[0,1,0,0]])
        self.obs_noise_covariance_R = np.array([[1,0],[0,1]]) 
        self.control_B = np.array([dt**2/2, dt**2/2, dt,dt])
 


    def predict(self):
        """
        predicts the future state
     
        gets the predicted state estimate
        """
        #no acceleration
        u = 0
        self.state_estimateX = np.dot(self.transition_F, self.state_estimateX) + np.dot(self.control_B, u)
        self.covariance_P = np.dot(np.dot(self.transition_F, self.covariance_P), self.transition_F.T) + self.noise_covariance_Q
        return self.state_estimateX

    def correct(self, Z):
        """
        corrects the kalman filter using the kalman gain
        returns the estimated state vector
        """
        n = self.transition_F.shape[1]
        y = Z - np.dot(self.obs_model_H, self.state_estimateX)
        kalman_brackets = self.obs_noise_covariance_R + np.dot(self.obs_model_H, np.dot(self.covariance_P, self.obs_model_H.T))
        Kalman_gain = np.dot(np.dot(self.covariance_P, self.obs_model_H.T), np.linalg.inv(kalman_brackets))
        self.state_estimateX = self.state_estimateX + np.dot(Kalman_gain, y)
        identity = np.eye(n)
        self.covariance_P = np.dot(np.dot(identity - np.dot(Kalman_gain, self.obs_model_H), self.covariance_P), (identity - np.dot(Kalman_gain, self.obs_model_H)).T) + np.dot(np.dot(Kalman_gain, self.obs_noise_covariance_R), Kalman_gain.T)

        return self.state_estimateX
    
    
      
