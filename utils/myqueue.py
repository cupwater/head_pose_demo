'''
Author: Peng Bo
Date: 2022-10-17 18:47:56
LastEditTime: 2022-11-21 20:29:33
Description: 

'''

import numpy as np
import pdb

class MyQueue:
    def __init__(self, queue_size=12, element_dim=1, pool_window=1):
        self.queue_size  = queue_size
        self.element_dim = element_dim
        self.queue = np.zeros((queue_size, element_dim), dtype=np.float32)
        self.pool_window = pool_window
        self.head = -1
        self.tail = -1

    def get_average(self):
        if self.element_num == self.queue_size:
            return np.mean(self.queue, axis=1)
        else:
            if self.head >= self.tail:
                return np.mean( self.queue[self.tail:(self.head+1)], axis=0 )
            else:
                temp = np.concatenate([ self.queue[self.tail:], self.queue[:self.head] ], axis=0) 
                return np.mean(temp, axis=0)

    def enqueue(self, element):
        if self.head != -1:
            self.queue[(self.head+1)%self.queue_size] = element
            self.head = (self.head+1) % self.queue_size
            if self.head == self.tail:
                self.dequeue()
        else:
            self.head = self.tail = 0
            self.queue[self.head] = element[0]
    
    def dequeue(self):
        if self.head == -1:
            print("error, the queue is empty")
            return None
        else:
            if self.tail == self.head:
                self.head = self.tail = -1
                return self.queue[0]
            self.tail = (self.tail+1) % self.queue_size
    
    def peek(self):
        return self.queue[self.head]
    
    def element_num(self):
        if (self.head + 1) % self.queue_size == self.tail:
            return self.queue_size
        elif self.head == self.tail:
            return 1
        else:
            return self.head - self.tail
    
    def smooth(self):
        smooth_element = []
        for i in range(self.pool_window):
            smooth_element.append(self.queue[(self.head-i+self.queue_size)%self.queue_size].tolist())
        smooth_element = np.mean(np.array(smooth_element).reshape(-1, self.element_dim), axis=0)
        return smooth_element

    def to_feature(self):
        feature = []
        for i in range(self.queue_size):
            feature += self.queue[(self.head-i-1+self.queue_size)%self.queue_size].tolist()

        x_max, x_min = 757.694, -125.634
        y_max, y_min = 753.617, -254.816
        feature = np.array(feature).reshape(-1, 2)
        feature[:, 0] = ((feature[:,0]-x_min) / (x_max-x_min) - 0.5) * 4
        feature[:, 1] = ((feature[:,1]-y_min) / (y_max-y_min) - 0.5) * 4
        feature = np.mean(feature.reshape(-1, self.pool_window, 2), axis=1)
        feature = feature.reshape(-1).tolist()
        return feature

    def to_feature1(self):
        feature = []
        for i in range(self.queue_size):
            feature += self.queue[(self.head-i-1+self.queue_size)%self.queue_size].tolist()

        x_max, x_min = 757.694, -125.634
        y_max, y_min = 753.617, -254.816
        feature = np.array(feature).reshape(-1, 2)
        feature[:, 0] = ((feature[:,0]-x_min) / (x_max-x_min) - 0.5) * 4
        feature[:, 1] = ((feature[:,1]-y_min) / (y_max-y_min) - 0.5) * 4
        feature = np.mean(feature.reshape(-1, self.pool_window, 2), axis=1)
        feature = feature.reshape(-1).tolist()
        return feature