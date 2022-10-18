'''
Author: Peng Bo
Date: 2022-10-18 15:44:37
LastEditTime: 2022-10-18 15:48:32
Description: Queue implemented by list

'''
# coding: utf8

class Queue_bylist():
    def __init__(self):
        self.entries = []
        self.length = 0
        self.front = 0
    def enqueue(self, item):
        self.entries.append(item)
        self.length = self.length + 1 

    def dequeue(self):
        self.length = self.length - 1
        dequeued = self.entries[self.front]
        self.front += 1
        self.entries = self.entries[self.front:]
        return dequeued

    def peek(self):
        return self.entries[0]

    def print(self):
        print(self.entries)