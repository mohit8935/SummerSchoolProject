# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import csv
import matplotlib.pyplot as plt

x = []
y = []

with open('val-loss.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        
        x.append(int(row[1]))
        y.append(float(row[2]))
        
plt.plot(x,y)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.savefig('val_loss.png')
plt.show()

        
        
        