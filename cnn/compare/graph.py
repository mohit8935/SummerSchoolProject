# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import csv
import matplotlib.pyplot as plt
"""
x = []
y = []

with open('val_loss.csv') as csvfile:
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
plt.show()"""
x = []
y = []
z = []
a = []
with open('val_loss.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        
        x.append(int(row[1])+1)
        y.append(float(row[2])*100)
        
with open('val.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        
        z.append(float(row[2])*100)
with open('val_loss3.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        
        a.append(float(row[2])*100)
        
plt.plot(x,y)
plt.plot(x,z)
plt.plot(x,a)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Accuracy Comparison')
plt.grid(True)
plt.legend(['Run1','Run2','Run3'],loc=2)
plt.style.use(['classic'])
plt.savefig('val_loss_comparison.png')
plt.show()
"""
x = []
y = []
with open('train-acc.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        
        x.append(int(row[1]))
        y.append(float(row[2])*100)
        
plt.plot(x,y)
plt.xlabel('Epochs')
plt.ylabel('Training')
plt.title('Training Loss')
plt.legend()
plt.savefig('train_acc.png')
plt.show()
x = []
y = []
with open('train_loss.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        
        x.append(int(row[1]))
        y.append(float(row[2]))
        
plt.plot(x,y)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('train_loss.png')
plt.show()
        
  """      
        