import os
import glob
import math
import numpy as np

NUMBER = 1 # 0 ≤ NUMBER ≤ 1

DirectorySession = '/Users/tomoya/Documents/MATLAB/balance_F02/balance_F02_N_cut_lab/balance_F02_N_cut1' # neutral or anger or happiness
#DirectorySession = '/Users/6ash!/Documents/MATLAB/balance_F02/balance_F02_N_cut'
os.chdir(DirectorySession) # change directory

fileList = sorted(glob.glob('*.lab'))
e = []

for file in fileList:
  os.chdir(DirectorySession)
  fnCompos = file.split('_')
  fileId = fnCompos[0] + '_' + fnCompos[1] + '_' + fnCompos[2] + '_cal' + str(NUMBER) + '.txt' # file name
  f = open(file, 'r') # open the file for reading
  data = f.readlines()
  f.close

  if fnCompos[4] == '0001.lab': # make directory
    os.chdir('/Users/tomoya/Documents/MATLAB/balance_F02/balance_F02_N_ctrl_e')
    FolderName = fnCompos[0] + '_' + fnCompos[1] + '_' + fnCompos[2] + '_cal'+ str(NUMBER)
    os.mkdir(FolderName)
  

  Array = [data[0].split()] # Array : label file
  num = 1
  end_number = len(data)
  #print(end_number)

  while num < end_number:
    Array.append(data[num].split())
    num += 1

  #print(Array[end_number-1][1])
  #print(Array[0][0])
  Total = round(float(Array[end_number-1][1]) - float(Array[0][0]), 7)
  #Total = Array[end_number-1][1] - Array[0][0]
  #print(Total)
  tl = [round(float(Array[0][1]) - float(Array[0][0]), 7)] # tl : duration per phoneme
  #print(tl)

  num = 1
  while num < end_number: # extract tl from Array
    tl.append(round(float(Array[num][1]) - float(Array[num][0]), 7))
    num += 1
  #print(tl)
  tl_changed = [NUMBER*(Total/end_number) + (1-NUMBER)*tl[0]]
  i = 1
  while i < end_number:
    tl_changed.append(NUMBER*(Total/end_number) + (1-NUMBER)*tl[i])
    i += 1
  #print(tl_changed)
  
  t_changed = [float(Array[0][0])] # t_changed : t'
  i = 1
  while i < end_number:
    j = 0
    t_sum = 0
    while j < i:
      t_sum = t_sum + tl_changed[j]
      j += 1
    #print(round(t_sum, 7))
    t_changed.append(t_sum + float(Array[0][0]))
    i += 1
  t_changed = [round(t_changed[ii], 7) for ii in range(len(t_changed))]
  
  
  # ε : 平均二乗誤差
  t = [float(Array[0][0])]
  i = 1
  while i < end_number:
    t.append(float(Array[i][0]))
    i += 1
  t_sub = [(t[0] - t_changed[0]) ** 2]
  i = 1
  while i < end_number:
    t_sub.append((t[i] - t_changed[i]) ** 2)
    i += 1
  #print(t_sub)
  e.append(str(math.sqrt(sum(t_sub)/(end_number-1))) + '\n')
  
#os.chdir('/Users/6ash!/Documents/MATLAB/balance_F02/balance_F02_N_cut')
os.chdir('/Users/tomoya/Documents/MATLAB/balance_F02/balance_F02_N_ctrl_lab')
os.chdir(FolderName)
fout = open(fileId, 'w') # open the file for writing
for out in e:
  fout.write(out)
fout.close