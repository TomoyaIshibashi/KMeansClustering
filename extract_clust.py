# クラスタリング後の情報を元にクラスタごとにデータファイルを分ける
# Input : クラスタのファイル名のリスト
# Output : ファイルの振り分け

import os, glob
import shutil

DirectorySession_main = '/Users/6ashi/Documents/Python_Code'
DirectorySession_FC_N = '/Users/6ashi/Documents/MATLAB/balance_F02_FC/balance_F02_FC_N/FFS'
DirectorySession_LC_N = '/Users/6ashi/Documents/MATLAB/balance_F02_LC/balance_F02_LC_N/FFS'
DirectorySession_cluster_N = '/Users/6ashi/Documents/MATLAB/Clustering/Clustering_F02/Clustering_N_A/G1/Cluster5'
os.chdir(DirectorySession_main) # change directory
#path = os.getcwd() # get current working directory
#print(path)

fileList = sorted(glob.glob('*.ffs'))
for file in fileList:
  os.chdir(DirectorySession_main)
  if file == 'cluster5.ffs':
    f = open(file, 'r') # open the file for reading
    filename = f.readlines()
    #print(filename)
    f.close

    for i in range(len(filename)):
      # balance_F02_FC_N_0001.ffs
      fnCompos = filename[i].split('_')
      if fnCompos[2] == 'FC':
        os.chdir(DirectorySession_FC_N)
        shutil.copy(filename[i].strip('\n'), DirectorySession_cluster_N)
      elif fnCompos[2] == 'LC':
        os.chdir(DirectorySession_LC_N)
        shutil.copy(filename[i].strip('\n'), DirectorySession_cluster_N)
