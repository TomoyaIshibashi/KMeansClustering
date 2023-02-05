# スムージングされたF0データを標準化するプログラム 
# Input : .ffsファイルが入ったFolder
# Output : .ffsファイルを標準化した.sfsファイル

import os
import glob
import pandas as pd
import shutil
import pathlib
import scipy.stats

DirectorySession_F0dataFolder = '/Users/6ashi/documents/MATLAB/Clustering/Clustering_F02/Clustering_N_A/G1/cluster5'
os.chdir(DirectorySession_F0dataFolder)

# F0データファイルを読み込んで行列化
fileList = sorted(glob.glob('*.ffs'))
cnt = 0
for file in fileList:
  cnt += 1
  #fnCompos = file.split('_')
  # balance_F02_L(F)C_N_0001.ffs
  #fileId = fnCompos[0] + '_' + fnCompos[1] + '_' + fnCompos[2] + '_' + fnCompos[3] + '_' + fnCompos[4]
  f = open(file, 'r')
  lines = f.readlines()
  F0data = [line.rstrip("\n") for line in lines]
  float_F0data = [float(i) for i in F0data]
  f.close

  F0data_std = scipy.stats.zscore(float_F0data)

  os.chdir(DirectorySession_F0dataFolder)
  fileId = pathlib.PurePath(file).stem # ファイル名(拡張子除く)を得る
  file_ = fileId + '.sfs' # 標準化されたffs
  g = open(file_, 'w')
  for i in range(len(F0data_std)):
      g.write(str(F0data_std[i])+'\n')
  g.close()

  """
  os.chdir(DirectorySession_F0dataFolder)
  fileId = pathlib.PurePath(file_).stem # ファイル名(拡張子除く)を得る
  to_name = fileId + '.sfs' # 標準化されたffs 
  shutil.move(file_, to_name) # ファイル名を変更する
  """
