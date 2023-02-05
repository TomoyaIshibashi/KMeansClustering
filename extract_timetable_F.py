import os
#import glob

DirectorySession = '/Users/tomoya/Documents/MATLAB/balance_F02/balance_F02_N_lab' #neutral or anger or happiness
os.chdir(DirectorySession) #ディレクトリの移動
#path = os.getcwd() #ディレクトリの取得
#print(path)

#fileList = glob.glob('*.lab')
#for file in fileList:
  #os.chdir(DirectorySession)
  #fnCompos = file.split('_')
  #fileId = fnCompos[0] + '_' + fnCompos[1] + '_' + fnCompos[2] + '_cut_' + fnCompos[3]
  #f = open(file, 'r') #ファイルオープン
f = open('balance_F02_N_cut_0023.lab', 'r')
#print(data)

num = 0
for row in f:
  if num % 2 == 0:
    split_row = row.rstrip('\n').split(' ')
    tb = split_row[0]
    te = split_row[1]
    phoneme = split_row[2]
    print(tb, te, phoneme)
    print(num)
    #print(tb)
  num += 1


data = f.read()
print(data)

#if num % 2 == 0:
  #idx = data.rfind("silE")
  #print(data[idx-20:])
f.close

  #os.chdir('/Users/tomoya/Documents/MATLAB/balance_F02/balance_F02_N_lab')
  #fout = open('balance_F02_N_cut_0023.lab', 'x')
  #fout.write(data[idx-20:]) #spの行を含めて出力
  #fout.close