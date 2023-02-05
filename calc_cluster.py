# resampleにより長さを揃えたF0データ(.sfs)の分布を取得し平均・分散を計算
# 
# Output : 時間あたりの平均・分散

import os
import numpy as np
import pandas as pd

# 長さを揃えたF0dataのフォルダのパスを取得
DirectorySession = '/Users/6ashi/Documents/MATLAB/EXL/EXL_N_A'

# データファイルを読み込んで行列化
os.chdir(DirectorySession)
df_original = pd.read_excel('F0data_N_1_cluster1.xlsx')

print(df_original)
dataset_original = np.array(df_original)

