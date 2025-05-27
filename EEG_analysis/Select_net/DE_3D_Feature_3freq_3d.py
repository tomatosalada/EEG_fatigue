import torch
import numpy as np
from scipy.io import loadmat

'''
file name: DE_4D_Feature
input: the 3D feature of all subjects

output: the 4D feature of all subjects
'''

if __name__ == '__main__':
    data = np.load("/workspace-cloud/toshiki.ohno/EEG_fatigue/EEG_analysis/processedData/data_3d_3freq.npy")
    print("data.shape: ", data.shape)  # [325680, 17, 3] 

    
    data_3d = data  # [samples, area, channels]->[325680, 17, 3]
    print("data_3d.shape :", data_3d.shape)

    # from 3D features 
    # [325680, 17, 3] -> [20355, 16, 17, 3]　　　　325680÷16=20355　(サンプル数, チャンク数（時間的特徴） , 特徴数)
    data_3d_reshape = np.zeros((20355, 16, 17, 3))
    for i in range(20355):
        for j in range(16):
            data_3d_reshape[i, j, :, :] = data_3d[i*16 + j, :, :]

    print("data_3d: ", data_3d[16, 3, 2])
    print("data_3d_reshape: ", data_3d_reshape[1, 0, 3, 2])
    # exchange dimension for CNN
    # [20355, 16, 17, 3] -> [20355, 16, 3, 17]
    data_3d_reshape = np.swapaxes(data_3d_reshape, 2, 3)
    
    print("data_3d_reshape.shape: ", data_3d_reshape.shape)
    np.save('/workspace-cloud/toshiki.ohno/EEG_fatigue/EEG_analysis/processedData/data_3d_3freq_flat.npy', data_3d_reshape)