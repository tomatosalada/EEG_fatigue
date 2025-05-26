import torch
import numpy as np
from toolbox import myDataset_5cv

'''
file name: DE_4D_Feature
input: 1. choose nth fold of 5-fold
       2. batch_size
       3. random seed
       4. the 4D feature of all subjects
       
output: train_dataloader and test_dataloader of nth fold
'''

#  choose nth fold
n = 0
# batch_size
batch_size = 150
# 随机数种子 random seed
seed = 20
# 使用gpu
device = torch.device(("cuda:0") if torch.cuda.is_available() else "cpu")

### 1.数据集 ### Dataset
data = np.load('/content/drive/MyDrive/ColabNotebooks/SEED-VIGfile/Toshi_net/processedData/data_4d.npy')   # 20355  
label = np.load('./processedData/label.npy')

data = torch.FloatTensor(data)
label = torch.FloatTensor(label)

print(data.shape)
print(label.shape)
print(label)

# 五折交叉验证 5-fold cross validation
train_dataloader, test_dataloader = myDataset_5cv(data, label, batch_size, n, seed)