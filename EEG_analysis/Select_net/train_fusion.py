from select_select import *
from toolbox import *
import matplotlib.pyplot as plt
import torch
import numpy as np
from dataloader_3freq_3d import *
import wandb
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")

'''
file name: DE_4D_Feature
input: 1. epoch
       2. optimizer

output: model.pth

Warning!!!!!
Before train begin, start up the visdom to Real-time monitoring accuracy
'''

# save highest accuracy model.pth
acc_low = 0.6
acc_list = []
acc_max = 0.7
test_accuracy = 0
# model
#myModel = Toshi_Net_3freq_select()
myModel = select_net_train()
myModel = myModel.to(device)

# loss function
loss_fn = torch.nn.MSELoss()

# optimizer
learningRate = 2e-3
optimizer = torch.optim.AdamW(myModel.parameters(), lr=learningRate, weight_decay=0.01) # AdamW

epoch = 40
# Record total step and loss
total_train_step = 0
total_test_step = 0
total_train_loss = 0
total_test_loss = 0

epochs = list(range(1, epoch + 1))  # epochの値のリスト [1, 2, 3, ..., epoch]
epoch_train_losses = []
epoch_train_accuracies = []
epoch_test_losses = []
epoch_test_accuracies = []
epoch_end = False 
removed_electrodes = []
no_improvement_count = 0

acc_mean_train = []
acc_mean_test = []
    

for i in range(epoch):
    print("--------------The {}th epoch of training starts------------".format(i + 1))
    total_train_loss = 0
    total_train_acc = 0
    train_step = 0

    # 初期化
    wandb.init(
        # プロジェクト名を設定
        project="project_EEG_3freq_channel_select_crossW", 
        # 実行名をつける(これまでのように、指定しない場合は、自動で作成してくれます）
        name=f"run_{2}", 
        # ハイパーパラメータ等の情報を保存
        config={
        "learning_rate": 0.001,
        "dataset": "SEED-VIG",
        "epochs": 200,
        }
    )

    print("remove:", removed_electrodes)
    print("count:", no_improvement_count)

    fold_accuracies_train = []  # 各foldの精度を格納するリストをエポックごとに初期化
    fold_accuracies_test = []

    # 5分割交差検証のループ
    for n in range(5):
        print(f"===== Fold {n + 1} =====")
        total_train_loss = 0
        total_train_acc = 0
        train_step = 0

        data = np.load('/workspace-cloud/toshiki.ohno/EEG_fatigue/EEG_analysis/processedData/data_3d_3freq_flat.npy') 
        label = np.load('/workspace-cloud/toshiki.ohno/EEG_fatigue/EEG_analysis/processedData/label.npy')

        data = torch.FloatTensor(data)
        label = torch.FloatTensor(label)

        # データローダーの作成
        train_dataloader, test_dataloader = myDataset_5cv(data, label, batch_size, n, seed)
    
        ### 1.TRAIN ###
        for data in train_dataloader:
            x, y = data
            x = x.to(device) #[150, 16, 3, 6, 9]
            y = y.to(device) #[150, 1]

            # ignore attention output when training
            outputs, removed_electrodes, no_improvement_count = myModel(x, i, test_accuracy, epoch_end)
            #print(outputs.shape)
            if epoch_end:
                epoch_end = False
            
            train_loss = loss_fn(outputs, y)
            # Calculate accuracy by toolbox.label_2class
            label_train = label_3class(y.cpu())
            label_train_pred = label_3class(outputs.cpu())
            total_train_acc += accuracy_score(label_train, label_train_pred)
            
            # Gradient update
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # Calculate loss
            total_train_loss = total_train_loss + train_loss.item()
            train_step = train_step + 1
            total_train_step = total_train_step + 1

            if total_train_step % 10 == 0:
                print("train step：{}，train average loss：{:.6f}".format(total_train_step, total_train_loss/train_step))
                #total_train_loss = 0
        
    # メトリクスの保存
        epoch_train_losses.append(total_train_loss / len(train_dataloader))
        epoch_train_accuracies.append(100.0 * (total_train_acc / len(train_dataloader)))

        # in this epoch, train accuracy
        print("train average accuracy: {:.4f}%".format(100.0 * (total_train_acc/len(train_dataloader))))

        # 各foldの精度をリストに格納
        fold_accuracies_train.append(total_train_acc/len(train_dataloader))

        ### 2.TEST ###
        total_test_acc = 0
        total_test_loss = 0
        test_step = 0
        r2 = 0
        attentionGraph = torch.Tensor()
        with torch.no_grad():
            for data in test_dataloader:
                testx, testy = data
                testx = testx.to(device)
                testy = testy.to(device)
                #print(testx.shape)
                outputs, removed_electrodes, no_improvement_count = myModel(testx, i, test_accuracy, epoch_end)
                #print(outputs.shape)
                
                label = label_3class(testy.cpu())
                label_pred = label_3class(outputs.cpu())
                test_loss = loss_fn(outputs, testy)
                total_test_loss = total_test_loss + test_loss.item()
                total_test_step = total_test_step + 1
                test_step = test_step + 1
                
                # use toolbox.myEvaluate to calculate
                conf, acc, report, pre, recall, f1, kappa = myEvaluate(label.cpu().numpy(), label_pred.cpu().numpy())
                # total accuracy
                total_test_acc += acc

        print("test average loss:{:.6f}".format(total_test_loss / test_step))
        # in this epoch test accuracy
        print("test average accuracy:{:.4f}%".format(100.0 * (total_test_acc / len(test_dataloader))))
        
        epoch_test_losses.append(total_test_loss / len(test_dataloader))
        epoch_test_accuracies.append(100.0 * (total_test_acc / len(test_dataloader)))

        # 評価値の記録
        wandb.log({"accuracy": 100.0 * (total_test_acc / len(test_dataloader)), "loss": total_test_loss / total_test_step})
        test_accuracy = total_test_acc / len(test_dataloader)

        #  If test accuracy more than the acc_low, save the model.pth
        if (total_test_acc/len(test_dataloader)) > acc_low:
            acc_low = (total_test_acc/len(test_dataloader))
            # 保存ファイル
            #torch.save(myModel.state_dict(), f'/content/drive/MyDrive/ColabNotebooks/SEED-VIGfile/Toshi_net/pth/model_fold_channelchange_cross_2.pth')
            #torch.save(myModel.state_dict(), f'/content/drive/MyDrive/ColabNotebooks/SEED-VIGfile/Toshi_net/pth/model_fold_channelchange_cross_Wno.pth')
        epoch_end = True
        # 各foldの精度をリストに格納
        fold_accuracies_test.append(total_test_acc/len(test_dataloader))

    # これまでのfoldの平均精度を出力
    if n == 4:
        avg_acc_so_far_train = np.mean(fold_accuracies_train)
        avg_acc_so_far_test = np.mean(fold_accuracies_test)

        acc_mean_train.append(avg_acc_so_far_train) 
        acc_mean_test.append(avg_acc_so_far_test) 
        print(f"Average train accuracy across folds 1-{n + 1}: {avg_acc_so_far_train:.4f}")
        print(f"Average test accuracy across folds 1-{n + 1}: {avg_acc_so_far_test:.4f}")

        print(acc_mean_test)
        best_score = max(acc_mean_test)
        if (best_score > acc_max):
            acc_max = best_score
            # 保存ファイル
            torch.save(myModel.state_dict(), f'/workspace-cloud/toshiki.ohno/EEG_fatigue/EEG_analysis/model_evaluation/model_fold_3freq_cross_mea-stdhalf.pth')


print("This is ", n, " fold, highest accuracy is: ", acc_low)
print("This is ", n, " fold, highest accuracy is: ", acc_max)

print(f"conf = {conf}")
print(f"acc = {acc}")
print(f"report = {report}" )
print(f"pre = {pre}" )
print(f"recall = {recall}" )
print(f"f1 = {f1}"  )
print(f"kappa = {kappa}" )

# 実行の終了を伝える
wandb.finish()

# グラフの描画
# Plot Training and Test Metrics
plt.figure(figsize=(12, 5))  # Adjust figure size for better readability

for i in range(5):
    fold_losses_train = epoch_train_losses[i::5]
    plt.plot(epochs[:len(fold_losses_train)], fold_losses_train, label=f'Training Loss (Fold {i+1})')
for i in range(5):
    fold_losses_test = epoch_test_losses[i::5]
    plt.plot(epochs[:len(fold_losses_test)], fold_losses_test, label=f'Training Loss (Fold {i+1})', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig('training_loss_cross_channelselect_W.png')  # 画像ファイルとして保存
plt.close()

for i in range(5):
    fold_accuracy_train = epoch_train_accuracies[i::5]
    plt.plot(epochs[:len(fold_accuracy_train)], fold_accuracy_train, label=f'Training Loss (Fold {i+1})')
for i in range(5):
    fold_accuracy_test = epoch_test_accuracies[i::5]
    plt.plot(epochs[:len(fold_accuracy_test)], fold_accuracy_test, label=f'Training Loss (Fold {i+1})', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()   
plt.savefig('training_accuracy_cross_channelselect_W.png')  # 画像ファイルとして保存
plt.close()

# 5分割交差検証のループ後、以下のように修正
avg_epoch_train_losses = []
for i in range(epoch):
  avg_loss = np.mean(epoch_train_losses[i*5:(i+1)*5])
  avg_epoch_train_losses.append(avg_loss)

# 5分割交差検証のループ後、以下のように修正
avg_epoch_test_losses = []
for i in range(epoch):
  avg_loss = np.mean(epoch_test_losses[i*5:(i+1)*5])
  avg_epoch_test_losses.append(avg_loss)

# プロット
plt.figure(figsize=(12, 5))
plt.plot(epochs, avg_epoch_train_losses, label='Training Validation Loss')  # 修正
plt.plot(epochs, avg_epoch_test_losses, label='Test Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Validation Loss')
plt.legend()
plt.savefig('avg_loss_plot_cross_channelselect_W.png')
plt.show()



plt.plot(epochs, acc_mean_train, label='Training Accuracy')
plt.plot(epochs, acc_mean_test, label='Test Accuracy', linestyle='--') # Dashed line for test
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy Cross Validation')
plt.legend()   
plt.savefig('avg_training_accuracy_cross_validation_channelselect_W.png')  # 画像ファイルとして保存
plt.close()
