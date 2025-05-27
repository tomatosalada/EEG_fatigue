from select_transformer import *
#from selectivetrans import *
from toolbox import *
import matplotlib.pyplot as plt
import torch
from dataloader_3freq_3d import *
import wandb
import warnings
import numpy as np
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
acc_low = 0.75
acc_list = []
# model
myModel = select_net()
myModel = myModel.to(device)
# loss function
loss_fn = torch.nn.MSELoss()

# optimizer
learningRate = 1e-3
optimizer = torch.optim.AdamW(myModel.parameters(), lr=learningRate, weight_decay=0.01) # AdamW

epoch = 200
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

for i in range(epoch):
    print("--------------The {}th epoch of training starts------------".format(i + 1))
    total_train_loss = 0
    total_train_acc = 0
    step = 0

    # 初期化
    wandb.init(
        # プロジェクト名を設定
        project="project_EEG_select_3freq_cnn_weight1", 
        # 実行名をつける(これまでのように、指定しない場合は、自動で作成してくれます）
        name=f"run_{6}", 
        # ハイパーパラメータ等の情報を保存
        config={
        "learning_rate": 0.02,
        "dataset": "SEED-VIG",
        "epochs": 200,
        }
    )
    
    ### 1.TRAIN ###
    for data in train_dataloader:
        x, y = data
        x = x.to(device) #[150, 16, 3, 17]
        y = y.to(device) #[150, 1]
        
        outputs, attention_weights, electrode_attention = myModel(x)
        #outputs= myModel(x)
        train_loss = loss_fn(outputs, y)
        # Calculate accuracy by toolbox.label_2class
        label_train = label_2class(y.cpu())
        label_train_pred = label_2class(outputs.cpu())
        total_train_acc += accuracy_score(label_train, label_train_pred)
        
        # Gradient update
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # Calculate loss
        total_train_loss = total_train_loss + train_loss.item()
        
        total_train_step = total_train_step + 1
        step = step + 1
        if total_train_step % 10 == 0:
            print("train step：{}，train average loss：{:.6f}".format(total_train_step, total_train_loss/step))
            #total_train_loss = 0
        """
        if total_train_step % 10 == 0:  # 10ステップごとにログを記録
            # wandbにログを記録
            wandb.log({"train_loss": train_loss.item()})

            # electrode_attention をヒートマップとして表示
            electrode_attention = electrode_attention.cpu().detach().numpy()  # NumPy配列に変換
            electrode_attention_map = np.mean(electrode_attention, axis=(0, 1))  # 層とヘッドで平均
            electrode_attention_map = electrode_attention_map[:, np.newaxis] * electrode_attention_map[np.newaxis, :]  # ブロードキャストを利用

            plt.figure(figsize=(8, 6))
            plt.imshow(electrode_attention_map, cmap='viridis')
            plt.colorbar()
            plt.xlabel('Electrode')
            plt.ylabel('Electrode')
            plt.title('Electrode Attention Map')
            # wandbにログを記録
            wandb.log({"electrode_attention_map": plt})
            plt.close()
            """
    
# メトリクスの保存
    epoch_train_losses.append(total_train_loss / len(train_dataloader))
    epoch_train_accuracies.append(100.0 * (total_train_acc / len(train_dataloader)))

    # in this epoch, train accuracy
    print("train average accuracy: {:.4f}%".format(100.0 * (total_train_acc/len(train_dataloader))))

    ### 2.TEST ###
    total_test_acc = 0
    total_test_loss = 0
    total_test_step = 0
    r2 = 0
    attentionGraph = torch.Tensor()
    with torch.no_grad():
        for data in test_dataloader:
            testx, testy = data
            testx = testx.to(device)
            testy = testy.to(device)
            outputs, attention_weights_test, electrode_attention_test = myModel(testx)
            #outputs = myModel(testx)
            label = label_2class(testy.cpu())
            label_pred = label_2class(outputs.cpu())
            test_loss = loss_fn(outputs, testy)
            total_test_loss = total_test_loss + test_loss.item()
            total_test_step = total_test_step + 1
            
            # use toolbox.myEvaluate to calculate
            conf, acc, report, pre, recall, f1, kappa = myEvaluate(label.cpu().numpy(), label_pred.cpu().numpy())
            # total accuracy
            total_test_acc += acc

            if total_test_step % 10 == 0:  # 10ステップごとにログを記録
            # wandbにログを記録 (テストデータ)
                wandb.log({"test_loss": test_loss.item()})
                
                # electrode_attention をヒートマップとして表示
                electrode_attention_test = electrode_attention_test.cpu().detach().numpy()  # NumPy配列に変換
                electrode_attention_map_test = np.mean(electrode_attention_test, axis=(0, 1))  # 層とヘッドで平均
                electrode_attention_map_test = electrode_attention_map_test[:, np.newaxis] * electrode_attention_map_test[np.newaxis, :]  # ブロードキャストを利用

                plt.figure(figsize=(8, 6))
                plt.imshow(electrode_attention_map_test, cmap='viridis')
                plt.colorbar()
                plt.xlabel('Electrode')
                plt.ylabel('Electrode')
                plt.title('Electrode Attention test Map')
                # wandbにログを記録
                wandb.log({"electrode_attention_map_test": plt})
                plt.close()

    print("test average loss:{:.6f}".format(total_test_loss / total_test_step))
    # in this epoch test accuracy
    print("test average accuracy:{:.4f}%".format(100.0 * (total_test_acc / len(test_dataloader))))
    
    epoch_test_losses.append(total_test_loss / len(test_dataloader))
    epoch_test_accuracies.append(100.0 * (total_test_acc / len(test_dataloader)))

    # 評価値の記録
    wandb.log({"accuracy": 100.0 * (total_test_acc / len(test_dataloader)), "loss": total_test_loss / total_test_step})

    #  If test accuracy more than the acc_low, save the model.pth
    if (total_test_acc/len(test_dataloader)) > acc_low:
        acc_low = (total_test_acc/len(test_dataloader))
        #torch.save(myModel.state_dict(), f'/content/drive/MyDrive/ColabNotebooks/SEED-VIGfile/Toshi_net/pth/model_%dfold_select_try_best.pth' % n)
        torch.save(myModel.state_dict(), f'/workspace-cloud/toshiki.ohno/EEG_fatigue/EEG_analysis/model_evaluation/model_%dfold_select_holdout.pth' % n)
        # 冗長な電極を削除
        removed_electrodes, threshold = remove_redundant_electrodes(attention_weights_test, electrode_attention_test)
        print("Removed electrodes:", removed_electrodes)
print("This is ", n, " fold, highest accuracy is: ", acc_low)

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


plt.plot(epochs, epoch_train_losses, label='Training Loss')
plt.plot(epochs, epoch_test_losses, label='Test Loss', linestyle='--') # Dashed line for test
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig('training_loss_select_mean.png')  # 画像ファイルとして保存
plt.close()


plt.plot(epochs, epoch_train_accuracies, label='Training Accuracy')
plt.plot(epochs, epoch_test_accuracies, label='Test Accuracy', linestyle='--') # Dashed line for test
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()   
plt.savefig('training_accuracy_select_mean.png')  # 画像ファイルとして保存
plt.close()