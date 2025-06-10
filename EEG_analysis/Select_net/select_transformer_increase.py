import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.transforms import Compose, Resize, ToTensor
#from einops import rearrange, reduce, repeat
#from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
import math
import matplotlib.pyplot as plt
import numpy as np
from dataloader_3freq_3d import *

#from torchinfo import summary
#from torchviz import make_dot


torch.manual_seed(970530)
torch.cuda.manual_seed_all(970530)

class Make2DMap(nn.Module):
    def __init__(self, img_rows=6, img_cols=9):
        super(Make2DMap, self).__init__()
        self.img_rows = img_rows
        self.img_cols = img_cols

    def forward(self, x):
        #x->[150, 16, 5, 17]
        # 2D map for 17 channels
        batch_size, seq_len, freq, channel = x.shape
        map2D = torch.zeros((batch_size, seq_len, freq, self.img_rows, self.img_cols))  # [150, 16, 5, 17]->[150, 16, 5, 6, 9]
        #各チャネルを二次元マップに割り当てる
        # 'FT7'(channel1) :
        map2D[:, :, :, 0, 0] = x[:, :, :, 0]
        # 'FT8'(channel2) :
        map2D[:, :, :, 0, 8] = x[:, :, :, 1]
        # 'T7' (channel3) :
        map2D[:, :, :, 1, 0] = x[:, :, :, 2]
        # 'T8' (channel4) :
        map2D[:, :, :, 1, 8] = x[:, :, :, 3]
        # 'TP7'(channel5) :
        map2D[:, :, :, 2, 0] = x[:, :, :, 4]
        # 'TP8'(channel6) :
        map2D[:, :, :, 2, 8] = x[:, :, :, 5]
        # 'CP1'(channel7) :
        map2D[:, :, :, 2, 3] = x[:, :, :, 6]
        # 'CP2'(channel8) :
        map2D[:, :, :, 2, 5] = x[:, :, :, 7]
        # 'P1' (channel9) :
        map2D[:, :, :, 3, 3] = x[:, :, :, 8]
        # 'PZ' (channel10):
        map2D[:, :, :, 3, 4] = x[:, :, :, 9]
        # 'P2' (channel11):
        map2D[:, :, :, 3, 5] = x[:, :, :, 10]
        # 'PO3'(channel12):
        map2D[:, :, :, 4, 3] = x[:, :, :, 11]
        # 'POZ'(channel13):
        map2D[:, :, :, 4, 4] = x[:, :, :, 12]
        # 'PO4'(channel14):
        map2D[:, :, :, 4, 5] = x[:, :, :, 13]
        # 'O1' (channel15):
        map2D[:, :, :, 5, 3] = x[:, :, :, 14]
        # 'OZ' (channel16):
        map2D[:, :, :, 5, 4] = x[:, :, :, 15]
        # 'O2' (channel17):
        map2D[:, :, :, 5, 5] = x[:, :, :, 16]
        
        return map2D.to(device) #[150, 16, 5, 6, 9]

class InputOperate(nn.Module):
    def __init__(self):
        super(InputOperate, self).__init__()

    def forward(self, x, removed_electrodes):
        for i in removed_electrodes:
            if i == 1:
                # 'FT7'(channel1) :
                x[:, :, :, 0, 0] = 0
            elif i == 2:    
                # 'FT8'(channel2) :
                x[:, :, :, 0, 8] = 0
            elif i == 3: 
                # 'T7' (channel3) :
                x[:, :, :, 1, 0] = 0
            elif i == 4: 
                # 'T8' (channel4) :
                x[:, :, :, 1, 8] = 0
            elif i == 5: 
                # 'TP7'(channel5) :
                x[:, :, :, 2, 0] = 0
            elif i == 6: 
                # 'TP8'(channel6) :
                x[:, :, :, 2, 8] = 0
            elif i == 7: 
                # 'CP1'(channel7) :
                x[:, :, :, 2, 3] = 0
            elif i == 8: 
                # 'CP2'(channel8) :
                x[:, :, :, 2, 5] = 0
            elif i == 9: 
                # 'P1' (channel9) :
                x[:, :, :, 3, 3] = 0
            elif i == 10: 
                # 'PZ' (channel10):
                x[:, :, :, 3, 4] = 0
            elif i == 11: 
                # 'P2' (channel11):
                x[:, :, :, 3, 5] = 0
            elif i == 12: 
                # 'PO3'(channel12):
                x[:, :, :, 4, 3] = 0
            elif i ==13: 
                # 'POZ'(channel13):
                x[:, :, :, 4, 4] = 0
            elif i == 14: 
                # 'PO4'(channel14):
                x[:, :, :, 4, 5] = 0
            elif i == 15: 
                # 'O1' (channel15):
                x[:, :, :, 5, 3] = 0
            elif i == 16: 
                # 'OZ' (channel16):
                x[:, :, :, 5, 4] = 0
            elif i == 17: 
                # 'O2' (channel17):
                x[:, :, :, 5, 5] = 0
        return x

class Onlyremove(nn.Module):
    def __init__(self):
        super(Onlyremove, self).__init__()

    def forward(self, x, removed_electrodes):
        for i in removed_electrodes:
            if i == 1:
                # 'FT7'(channel1) :
                x[:, :, :, 0] = 0
            elif i == 2:    
                # 'FT8'(channel2) :
                x[:, :, :, 1] = 0
            elif i == 3: 
                # 'T7' (channel3) :
                x[:, :, :, 2] = 0
            elif i == 4: 
                # 'T8' (channel4) :
                x[:, :, :, 3] = 0
            elif i == 5: 
                # 'TP7'(channel5) :
                x[:, :, :, 4] = 0
            elif i == 6: 
                # 'TP8'(channel6) :
                x[:, :, :, 5] = 0
            elif i == 7: 
                # 'CP1'(channel7) :
                x[:, :, :, 6] = 0
            elif i == 8: 
                # 'CP2'(channel8) :
                x[:, :, :, 7] = 0
            elif i == 9: 
                # 'P1' (channel9) :
                x[:, :, :, 8] = 0
            elif i == 10: 
                # 'PZ' (channel10):
                x[:, :, :, 9] = 0
            elif i == 11: 
                # 'P2' (channel11):
                x[:, :, :, 10] = 0
            elif i == 12: 
                # 'PO3'(channel12):
                x[:, :, :, 11] = 0
            elif i ==13: 
                # 'POZ'(channel13):
                x[:, :, :, 12] = 0
            elif i == 14: 
                # 'PO4'(channel14):
                x[:, :, :, 13] = 0
            elif i == 15: 
                # 'O1' (channel15):
                x[:, :, :, 14] = 0
            elif i == 16: 
                # 'OZ' (channel16):
                x[:, :, :, 15] = 0
            elif i == 17: 
                # 'O2' (channel17):
                x[:, :, :, 16] = 0
        return x

class WeightedAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_heads) / num_heads).cuda()  # 初期値は均等

    def forward(self, attention_weights):
        # ソフトマックスで重み係数を正規化
        normalized_weights = F.softmax(self.weights, dim=0)
        # 各ヘッドのAttention weightに重み係数を掛けて平均
        weighted_avg_attention = sum(
            w * attention.mean(dim=0)
            for w, attention in zip(normalized_weights, attention_weights)
        )
        return weighted_avg_attention

def remove_redundant_electrodes(attention_weights, electrode_attention):
    
    """
    Attention スコアを用いて冗長な電極を削除する関数

    Args:
        attention_weights (list): 各層のAttention weightを格納したリスト
        electrode_attention (torch.Tensor): 電極ごとのAttention weight
        threshold (float): Attention スコアの閾値

    Returns:
        list: 削除する電極のインデックスのリスト
    """

    num_layers = len(attention_weights)
    num_heads = len(attention_weights[0])
    batch_size, num_electrodes, _ = attention_weights[0][0].shape

    weighted_attention = WeightedAttention(num_heads).cuda()  # WeightedAttentionモジュールをインスタンス化

    # 各電極ペアのAttentionスコアの平均値を計算
    avg_attention_scores = torch.zeros((num_electrodes, num_electrodes)).cuda()
    for l in range(num_layers):
        # 各層のAttention weightをWeightedAttentionモジュールに入力
        avg_attention_scores += weighted_attention(attention_weights[l])
    avg_attention_scores /= num_layers  # 層数で平均化
    
    # Attentionスコアの平均値と標準偏差を計算
    mean_attention = avg_attention_scores.mean()
    std_attention = avg_attention_scores.std()

    # 平均値 + 標準偏差を閾値とする

    #threshold = mean_attention + std_attention

    threshold = mean_attention 


    print(threshold)
    # 閾値を超えた電極ペアを削除
    removed_electrodes = []
    for i in range(num_electrodes):
        for j in range(i + 1, num_electrodes):
            if avg_attention_scores[i, j] > threshold:
                print(f"Electrode pair ({i}, {j}) exceeds threshold with score: {avg_attention_scores[i, j]}")  
                # iとjのうち、平均Attentionスコアが小さい方を削除
                if (electrode_attention.mean(axis=0)[i] < electrode_attention.mean(axis=0)[j]).all():
                    removed_electrodes.append(i)
                else:
                    removed_electrodes.append(j)
    # 重複を削除
    removed_electrodes = list(set(removed_electrodes))

    return removed_electrodes, threshold

# depthwise separable convolution(DS Conv):
# depthwise conv + pointwise conv + bn + relu
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernal_size = kernel_size
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Context module in DSC module
class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3x3BNReLU, self).__init__()
        self.conv3x3 = depthwise_separable_conv(in_channel, out_channel, 3)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv3x3(x)))

class ContextModule(nn.Module):
    def __init__(self, in_channel):
        super(ContextModule, self).__init__()
        self.stem = Conv3x3BNReLU(in_channel, in_channel // 2)
        self.branch1_conv3x3 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)
        self.branch2_conv3x3_1 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)
        self.branch2_conv3x3_2 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)

    def forward(self, x):
        x = self.stem(x)
        # branch1
        x1 = self.branch1_conv3x3(x)
        # branch2
        x2 = self.branch2_conv3x3_1(x)
        x2 = self.branch2_conv3x3_2(x2)
        # concat
        return torch.cat([x1, x2], dim=1)

#Transformer
class Config:
    def __init__(self):
        # モデルの設定
        self.hidden_size = 64        # 隠れ層の次元数
        self.num_atention_heads = 17   # アテンションヘッドの数
        self.num_hidden_layers = 4    # Transformer Encoder層の数
        self.intermediate_size = 256  # FeedForwardネットワークの中間層の次元数
        self.hidden_dropout_prob = 0.3 # ドロップアウト率
        self.num_embedding = 32
        self.num_labels = 1            # 分類クラス数

def scaled_dot_product_attention(query, key, value):
    dim_k = torch.tensor(query.size(-1))  # torch.Sizeをテンソルに変換
    scores = torch.bmm(query, key.transpose(1, 2) / torch.sqrt(dim_k))
    weights = F.softmax(scores, dim = -1)
    return torch.bmm(weights, value), weights

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs, weights = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_atention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(num_heads * head_dim, embed_dim)
    
    def forward(self, hidden_state):
        # 各ヘッドからの出力をリストに格納
        attn_outputs = []
        all_weights = []  # 各ヘッドのアテンション重みを格納するリスト

        for h in self.heads:
            attn_output, weights = h(hidden_state)  # 各ヘッドの出力を取得
            attn_outputs.append(attn_output)
            all_weights.append(weights)  # アテンション重みをリストに追加

        x = torch.cat(attn_outputs, dim=2)  # アテンション出力を結合
        x = self.output_linear(x)
        # all_weightsをコピー
        original_all_weights = all_weights.copy()

        # 各ヘッドのAttention weightを平均化
        mean_attention_weights = torch.mean(torch.stack(all_weights), dim=0)  # (batch_size, seq_len, seq_len)

        # 電極ごとのAttention weightを計算
        electrode_attention = torch.mean(mean_attention_weights, dim=1)  # (batch_size, seq_len)


        return x, original_all_weights,  electrode_attention  # 電極ごとのAttention weightを追加

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
    
    def forward(self, x):
        #レイヤー正規化を適用し、入力をクエリ、キーバリューにコピー
        hidden_state = self.layer_norm_1(x)
        #スキップ接続付きのアテンションを適用
        # アテンションの出力のみを取得
        x1, attention_weights, electrode_attention = self.attention(hidden_state)  # アテンション重みを取得
        x = x + x1
        # attention_weights をコピー
        original_attention_weights = [weights.clone().cuda() for weights in attention_weights]

        #スキップ接続付きの順伝搬層を適用
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x, original_attention_weights, electrode_attention

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 位置エンコーディングのテンソルを用意
        self.position_embedding = nn.Parameter(torch.zeros(1, config.num_embedding, config.hidden_size), requires_grad=False)
        self.init_position_embedding(config)

    def init_position_embedding(self, config):
        # 位置のインデックスを用意
        position = torch.arange(0, config.num_embedding).unsqueeze(1).float()
        # div_termはサイン・コサインの周期を決定する係数
        div_term = torch.exp(torch.arange(0, config.hidden_size, 2).float() * -(math.log(10000.0) / config.hidden_size))

        # サインとコサインの値を別々に計算
        sin_embedding = torch.sin(position * div_term)
        cos_embedding = torch.cos(position * div_term)

        # サイン・コサインを交互に配置
        position_embedding = torch.zeros(position.size(0), config.hidden_size)
        position_embedding[:, 0::2] = sin_embedding
        position_embedding[:, 1::2] = cos_embedding

        # 計算結果を `self.position_embedding` にコピー
        self.position_embedding.data = position_embedding.unsqueeze(0)

    def forward(self, x):
        # 埋め込みを加算
        return x + self.position_embedding[:, :x.size(1), :]

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, x, return_attention=False):
        all_attention_weights = []  # 全ての層のアテンション重みを格納するリスト
        all_electrode_attentions = []  # 全ての層の電極ごとのAttention weightを格納するリスト
        x = self.embeddings(x)
        for layer in self.layers:
            x, attention_weights, electrode_attention = layer(x)  # 各層のアテンション重みを取得
            # all_attention_weights.append(attention_weights)  # アテンション重みを追加 <- 修正前
            all_attention_weights.append([weights.clone() for weights in attention_weights])  # アテンション重みを追加 <- 修正後
            all_electrode_attentions.append(electrode_attention)  # 電極ごとのAttention weightを追加

            # 最終層の出力を各シーケンスの平均として集約する
        x = x.mean(dim=1)
        if return_attention:
            return x, all_attention_weights, torch.stack(all_electrode_attentions)  # 全ての層のアテンション重みを返す
        else:
            return x
        

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes, config):
        super().__init__()
        # global average pooling
        self.clshead = nn.Sequential(
            nn.Linear(emb_size, 32),
            nn.LayerNorm(32),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


# Toshi-Net:
# Attention module + DSC module + transformer module
class select_net(nn.Module):
    def __init__(self, num_classes=1):
        super(select_net, self).__init__()
        self.conv1 = nn.Conv1d(48, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(32, False)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 48, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(48, False)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
      
        #transformer
        config = Config()
        self.transformer = TransformerEncoder(config)
        self.classification = ClassificationHead(config.hidden_size, num_classes, config)

    def forward(self, x):
        #[batch、時系列、周波数、特徴量]→x=[150, 16, 3, 17]
        batch_size, seq_len, freq_bands, node_dim = x.shape  # 各次元のサイズを取得
        
        x_skip = x.view(batch_size, -1, node_dim)
        x1 = x.view(batch_size, -1, node_dim)
        x1 = self.conv1(x1)
        x1 = self.batchnorm1(x1)
        x2 = self.conv2(x1)
        #x2 = F.dropout(x2, 0.25)
        x1 = self.conv3(x2)

        x = x_skip + x1

        x = self.conv4(x)
        #x = self.conv5(x)
        x = x + x2
        x = x.permute(0, 2, 1)#[150, 17, 16, 3]
              
        #x = x.view(batch_size, -1, node_dim) #[150, 17, 48]
        #x = x.permute(0, 2, 1)
        
        #Transformer
        #out = self.transformer(out)
        # 出力とアテンション重みの取得
        out, attention_weights, electrode_attention = self.transformer(x, return_attention=True)  # return_attention=True を追加
        #out -> [batch, 64]
        out = self.classification(out)

        return out, attention_weights, electrode_attention

if __name__ == '__main__':
    input = torch.rand((32, 16, 3, 17))
    net = select_net()
    output,attention_weights, electrode_attention = net(input)
    print("Input shape     : ", input.shape)
    print("Output shape    : ", output.shape)
    print("Atten shape  : ", electrode_attention.shape)
    # 冗長な電極を削除
    removed_electrodes = remove_redundant_electrodes(attention_weights, electrode_attention)
    print("Removed electrodes:", removed_electrodes)
    #summary(net, input_size=(150, 16, 3, 17))
    # 計算グラフの可視化
    """
    dot = make_dot(output, params=dict(net.named_parameters()))
    dot.render("model_graph", format="png") # model_graph.png に保存
    """