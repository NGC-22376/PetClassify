import torch
import torch.nn as nn
import torch.nn.functional as F

class MHSA(nn.Module):
    def __init__(self, dim, num_heads, use_layer_scale=False, use_residual=True):
        super(MHSA, self).__init__()
        self.num_heads = num_heads
        self.use_layer_scale = use_layer_scale  #层缩放机制
        self.use_residual = use_residual  #残差连接
        self.scale = nn.Parameter(torch.ones(1)) if use_layer_scale else None  #self.scale缩放因子

        self.query_dense = nn.Linear(dim, dim)  #查询
        self.key_dense = nn.Linear(dim, dim)   #键
        self.value_dense = nn.Linear(dim, dim)   #值   生成线性变换层
        self.out_dense = nn.Linear(dim, dim)
        self.batch_norm = nn.BatchNorm1d(dim)  # BatchNorm1d对输出进行批量归一化
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)  #输入的第一个维度是批量大小

    def forward(self, x):
        # #将输入的x转换为查询q，键k，值v
        q = self.query_dense(x)
        k = self.key_dense(x)
        v = self.value_dense(x)

        # 对q,k,v进行转置(batch_size, seq_len, dim)变为 (seq_len, batch_size, dim)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output, _ = self.attention(q, k, v)
        attn_output = attn_output.transpose(0, 1)  # 计算自注意力机制的输出，并转置(seq_len, batch_size, dim) 变为(batch_size, seq_len, dim)


        output = self.out_dense(attn_output)  #使用outdense层对attn_output进行线性变换，得到最终输出output

        # Batch Normalization对output进行批量归一化
        output = output.transpose(1, 2)  # 转置(batch_size, seq_len, dim) 变为(batch_size, dim, seq_len)适应BatchNorm1d的要求
        output = self.batch_norm(output)
        output = output.transpose(1, 2)  # 批量归一化后，再次转置回去(batch_size, dim, seq_len) 变为(batch_size, seq_len, dim)
        
        #残差连接，避免梯度消失
        if self.use_residual:
            output = output + x
       
      #对output应用缩放因子
        if self.use_layer_scale:
            output = output * self.scale 

        return output
