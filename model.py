import math
import torch
import torch.nn as nn
from GAT import GAT
import numpy as np
import pandas as pd


class S_ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(S_ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):

        B, H, T, N, D = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, T, N, N, D)
        index_sample = torch.randint(N, (N, sample_k))  # 想要170个传感器的Q所取的K各不相同，取值范围为[0,N),大小为(N, sample_k)
        K_sample = K_expand[:, :, :, torch.arange(N).unsqueeze(1), index_sample, :]  # 广播
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), sample_k)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None, None],
                   torch.arange(H)[None, :, None, None],
                   torch.arange(T)[None, None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, N):
        B, H, T, N, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)  # 计算170个V的平均值
            contex = V_sum.unsqueeze(-2).expand(B, H, T, N, V_sum.shape[-1]).clone()
        else:  # use mask   decoder的self-attention使用
            contex = V.cumsum(dim=-2)  # 倒数第二维进行累加操作
        return contex

    def _update_context(self, context_in, V, scores, index, N):
        B, H, T, N, D = V.shape

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None, None],
        torch.arange(H)[None, :, None, None],
        torch.arange(T)[None, None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # 将活跃的query进行更新

        return (context_in, None)

    def forward(self, queries, keys, values):

        B, H, T, N, D = queries.shape

        U_part = self.factor * np.ceil(np.log(N)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(N)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < N else N
        u = u if u < N else N

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = 1. / math.sqrt(D)
        scores_top = scores_top * scale  # active query得分计算完毕
        # get the context
        context = self._get_initial_context(values, N)  # 计算得到平均V
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, N)
        context = context.permute(0, 3, 2, 1, 4).contiguous()
        context = context.reshape(B, N, T, -1)
        return context


class T_ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(T_ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):

        B, H, N, T, D = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, N, T, T, D)
        index_sample = torch.randint(T, (T, sample_k))  # 想要36个时间步的Q所取的K各不相同，取值范围为[0,T),大小为(T, sample_k)
        K_sample = K_expand[:, :, :, torch.arange(T).unsqueeze(1), index_sample, :]  # 广播
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), sample_k)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None, None],
                   torch.arange(H)[None, :, None, None],
                   torch.arange(N)[None, None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, T):
        B, H, N, T, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)  # 计算170个V的平均值
            contex = V_sum.unsqueeze(-2).expand(B, H, N, T, V_sum.shape[-1]).clone()
        else:  # use mask   decoder的self-attention使用
            contex = V.cumsum(dim=-2)  # 倒数第二维进行累加操作
        return contex

    def _update_context(self, context_in, V, scores, index, T):
        B, H, N, T, D = V.shape

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None, None],
        torch.arange(H)[None, :, None, None],
        torch.arange(N)[None, None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # 将活跃的query进行更新

        return (context_in, None)

    def forward(self, queries, keys, values):

        B, H, N, T, D = queries.shape

        U_part = self.factor * np.ceil(np.log(T)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(T)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < T else T
        u = u if u < T else T

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = 1. / math.sqrt(D)
        scores_top = scores_top * scale  # active query得分计算完毕
        # get the context
        context = self._get_initial_context(values, T)  # 计算得到平均V
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, T)
        context = context.permute(0, 2, 3, 1, 4).contiguous()
        context = context.reshape(B, N, T, -1)
        return context


class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.S_self_attention = S_ProbAttention()  # 空间自注意力
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]

        context = self.S_self_attention(Q, K, V)
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.T_self_attention = T_ProbAttention()  # 时间自注意力

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        context = self.T_self_attention(Q, K, V)  # [B, h, N, T, d_k]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output


class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion, device):
        super(STransformer, self).__init__()
        # Spatial Embedding
        self.adj = adj
        self.D_S = adj.to(device)
        self.embed_liner = nn.Linear(adj.shape[0], embed_size)

        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        # 使用GAT进行图注意力计算
        self.gat = GAT(device, embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)
        self.device = device

    def forward(self, value, key, query):
        B, N, T, C = query.shape
        D_S = self.embed_liner(self.D_S)  # [N, C]
        D_S = D_S.expand(B, T, N, C)  # [B, T, N, C]相当于在第2维复制了T份, 第一维复制B份
        D_S = D_S.permute(0, 2, 1, 3)  # [B, N, T, C]


        # GAT 部分
        X_G = torch.Tensor(B, N, 0, C).to(self.device)
        # 对每一个时间步进行图注意力处理
        for t in range(query.shape[2]):
            o, attention_weights = self.gat(query[:, :, t, :], self.adj)  # [B, N, C]
            o = o.unsqueeze(2)  # shape [N, 1, C] [B, N, 1, C]
            X_G = torch.cat((X_G, o), dim=2)
        # 最后X_G [B, N, T, C]

        # Spatial Transformer 部分
        query = query + D_S
        attention = self.attention(query, query, query)  # (B, N, T, C)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))

        # 融合 STransformer and GAT
        g = torch.sigmoid(self.fs(U_S) + self.fg(X_G))  # (7)
        out = g * U_S + (1 - g) * X_G  # (8)

        return out  # (B, N, T, C)


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_size, max_len=100):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TemporalEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(TemporalEmbedding, self).__init__()

        minute_size = 70
        hour_size = 30
        weekday_size = 10
        month_size = 40
        year_size = 400

        Embed = nn.Embedding

        self.minute_embed = Embed(minute_size, embed_size)
        self.hour_embed = Embed(hour_size, embed_size)
        self.weekday_embed = Embed(weekday_size, embed_size)
        self.month_embed = Embed(month_size, embed_size)
        self.year_embed = Embed(year_size, embed_size)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, 0])
        hour_x = self.hour_embed(x[:, 1])
        weekday_x = self.weekday_embed(x[:, 2])
        month_x = self.month_embed(x[:, 3])
        year_x = self.year_embed(x[:, 4])

        return minute_x + hour_x + weekday_x + month_x + year_x


class DataEmbedding(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.position_embedding = PositionalEmbedding(embed_size=embed_size)
        self.temporal_embedding = TemporalEmbedding(embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion, device):
        super(TTransformer, self).__init__()

        self.time_num = time_num
        self.encoder_embedding = DataEmbedding(embed_size, dropout)
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, value, key, query, t, time_encoder):
        B, N, T, C = query.shape

        input_encoder = torch.zeros([0, T, C]).to(self.device)
        for b in range(B):
            input_encoder_tmp = self.encoder_embedding(query[b, :, :, :], time_encoder[b, :, :])
            input_encoder = torch.cat([input_encoder, input_encoder_tmp], dim=0).to(self.device)

        input_encoder = input_encoder.view(B, N, T, C)

        attention = self.attention(input_encoder, input_encoder, input_encoder)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + input_encoder))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out



class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, device, dropout, forward_expansion):
        super(STTransformerBlock, self).__init__()
        self.STransformer = STransformer(embed_size, heads, adj, dropout, forward_expansion, device)  # 对应论文spatial Block
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion, device)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t, time_encoder):
        x1 = self.norm1(self.STransformer(value, key, query) + query)  # (B, N, T, C)
        x2 = self.dropout(self.norm2(self.TTransformer(x1, x1, x1, t, time_encoder) + x1))
        return x2


### Encoder
class Encoder(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            dropout,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layer = STTransformerBlock(
                    embed_size,
                    heads,
                    adj,
                    time_num,
                    device,
                    dropout,
                    forward_expansion)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t, time_encoder):
        # x: [N, T, C]  [B, N, T, C]
        out = self.dropout(x)
        # In the Encoder the query, key, value are all the same.
        out = self.layer(out, out, out, t, time_encoder)
        return out



class Transformer(nn.Module):
    def __init__(
            self,
            adj,
            embed_size,
            heads,
            time_num,
            forward_expansion,
            dropout,
            device
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            dropout
        )
        self.device = device

    def forward(self, src, t, time_encoder):
        ## scr: [N, T, C]   [B, N, T, C]
        enc_src = self.encoder(src, t, time_encoder)
        return enc_src  # [B, N, T, C]




class STGAGRTN(nn.Module):
    def __init__(
            self,
            adj,
            in_channels,
            embed_size,
            time_num,
            T_dim,
            output_T_dim,
            heads,
            forward_expansion,
            dropout,
            device):
        super(STGAGRTN, self).__init__()

        self.forward_expansion = forward_expansion
        # 第一次卷积扩充通道数
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)

        self.Transformer = Transformer(
            adj,
            embed_size,
            heads,
            time_num,
            forward_expansion,
            dropout,
            device
        )
        self.short_gru = nn.GRU(in_channels,
                                embed_size,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=False)
        self.long_gru = nn.GRU(in_channels,
                               embed_size,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=False)

        # 缩小时间维度。
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, x, time_encoder):
        x = x.to(self.device)
        time_encoder = time_encoder.to(self.device)
        ####
        x = x.permute(0, 2, 3, 1)
        [B, N, S, d] = x.shape
        x = x.reshape(-1, S, d)
        x_short = x[:, 24:, :]  # hour sample
        x_long = x[:, :24, :]  # history sample
        x_long_week = x_long[:, :12, :]
        x_long_day = x_long[:, 12:, :]
        # 对于recent time使用short——gru
        input_Transformer_short, _ = self.short_gru(x_short)
        # 对周周期与日周期使用long--gru
        x_long_week, _ = self.long_gru(x_long_week)
        x_long_day, _ = self.long_gru(x_long_day)
        input_Transformer_long = torch.cat((x_long_week, x_long_day), dim=1)
        input_Transformer = torch.cat((input_Transformer_long, input_Transformer_short), dim=1)
        input_Transformer = input_Transformer.reshape(B, N, S, -1)
        ####

        # input_Transformer = self.conv1(x)
        # input_Transformer = input_Transformer.permute(0, 2, 3, 1)

        output_Transformer = self.Transformer(input_Transformer, self.forward_expansion, time_encoder)
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)

        out = self.relu(self.conv2(output_Transformer))  # 等号左边 out shape: [1, output_T_dim, N, C]
        out = out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, C, N, output_T_dim]
        out = self.conv3(out)  # 等号左边 out shape: [B, 1, N, output_T_dim]
        out = out.squeeze(1)

        return out  # [B, N, output_dim]



