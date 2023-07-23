import math
import torch
import torch.nn as nn
from GAT import GAT
import numpy as np


class S_SelfAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1):
        super(S_SelfAttention, self).__init__()

        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def _QK(self, Q, K, sample_k, n_top):
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
        V_sum = V.mean(dim=-2)  # 计算170个V的平均值
        contex = V_sum.unsqueeze(-2).expand(B, H, T, N, V_sum.shape[-1]).clone()

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

        scores_top, index = self._QK(queries, keys, sample_k=U_part, n_top=u)

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


class T_SelfAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1):
        super(T_SelfAttention, self).__init__()

        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def _QK(self, Q, K, sample_k, n_top):
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
        V_sum = V.mean(dim=-2)  # 计算170个V的平均值
        contex = V_sum.unsqueeze(-2).expand(B, H, N, T, V_sum.shape[-1]).clone()

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

        scores_top, index = self._QK(queries, keys, sample_k=U_part, n_top=u)

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


class SImprovedMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SImprovedMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.S_self_attention = S_SelfAttention()  # 空间自注意力
        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        B, N, t, d = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, t, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, t, N, d_k]
        K = self.W_K(input_K).view(B, N, t, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, t, N, d_k]
        V = self.W_V(input_V).view(B, N, t, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, t, N, d_k]

        context = self.S_self_attention(Q, K, V)
        output = self.fc_out(context)
        return output


class TImprovedMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TImprovedMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.T_self_attention = T_SelfAttention()  # 时间自注意力

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        B, N, t, d = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, t, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, t, d_k]
        K = self.W_K(input_K).view(B, N, t, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, t, d_k]
        V = self.W_V(input_V).view(B, N, t, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, t, d_k]

        context = self.T_self_attention(Q, K, V)  # [B, h, N, t, d_k]
        output = self.fc_out(context)
        return output


class SpatialTransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion, device):
        super(SpatialTransformer, self).__init__()

        # Spatial Embedding
        self.adj = adj.to(device)
        self.embed_linear = nn.Linear(adj.shape[0], embed_size)

        self.attention = SImprovedMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.device = device
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_spatial_transformer):
        B, N, t, d = input_spatial_transformer.shape
        # spatial position embedding
        spatial_position_embedding = self.embed_linear(self.adj)  # [N, d]
        spatial_position_embedding = spatial_position_embedding.expand(B, t, N, d)  # [B, t, N, d]相当于在第2维复制了t份, 第一维复制B份
        spatial_position_embedding = spatial_position_embedding.permute(0, 2, 1, 3)  # [B, N, t, d]

        input_Q = input_spatial_transformer + spatial_position_embedding
        input_K = input_spatial_transformer + spatial_position_embedding
        input_V = input_spatial_transformer + spatial_position_embedding

        attention = self.attention(input_Q, input_K, input_V)  # (B, N, t, d)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + input_Q))
        forward = self.feed_forward(x)
        X_S = self.dropout(self.norm2(forward + x))

        return X_S  # (B, N, t, d)


class PositionEncoding(nn.Module):
    def __init__(self, embed_size, max_len=100):
        super(PositionEncoding, self).__init__()

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


class TimeCoding(nn.Module):
    def __init__(self, embed_size):
        super(TimeCoding, self).__init__()

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


class TemporalPositionEmbedding(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super(TemporalPositionEmbedding, self).__init__()

        self.position_encoding = PositionEncoding(embed_size=embed_size)
        self.time_coding = TimeCoding(embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x + self.position_encoding(x) + self.time_coding(x_mark)
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion, device):
        super(TemporalTransformer, self).__init__()

        self.time_num = time_num
        self.temporal_position_embedding = TemporalPositionEmbedding(embed_size, dropout)
        self.attention = TImprovedMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, input_temporal_transformer, time_features):
        B, N, t, d = input_temporal_transformer.shape

        # 融合时间特征信息
        input_encoder = torch.zeros([0, t, d]).to(self.device)
        for b in range(B):
            input_encoder_tmp = self.temporal_position_embedding(input_temporal_transformer[b, :, :, :], time_features[b, :, :])
            input_encoder = torch.cat([input_encoder, input_encoder_tmp], dim=0).to(self.device)

        input_encoder = input_encoder.view(B, N, t, d)
        input_Q = input_encoder
        input_K = input_encoder
        input_V = input_encoder

        attention = self.attention(input_Q, input_K, input_V)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + input_Q))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class SpatialBlock(nn.Module):
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
        super(SpatialBlock, self).__init__()

        self.STransformer = SpatialTransformer(embed_size, heads, adj, dropout, forward_expansion, device)
        self.gat_layer = GAT(device, embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.adj = adj

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)
        self.device = device

    def forward(self, input_spatial_block):
        input_spatial_block = self.dropout(input_spatial_block)
        input_gat_layer = input_spatial_block
        input_spatial_transformer = input_spatial_block
        B, N, t, d = input_spatial_block.shape

        # GAT 部分
        X_G = torch.Tensor(B, N, 0, d).to(self.device)
        # 对每一个时间步进行图注意力处理
        for i in range(input_gat_layer.shape[2]):
            o, attention_weights = self.gat_layer(input_gat_layer[:, :, i, :], self.adj)  # [B, N, d]
            o = o.unsqueeze(2)  # shape [N, 1, d] [B, N, 1, d]
            X_G = torch.cat((X_G, o), dim=2)
        # 最后X_G [B, N, t, d]

        # Spatial Transformer 部分
        X_S = self.STransformer(input_spatial_transformer)

        # 融合 STransformer and GAT
        g = torch.sigmoid(self.fs(X_S) + self.fg(X_G))
        out = g * X_S + (1 - g) * X_G

        return out  # (B, N, t, d)


class GRULayer(nn.Module):
    def __init__(self, in_channels, embed_size):
        super(GRULayer, self).__init__()

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

    def forward(self, x):
        [B, N, t, d] = x.shape
        x = x.reshape(-1, t, d)

        x_short = x[:, 24:, :]  # 近期数据
        x_long = x[:, :24, :]  # 周期数据
        x_long_week = x_long[:, :12, :]
        x_long_day = x_long[:, 12:, :]

        # 对于recent time使用short——gru
        x_short, _ = self.short_gru(x_short)
        # 对周周期与日周期使用long--gru
        x_long_week, _ = self.long_gru(x_long_week)
        x_long_day, _ = self.long_gru(x_long_day)

        x_long = torch.cat((x_long_week, x_long_day), dim=1)
        output_gru_layer = torch.cat((x_long, x_short), dim=1)
        output_gru_layer = output_gru_layer.reshape(B, N, t, -1)

        return output_gru_layer


class PredictionLayer(nn.Module):
    def __init__(self, T_dim, output_T_dim, embed_size):
        super(PredictionLayer, self).__init__()

        # 缩小时间维度。
        self.conv1 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv2 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, input_prediction_layer):
        out = self.relu(self.conv1(input_prediction_layer))  # 等号左边 out shape: [B, T, N, d]
        out = out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, d, N, T]
        out = self.conv2(out)  # 等号左边 out shape: [B, 1, N, T]
        out = out.squeeze(1)

        return out


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

        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.forward_expansion = forward_expansion

        self.gru_layer = GRULayer(in_channels, embed_size)

        self.spatial_block = SpatialBlock(
            adj,
            embed_size,
            heads,
            time_num,
            forward_expansion,
            dropout,
            device
        )

        self.temporal_transformer = TemporalTransformer(
            embed_size,
            heads,
            time_num,
            dropout,
            forward_expansion,
            device
        )

        self.prediction_layer = PredictionLayer(T_dim, output_T_dim, embed_size)

    def forward(self, x, time_features):
        x = x.to(self.device)
        time_features = time_features.to(self.device)
        x = x.permute(0, 2, 3, 1)  # [B, N, t, 1]

        input_spatial_block = self.gru_layer(x)

        output_spatial_block = self.norm1(self.spatial_block(input_spatial_block) + input_spatial_block)

        output_temporal_transformer = self.dropout(self.norm2(self.temporal_transformer(output_spatial_block, time_features) + output_spatial_block))

        input_prediction_layer = output_temporal_transformer.permute(0, 2, 1, 3)
        out = self.prediction_layer(input_prediction_layer)

        return out  # [B, N, T]



