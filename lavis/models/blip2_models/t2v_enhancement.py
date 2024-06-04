import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class T2V_TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask,
        vid_len,
        pos: Optional[Tensor] = None,
    ):
        # print('before src shape :', src.shape)
        pos_src = self.with_pos_embed(src, pos)
        q, k, v = (pos_src[:vid_len], pos_src[vid_len:], src[vid_len:])

        assert q.shape[0] == vid_len
        qmask, kmask = src_mask[:, :vid_len].unsqueeze(2), src_mask[
            :, vid_len:
        ].unsqueeze(1)
        attn_mask = (
            torch.matmul(qmask.float(), kmask.float())
            .bool()
            .repeat(self.nhead, 1, 1)
        )

        src2 = self.self_attn(
            q,
            k,
            value=v,
            attn_mask=attn_mask,
            key_padding_mask=src_mask[:, vid_len:],
        )[0]

        # src2 = src2.permute(1, 0, 2)
        src2 = src[:vid_len] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src = torch.cat([src2, src[vid_len:]], dim=0)
        return src

    def forward_pre(
        self,
        src,
        src_mask,
        vid_len,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        pos_src = self.with_pos_embed(src2, pos)
        q, k, v = (
            pos_src[:vid_len],
            pos_src[vid_len:],
            src2[vid_len:],
        )
        assert q.shape[0] == vid_len
        qmask, kmask = src_mask[:, :vid_len].unsqueeze(2), src_mask[
            :, vid_len:
        ].unsqueeze(1)
        attn_mask = (
            torch.matmul(qmask.float(), kmask.float())
            .bool()
            .repeat(self.nhead, 1, 1)
        )

        src2 = self.self_attn(
            q,
            k,
            value=v,
            attn_mask=attn_mask,
            key_padding_mask=src_mask[:, vid_len:],
        )[0]

        # qmask: B, T, 1 -> T, B, 1
        # src2: T, B, C
        src2 = src[:vid_len] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)  # * qmask.permute(1, 0, 2)
        src = torch.cat([src2, src[vid_len:]], dim=0)
        return src

    def forward(
        self,
        src,
        src_mask,
        vid_len,
        pos: Optional[Tensor] = None,
        **kwargs,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, vid_len, pos)
        # For tvsum, add kwargs
        return self.forward_post(src, src_mask, vid_len, pos, **kwargs)


class TransformerEncoder(nn.Module):
    def __init__(
        self, encoder_layer, num_layers, norm=None, return_intermediate=False
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    # for tvsum, add kwargs
    def forward(
        self,
        src,
        src_mask,
        vid_len,
        pos: Optional[Tensor] = None,
        **kwargs,
    ):
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                src_mask,
                vid_len,
                pos=pos,
                **kwargs,
            )
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


if __name__ == "__main__":
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    activation = "relu"
    normalize_before = False

    t2v_encoder_layer = T2V_TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    t2v_encoder = TransformerEncoder(
        t2v_encoder_layer, num_encoder_layers, encoder_norm
    )

    text = torch.rand(1, 10, 512)
    video = torch.rand(1, 100, 512)

    text_mask = torch.ones(1, 10).bool()
    video_mask = torch.ones(1, 100).bool()

    src = torch.cat((video, text), dim=1).permute(1, 0, 2)
    src_mask = torch.cat((video_mask, text_mask), dim=1)

    pos = None  # torch.rand(1, 110, 512)
    output = t2v_encoder(src, src_mask, 100, pos)
    output = output.permute(1, 0, 2)

    enhanced_video = output[:, :100]
    enhanced_text = output[:, 100:]

    print(enhanced_video.shape)
    print(enhanced_text.shape)
    # print(output.shape)
