from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from .tools.resnet import Resnet1D
from .tools.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from collections import OrderedDict

import torch_geometric
from torch_geometric.nn import GCNConv

from ..utils.body_parts import UBODY_IDX, LBODY_IDX, UBODY_nfeats, LBODY_nfeats, div_body_feats, IPOSE_PATH


class VQVae(nn.Module):

    def __init__(self,
                 nfeats: int,
                 quantizer: str = "ema_reset",
                 code_num=[512, 512],
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()

        self.code_dim = code_dim

        self.ubody_idx = torch.tensor(UBODY_IDX)
        self.lbody_idx = torch.tensor(LBODY_IDX)

        self.nfeats = nfeats
        self.ubody_nfeats = UBODY_nfeats
        self.lbody_nfeats = LBODY_nfeats

        # self.ubody_encoder = Encoder(self.ubody_nfeats,
        #                        output_emb_width,
        #                        down_t,
        #                        stride_t,
        #                        width,
        #                        depth,
        #                        dilation_growth_rate,
        #                        activation=activation,
        #                        norm=norm)
        # self.lbody_encoder = Encoder(self.lbody_nfeats,
        #                        output_emb_width,
        #                        down_t,
        #                        stride_t,
        #                        width,
        #                        depth,
        #                        dilation_growth_rate,
        #                        activation=activation,
        #                        norm=norm)
        

        # self.ubody_decoder = Decoder(self.ubody_nfeats,
        #                        output_emb_width,
        #                        down_t,
        #                        stride_t,
        #                        width,
        #                        depth,
        #                        dilation_growth_rate,
        #                        activation=activation,
        #                        norm=norm)
        # self.lbody_decoder = Decoder(self.lbody_nfeats,
        #                        output_emb_width,
        #                        down_t,
        #                        stride_t,
        #                        width,
        #                        depth,
        #                        dilation_growth_rate,
        #                        activation=activation,
        #                        norm=norm)


        self.encoder = Encoder((self.ubody_nfeats, self.lbody_nfeats),
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        
        self.decoder = Decoder((self.ubody_nfeats, self.lbody_nfeats),
                                 output_emb_width,
                                 down_t,
                                 stride_t,
                                 width,
                                 depth,
                                 dilation_growth_rate,
                                 activation=activation,
                                 norm=norm)

        self.body_part_merge = nn.Linear(self.ubody_nfeats + self.lbody_nfeats, nfeats)

        self.ubody_code_num = code_num[0]
        self.lbody_code_num = code_num[1]

        if quantizer == "ema_reset":
            self.ubody_quantizer = QuantizeEMAReset(self.ubody_code_num, code_dim, mu=0.99)
            self.lbody_quantizer = QuantizeEMAReset(self.lbody_code_num, code_dim, mu=0.99)
        elif quantizer == "orig":
            self.ubody_quantizer = Quantizer(self.ubody_code_num, code_dim, beta=1.0)
            self.lbody_quantizer = Quantizer(self.lbody_code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.ubody_quantizer = QuantizeEMA(self.ubody_code_num, code_dim, mu=0.99)
            self.lbody_quantizer = QuantizeEMA(self.lbody_code_num, code_dim, mu=0.99)
        elif quantizer == "reset":
            self.ubody_quantizer = QuantizeReset(self.ubody_code_num, code_dim)
            self.lbody_quantizer = QuantizeReset(self.lbody_code_num, code_dim)
    

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)

        ubody_x, lbody_x = div_body_feats(x)
        ubody_x = ubody_x.permute(0, 2, 1)
        lbody_x = lbody_x.permute(0, 2, 1)

        return ubody_x, lbody_x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
        # Preprocess
        # features: (bs, len_seq, len_feats[263])
        ux_in, lx_in = self.preprocess(features)

        # Encode
        # ux_encoder = self.ubody_encoder(ux_in)
        # lx_encoder = self.lbody_encoder(lx_in)
        ux_encoder, lx_encoder = self.encoder(ux_in, lx_in)

        # quantization
        ux_quantized, ubody_loss, ubody_perplexity = self.ubody_quantizer(ux_encoder)
        lx_quantized, lbody_loss, lbody_perplexity = self.lbody_quantizer(lx_encoder)

        # decoder
        # ux_decoder = self.ubody_decoder(ux_quantized)
        # lx_decoder = self.lbody_decoder(lx_quantized)
        ux_decoder, lx_decoder = self.decoder(ux_quantized, lx_quantized)
        ux_out = self.postprocess(ux_decoder) # (bs, T, self.ubody_nfeats)
        lx_out = self.postprocess(lx_decoder) # (bs, T, self.lbody_nfeats)

        x_out = self.body_part_merge(torch.cat([ux_out, lx_out], dim=-1)) # (bs, T, nfeats)
    
        return x_out, ux_out, lx_out, ubody_loss, lbody_loss, ubody_perplexity, lbody_perplexity

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        N, T, _ = features.shape
        ux_in, lx_in = self.preprocess(features)
        
        ux_encoder = self.ubody_encoder(ux_in)
        ux_encoder = self.postprocess(ux_encoder)
        ux_encoder = ux_encoder.contiguous().view(-1, ux_encoder.shape[-1])  # (NT, C)
        ucode_idx = self.ubody_quantizer.quantize(ux_encoder)
        ucode_idx = ucode_idx.view(N, -1, 1)
        
        lx_encoder = self.lbody_encoder(lx_in)
        lx_encoder = self.postprocess(lx_encoder)
        lx_encoder = lx_encoder.contiguous().view(-1, lx_encoder.shape[-1])  # (NT, C)
        lcode_idx = self.lbody_quantizer.quantize(lx_encoder)
        lcode_idx = lcode_idx.view(N, -1, 1)
        
        code_idx = torch.cat([ucode_idx, lcode_idx], dim=-1) # (N, T, 2)

        # latent, dist
        return code_idx, None
    

    def decode(self, z, no_uz=False, no_lz=False):

        ### Output feat inclues:
        ### root_angular_vel (1), root_linear_vel (2), root_y_pos (1), local_joint_pos (21*3), local_joint_rot (21*6), local_joint_vel (22*3), foot_contact (4)

        # if z.dim() == 2:
        #     z = z.unsqueeze(0)
        
        uz, lz = z[..., 0], z[..., 1]

        if no_uz and no_lz:

            print('uz and lz are None. Returning Identity Pose')
            x_out = torch.tensor(np.load(IPOSE_PATH))


        if no_uz and not no_lz:

            lx_d = self.lbody_quantizer.dequantize(lz)
            lx_d = lx_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
            lx_decoder = self.lbody_decoder(lx_d)
            lx_out = self.postprocess(lx_decoder)

            ipose = torch.tensor(np.load(IPOSE_PATH)).to(lz.device)
            ipose = ipose.view(1, 1, -1)
            ipose = ipose.repeat(1, lx_out.shape[1], 1)
            ux_out, _ = div_body_feats(ipose)

            ux_out[:, :, :4] = lx_out[:, :, :4]

            x_out = self.body_part_merge(torch.cat([ux_out, lx_out], dim=-1)) # (bs, T, nfeats)

        elif not no_uz and no_lz:

            ux_d = self.ubody_quantizer.dequantize(uz)
            ux_d = ux_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
            ux_decoder = self.ubody_decoder(ux_d)
            ux_out = self.postprocess(ux_decoder)

            ipose = torch.tensor(np.load(IPOSE_PATH)).to(uz.device)
            ipose = ipose.view(1, 1, -1)
            ipose = ipose.repeat(1, ux_out.shape[1], 1)
            _, lx_out = div_body_feats(ipose)

            lx_out[:, :, :4] = ux_out[:, :, :4]

            x_out = self.body_part_merge(torch.cat([ux_out, lx_out], dim=-1)) # (bs, T, nfeats)

        else:
            ux_d = self.ubody_quantizer.dequantize(uz)
            lx_d = self.lbody_quantizer.dequantize(lz)
            ux_d = ux_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
            lx_d = lx_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

            # decoder
            ux_decoder = self.ubody_decoder(ux_d)
            lx_decoder = self.lbody_decoder(lx_d)

            ux_out = self.postprocess(ux_decoder)
            lx_out = self.postprocess(lx_decoder)

            x_out = self.body_part_merge(torch.cat([ux_out, lx_out], dim=-1)) # (bs, T, nfeats)

        return x_out

# class Encoder(nn.Module):

#     def __init__(self,
#                  input_emb_width=3,
#                  output_emb_width=512,
#                  down_t=3,
#                  stride_t=2,
#                  width=512,
#                  depth=3,
#                  dilation_growth_rate=3,
#                  activation='relu',
#                  norm=None):
#         super().__init__()

#         blocks = []
#         filter_t, pad_t = stride_t * 2, stride_t // 2
#         blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())

#         for i in range(down_t):
#             input_dim = width
#             block = nn.Sequential(
#                 nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
#                 Resnet1D(width,
#                          depth,
#                          dilation_growth_rate,
#                          activation=activation,
#                          norm=norm),
#             )
#             blocks.append(block)
#         blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
#         self.model = nn.Sequential(*blocks)

#     def forward(self, x):
#         x = self.model(x)
#         return x
    

class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=(UBODY_nfeats, LBODY_nfeats),
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        filter_t, pad_t = stride_t * 2, stride_t // 2

        self.max_embed_width = max(input_emb_width)
        self.gcn_out_dim = output_emb_width
        self.lbody_upsample = nn.Conv1d(input_emb_width[1], self.max_embed_width, 3, 1, 1)
        self.gcn = GCNConv(self.max_embed_width, width)
        self.edge_idx = torch.tensor([[0, 1], [1, 0]])

        ubody_blocks, lbody_blocks = [], []

        for i in range(down_t):
            input_dim = width
            ubody_block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            ubody_blocks.append(ubody_block)

            lbody_block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            lbody_blocks.append(lbody_block)
        
        ubody_blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        lbody_blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        
        self.ubody_model = nn.Sequential(*ubody_blocks)
        self.lbody_model = nn.Sequential(*lbody_blocks)


    def forward(self, ux, lx):

        # ux: (bs, 163, T), lx: (bs, 107, T)

        lx = self.lbody_upsample(lx) # (bs, 163, T)

        x = torch.cat([ux.unsqueeze(1), lx.unsqueeze(1)], dim=1) # (bs, 2, 163, T)

        bs, _, n, t = x.shape

        gcn_out = []
        for i in range(t):
            x_t = x[:, :, :, i]
            x_t = x_t.view(-1, n)
            try:
                x_t = self.gcn(x_t, self.edge_idx)
            except:
                self.gcn.to(x_t.device)
                self.edge_idx = self.edge_idx.to(x_t.device)
                x_t = self.gcn(x_t, self.edge_idx)

            gcn_out.append(x_t.view(bs, 2, self.gcn_out_dim))
        
        gcn_out = torch.stack(gcn_out, dim=-1) # (bs, 2, 512, T)

        ux, lx = torch.split(gcn_out, 1, dim=1) # (bs, 1, 512, T)

        ux = self.ubody_model(ux.squeeze(1)) # (bs, 512, T/4)
        lx = self.lbody_model(lx.squeeze(1)) # (bs, 512, T/4)

        return ux, lx


# class Decoder(nn.Module):

#     def __init__(self,
#                  input_emb_width=3,
#                  output_emb_width=512,
#                  down_t=3,
#                  stride_t=2,
#                  width=512,
#                  depth=3,
#                  dilation_growth_rate=3,
#                  activation='relu',
#                  norm=None):
#         super().__init__()
#         blocks = []

#         filter_t, pad_t = stride_t * 2, stride_t // 2
#         blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
#         for i in range(down_t):
#             out_dim = width
#             block = nn.Sequential(
#                 Resnet1D(width,
#                          depth,
#                          dilation_growth_rate,
#                          reverse_dilation=True,
#                          activation=activation,
#                          norm=norm), nn.Upsample(scale_factor=2,
#                                                  mode='nearest'),
#                 nn.Conv1d(width, out_dim, 3, 1, 1))
#             blocks.append(block)
#         blocks.append(nn.Conv1d(width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
#         blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
#         self.model = nn.Sequential(*blocks)

#     def forward(self, x):
#         x = self.model(x)
#         return x


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=(UBODY_nfeats, LBODY_nfeats),
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        ubody_blocks, lbody_blocks = [], []

        ubody_blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        ubody_blocks.append(nn.ReLU())
        lbody_blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        lbody_blocks.append(nn.ReLU())

        for i in range(down_t):
            out_dim = width
            ubody_block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            ubody_blocks.append(ubody_block)

            lbody_block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            lbody_blocks.append(lbody_block)

        ubody_blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        ubody_blocks.append(nn.ReLU())
        ubody_blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        self.ubody_model = nn.Sequential(*ubody_blocks)

        lbody_blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        lbody_blocks.append(nn.ReLU())
        lbody_blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        self.lbody_model = nn.Sequential(*lbody_blocks)

        self.gcn_out_dim = max(input_emb_width)
        self.gcn = GCNConv(width, self.gcn_out_dim)
        self.edge_idx = torch.tensor([[0, 1], [1, 0]])

        self.ubody_conv = nn.Conv1d(self.gcn_out_dim, input_emb_width[0], 3, 1, 1)
        self.lbody_conv = nn.Conv1d(self.gcn_out_dim, input_emb_width[1], 3, 1, 1)

        
    def forward(self, ux, lx):
        
        ux_decoder = self.ubody_model(ux) # (bs, 512, T)
        lx_decoder = self.lbody_model(lx) # (bs, 512, T)

        x = torch.cat([ux_decoder.unsqueeze(1), lx_decoder.unsqueeze(1)], dim=1) # (bs, 2, 512, T)

        bs, _, n, t = x.shape

        gcn_out = []
        for i in range(t):
            x_t = x[:, :, :, i]
            x_t = x_t.view(-1, n)
            try:
                x_t = self.gcn(x_t, self.edge_idx)
            except:
                self.gcn.to(x_t.device)
                self.edge_idx = self.edge_idx.to(x_t.device)
                x_t = self.gcn(x_t, self.edge_idx)

            gcn_out.append(x_t.view(bs, 2, self.gcn_out_dim))

        gcn_out = torch.stack(gcn_out, dim=-1) # (bs, 2, 163, T)
                
        ux = self.ubody_conv(gcn_out[:, 0, :, :])
        lx = self.lbody_conv(gcn_out[:, 1, :, :])

        return ux, lx
