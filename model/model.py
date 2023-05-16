import torch
from torch import nn
import torch.nn.functional as F
import math


class PositionalDependentLayer(nn.Module):
    #@profile
    def __init__(self, N, input_dim, output_dim, layer_num=1, grid_arrangement_str='gray_code', coord_batch_size=1, hidden_layers=4):
        super().__init__()
        self.N = N  
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.grid_arrangement_str = grid_arrangement_str
        self.coord_B = coord_batch_size
        self.total_hidden_layers = hidden_layers

        self.weight_tile = nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False) for i in range(self.N)])
        for i in range(self.N):
            torch.nn.init.kaiming_uniform_(self.weight_tile[i].weight)
        self.bias = torch.zeros(output_dim).cuda()
        self.tile_id = None

    #@profile
    def get_affine_transform(self, in_coords, layer_num=1):
        if self.tile_id is None:
            H = int(math.sqrt(self.N))
            _A = torch.tensor([])
            _b = None

            if self.grid_arrangement_str == 'gray_code':
                if layer_num == 1:
                    _A = 2
                elif layer_num > 1:
                    _A = 2 ** (layer_num-1)
                _b = 0.5

            elif self.grid_arrangement_str == 'quad_tree':
                pass

            elif self.grid_arrangement_str == 'fine_to_coarse':
                _A = 2 ** (10-layer_num)
                _b = 0.0
                
            affine_feats = in_coords*_A + _b
            x, y = affine_feats[:, 0], affine_feats[:, 1]
            x = torch.floor(x).long() % H
            y = torch.floor(y).long() % H
            self.tile_id = (H*x + y).view(self.coord_B, 1)
        return self.tile_id      

    #@profile
    def positional_dependent_linear_1d(self, in_feats, in_coords):
        r"""Linear layer with position-dependent weight.
        Assuming the input coordinate is 2D
        Args:
            self.weight_tile (N list): Tile of N weight(Cout, Cin) matrices
            self.bias (Cout tensor): Bias vector
            in_feats (B * Cin tensor): Batched input features
            in_coords (B * 2): Batched input coordinates
        Returns:
            out_feats (B * Cout tensor): Batched output features
        """
        N = self.N # Tile size
        B = in_feats.shape[0]
        Cout = self.output_dim # Out channel count
        Cin = self.input_dim # In channel count

        tile_id = self.get_affine_transform(in_coords, layer_num=self.layer_num)    # (B, 1)

        out_feats = torch.empty([B, Cout]).cuda()
        for t in range(N):
            mask = tile_id == t # (B, 1)
            mask = mask.cuda()
            sel_in_feats = torch.masked_select(in_feats, mask)  # (number of tile_id == t)
            sel_in_feats = sel_in_feats.reshape(-1, Cin)
            sel_weight = self.weight_tile[t].weight.cuda()
            sel_out_feats = sel_in_feats @ sel_weight.T
            out_feats.masked_scatter_(mask, sel_out_feats)
        
        return out_feats + self.bias

    #@profile
    def forward(self, args):
        in_feats, in_coords = args

        B = self.coord_B # Coord Batch size
        output = None
        B_remain = 0    # batch 크기씩 자르고 난 이후, 남은 픽셀들 개수 0 <= B_remain < B
        
        for idx in range(0, in_feats.shape[0], B):
            if in_feats.shape[0] - idx < B:
                B_remain = in_feats.shape[0] - idx

            out = self.positional_dependent_linear_1d(in_feats[idx:idx+B, :], in_coords[idx:idx+B, :]).view(B, -1)
            if output is not None:
                output = torch.cat((output, out), 0)
            else:
                output = out
        
        if B_remain > 0:
            idx = in_feats.shape[0] - B_remain
            out = self.positional_dependent_linear_1d(in_feats[idx:, :], in_coords[idx:, :]).view(B_remain, -1)
            output = torch.cat((output, out), 0)

        output = torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)(output)

        if not (self.layer_num == self.total_hidden_layers+1): # number of hidden layers.
            return output, in_coords
        else:
            return output


class MLP(nn.Module):
    def __init__(self, tile_size, input_dim, hidden_features, hidden_layers, output_dim, grid_arrangement_str='gray_code', coord_batch_size=1):
        super().__init__()
        
        self.N = tile_size
        self.net = []
        self.net.append(PositionalDependentLayer(self.N, input_dim, hidden_features, layer_num=1, grid_arrangement_str=grid_arrangement_str, 
                                                 coord_batch_size=coord_batch_size, hidden_layers=hidden_layers))

        for i in range(hidden_layers):
            self.net.append(PositionalDependentLayer(self.N, hidden_features, hidden_features, layer_num=i+2, grid_arrangement_str=grid_arrangement_str, 
                                                     coord_batch_size=coord_batch_size, hidden_layers=hidden_layers))
            
        self.net.append(nn.Linear(hidden_features, output_dim, bias=True))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)   # (sidelen, 2)
        output = self.net((coords, coords))
        return output