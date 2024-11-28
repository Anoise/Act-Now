import torch
import torch.nn.functional as F
from torch import optim

class LadeOptimizer:
    def __init__(self, model, lr):
        self.stati_optim = optim.AdamW(model.stati_model.parameters(), lr=lr)
        self.decom_optim = optim.AdamW(model.decom_model.parameters(), lr=lr)
        
    def zero_grad(self):
        self.stati_optim.zero_grad()
        self.decom_optim.zero_grad()
    
    def step(self):
        self.stati_optim.step()
        self.decom_optim.step()


def get_edges(x, gama):
    x = F.normalize(x, dim=-1)
    edge = x.matmul(x.T)
    edge[edge< gama] = 0 
    edge[edge>= gama] = 1 
    spa_edge = edge.to_sparse().coalesce()
    indices = spa_edge.indices().long()
    values = spa_edge.values().float()

    return x, indices, values


class StatiModel(torch.nn.Module):
    def __init__(self, configs):
        super(StatiModel, self).__init__()
        
        self.ln1 = torch.nn.Linear(configs.seq_len, configs.d_model)
        self.ln2 = torch.nn.Linear(configs.d_model, configs.pred_len)

        print("Statistics Model ...")
    
    def forward(self, x_m, x_s):
        
        stati = torch.cat((x_m,x_s),-1)
        
        h1 = self.ln1(stati.permute(0,2,1))
        h1 = F.relu(h1)
        h2 = self.ln2(h1).permute(0,2,1)
        
        return h2.chunk(2,-1)


class DecomModel(torch.nn.Module):
    def __init__(self, configs):
        super(DecomModel, self).__init__()
        out_channels= configs.d_model * 2
        self.pred_len = configs.pred_len

        self.ln_x1 = torch.nn.Linear(configs.seq_len, out_channels)
        self.ln_x2 = torch.nn.Linear(out_channels, configs.pred_len)
        
        self.ln_o1 = torch.nn.Linear(configs.pred_len, out_channels)
        self.ln_o2 = torch.nn.Linear(out_channels, configs.pred_len)
        
        print("Decomposition Model ...")

    def forward(self, x, mean, std, edge_indexs=None, edge_weights=None):

        h1 = self.ln_x1(x.permute(0,2,1))
        h1 = F.relu(h1)
        h2 = self.ln_x2(h1)

        o1 = h2 * std.permute(0,2,1)

        o1 = self.ln_o1(o1)
        o1 = F.relu(o1)
        o2 = self.ln_o2(o1).permute(0,2,1) + mean

        return o2
    


class Lade(torch.nn.Module):
    def __init__(self, configs):
        super(Lade, self).__init__()
        
        self.stati_model = StatiModel(configs)
        self.decom_model = DecomModel(configs)
        
        print("Lade Model ...")
        
    def forward(self, x, y=None, criterion=None, edge_indexs=None, edge_weights=None):
        
        ### decomposition of x
        _x, _mean, _std = self.decompose(x)
        # print(_x.shape, _mean.shape, _std.shape,'xxx')
        mean, std = self.stati_model(_mean, _std)

        out = self.decom_model(_x, mean.detach(), std.detach())
        
        if y is not None:
            assert criterion is not None
            
            _, y_mean, y_std = self.decompose(y)

            stati_loss = criterion(mean, y_mean) + criterion(std, y_std)

            loss = criterion(out, y)
            
            return out, loss, stati_loss

        return out
        
    def decompose(self, inputs):
        means = inputs.mean(-1, keepdim=True).detach()
        inputs = inputs - means
        stdev = torch.sqrt(torch.var(inputs, dim=-1, keepdim=True, unbiased=False) + 1e-7)
        inputs /= stdev
        return inputs, means, stdev