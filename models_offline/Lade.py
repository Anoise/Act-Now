import torch
import torch.nn.functional as F


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
        out_channels= configs.d_model * 2
        self.pred_len = configs.pred_len
        
        self.l_m1 = torch.nn.Linear(1, out_channels//4)
        self.l_m2 = torch.nn.Linear(out_channels//4, 1)
        
        self.l_s1 = torch.nn.Linear(1, out_channels//4)
        self.l_s2 = torch.nn.Linear(out_channels//4, 1)

        print("Statistics Model ...")
    
    def forward(self, x_m, x_s):
        
        m1 = self.l_m1(x_m)
        m1 = F.relu(m1)
        m2 = self.l_m2(m1)

        s1 = self.l_s1(x_s)
        s1 = F.relu(s1)
        s2 = self.l_s2(s1)
        
        return m2, s2


class DecomModel(torch.nn.Module):
    def __init__(self, configs):
        super(DecomModel, self).__init__()
        out_channels= configs.d_model * 2
        self.pred_len = configs.pred_len
        
        self.l_x1 = torch.nn.Linear(configs.seq_len, out_channels)
        self.l_x2 = torch.nn.Linear(out_channels, configs.pred_len)
        
        self.l_o1 = torch.nn.Linear(configs.pred_len, out_channels)
        self.l_o2 = torch.nn.Linear(out_channels, configs.pred_len)

        print("Decomposition Model ...")

    def forward(self, x, mean, std, edge_indexs=None, edge_weights=None):
        
        x1 = self.l_x1(x)
        x1 = F.relu(x1)
        x2 = self.l_x2(x1)

        out = x2 * std

        out = self.l_o1(out)
        out = F.relu(out)
        out = self.l_o2(out) + mean

        return out
    


class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.stati_model = StatiModel(configs)
        self.decom_model = DecomModel(configs)
        
        print("Lade Model ...")
        
    def forward(self, x, y=None, criterion=None, edge_indexs=None, edge_weights=None):
        
        ### decomposition of x
        _x, _mean, _std = self.decompose(x)
        
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