import torch
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import Dataset
from utils.timefeatures import time_features

class LargeGraphDataset(Dataset):
   
    def __init__(self, data_path, edge_path=None, lags = 3, p_len=1,  partition = 8, train=True, sample_freq=1, rand_sample=False, use_edge=False, timeenc=0):
        
        if 'C2TM' in data_path:
            df = pd.read_csv(data_path)/(1024*1024*100)
            df = df.drop('time',axis=1)
            self._data = df.values
        elif 'CBS' in data_path:
            df = pd.read_csv(data_path)
            df = df.drop('time',axis=1)
            self._data = df.values
        else: 
            # self._data = np.load(data_path)
            self._data = np.load(data_path)[...,-1]
            #self._data = np.load(data_path)[...,:2]
            print(self._data.shape, 'Internet ...')
 
        self._data =  torch.FloatTensor(self._data)
        # self.timeenc = timeenc
        # self.data_stamp = self._get_stamp(df[['time']])
        # print('data stamp', self.data_stamp.shape)
        
        self.edge_path = edge_path
        self.lags = lags
        self.p_len = p_len
        self.partition = partition
        self.train = train
        self.rand_sample = rand_sample
        self.use_edge = use_edge
        self.sample_freq = sample_freq

        self.n_node = self._data.shape[1]
        self.n_subnode = self.n_node//self.partition
        self.times =  (self._data.shape[0] - lags - p_len)//self.sample_freq
        
        
        print('data shape: ',self._data.shape, self.n_node, self.sample_freq, self.partition, 'times= ',self.times)
        
        if edge_path is not None:
            with open(edge_path, 'rb') as f:
                self._adj_mx = pickle.load(f)["adj_mx"]
        else:
            self._adj_mx = np.diag([1 for _ in range(self._data.shape[1])])
            print(self._adj_mx.shape, 'adj_max shape ...')

    def _get_stamp(self, df_stamp):
        
        df_stamp['date'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        return data_stamp
    

    def __getitem__(self, time_index):
    
        if self.rand_sample and self.train:
            return self._randn_sample(time_index)
        
        idx = time_index * self.sample_freq 
        
        return self._order_sample(idx)
    
    
    def _order_sample(self, time_index):
        
        part_idx = time_index // self.times // self.sample_freq
        begin_node_idx = part_idx*self.n_subnode
        end_node_idx = (part_idx+1)*self.n_subnode
        time_idx = time_index % self.times
            
        x = self._data[time_idx : time_idx + self.lags, begin_node_idx:end_node_idx]
        y = self._data[time_idx + self.lags: time_idx+ self.lags+ self.p_len, begin_node_idx:end_node_idx]
        
        if self.use_edge:
            edge_index, edge_weight = self._part_order_edges(begin_node_idx, end_node_idx)
            return x,  y, edge_index, edge_weight, time_idx
        # print(time_idx,'time idx', part_idx, begin_node_idx, end_node_idx)
        return x, y, time_idx

    def _randn_sample(self, time_index):
        idxes = np.random.randint(0, self.n_node, self.n_subnode)
        time_idx = time_index % self.times
        x = self._data[time_idx : time_idx + self.lags, idxes]
        y = self._data[time_idx + self.lags: time_idx+ self.lags+ self.p_len, idxes]
        
        if self.use_edge:
            edge_index, edge_weight = self._part_randn_edges(idxes)
            return x,  y, edge_index, edge_weight, time_idx
        
        return x, y, time_idx
    
    def _part_order_edges(self,begin_node_idx, end_node_idx):
        
        idxes = np.arange(begin_node_idx, end_node_idx)
        
        edge_rows = self._adj_mx[idxes]
        edge =  edge_rows[:, idxes]
        edge = torch.Tensor(edge)
        spa_edge = edge.to_sparse().coalesce()
        edge_index = spa_edge.indices().long()
        edge_weight = spa_edge.values().float()
        
        return edge_index, edge_weight

    def _part_randn_edges(self, idxes):
        
        edge_rows = self._adj_mx[idxes]
        edge =  edge_rows[:, idxes]
        edge = torch.Tensor(edge)
        spa_edge = edge.to_sparse().coalesce()
        edge_index = spa_edge.indices().long()
        edge_weight = spa_edge.values().float()
        
        return edge_index, edge_weight

    def __len__(self):
        #n_lastnode = self.n_node % self.partition
        return self.times * self.partition #if n_lastnode==0 else self.times * (self.partition+1)
