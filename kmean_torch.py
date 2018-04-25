import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from tqdm import trange
from torch import Tensor

CUDA = torch.cuda.is_available()

class kmeans_core:
    def __init__(self, k, data_array, batch_size=1000, epochs=200):
        """
        kmeans by batch
        """
        self.k = k
        self.data_array = data_array
        self.dataset = self.get_ds()

        self.dim = data_array.shape[-1]
        self.data_len = data_array.shape[0]
        self.cent = Variable(Tensor(data_array[np.random.choice(range(self.data_len), k)]))
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.iters = len(self.dataloader)
        if CUDA:
            self.cent = self.cent.cuda()


    def get_ds(self):
        return TensorDataset(data_tensor=Tensor(self.data_array),
                             target_tensor=torch.zeros(self.data_array.shape[0]))

    def run(self):
        for e in range(self.epochs):
            t = trange(self.iters)
            gen = iter(self.dataloader)
            start = self.cent.clone()
            for i in t:
                dt, _ = next(gen)
                dt = Variable(dt)
                if CUDA:
                    dt = dt.cuda()
                self.step(dt)
                t.set_description("[epoch:%s\t iter:%s] \tk:%s" % (e, i, self.k))

            if self.cent.size()[0] == start.size()[0]:
                if self.cent.sum().data[0] == start.sum().data[0]:
                    print("Centeroids is not shifting anymore")
                    break
        gen = iter(self.dataloader)
        t = trange(self.iters)
        for i in t:
            dt, _ = next(gen)
            dt = Variable(dt)
            if i == 0:
                self.idx = self.calc_idx(dt)
            else:
                self.idx = torch.cat([self.idx, self.calc_idx(dt)], dim=-1)
        return self.idx

    def step(self, dt):
        idx = self.calc_idx(dt)
        self.new_c(idx, dt)

    def calc_distance(self, dt):
        bs = dt.size()[0]
        distance = torch.pow(self.cent.unsqueeze(0).repeat(bs, 1, 1) - dt.unsqueeze(1).repeat(1, self.k, 1), 2).mean(
            dim=-1)
        return distance

    def calc_idx(self, dt):
        distance = self.calc_distance(dt)
        val, idx = torch.min(distance, dim=-1)
        return idx

    def new_c(self, idx, dt):
        z = Variable(torch.zeros(self.k, self.dim))
        o = Variable(torch.zeros(self.k))
        ct = o.index_add(0, idx, Variable(torch.ones(dt.size()[0])))
        # slice used to remove zero
        slice_ = (ct > 0)

        cent_sum = z.index_add(0, idx, dt)[slice_.view(-1, 1)].view(-1, self.dim)
        ct = ct[slice_].view(-1, 1)

        self.cent = cent_sum / ct
        self.k = self.cent.size()[0]
