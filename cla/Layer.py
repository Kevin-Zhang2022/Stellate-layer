import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import snntorch as snn
from scipy import stats
import numpy as np
import cla.gtfb as gt



# from main.train import gp
# from main.train import sample_rate,band_width,channels

class Basic(nn.Module):
    def __init__(self, in_features, out_features):
        super(Basic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mem=0
        self.spk=0

    @staticmethod
    def LIF(I, mem, beta=0.95, uth=1):
        spk=(mem>uth).float()
        mem=beta*mem+I-spk*uth
        return spk,mem

    def reset_sm(self):
        self.mem=0
        self.spk=0

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.out_features) if self.out_features > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def reset_mask(self, scale_range):
        in_features = self.in_features
        out_features = self.out_features

        mean = np.arange(0,in_features,in_features/out_features)
        scale = np.arange(scale_range[0],scale_range[1],(scale_range[1]-scale_range[0])/out_features)

        x = np.arange(0, in_features, 1)

        mask = []
        for i in range(out_features):
            norm = stats.norm(loc=mean[i], scale=scale[i])
            y = norm.pdf(x) * (np.sqrt(2 * np.pi) * scale[i])
            mask.append(y)
        mask = np.array(mask)
        mask = torch.tensor(mask, dtype=torch.float32)
        return mask

    def set_w(self, std):
        in_features = self.in_features
        out_features = self.out_features

        x = np.arange(0, in_features, 1)
        w = []
        for i in range(out_features):
            norm = stats.norm(loc=i, scale=std)
            y = norm.pdf(x)
            w.append(y)
        w = np.array(w)
        w = torch.tensor(w, dtype=torch.float32)
        return w

    def set_w_li(self, std):
        in_features = self.in_features
        out_features = self.out_features

        x = np.arange(0, in_features, 1)
        w = []
        for i in range(out_features):
            norm = stats.norm(loc=i, scale=std)
            y = norm.pdf(x)
            y[i]=0
            w.append(y)
        w = torch.tensor(np.array(w), dtype=torch.float32)
        return w


class Cochlea(Basic):
    def __init__(self,
                 fr,
                 channels,
                 bf,
                 fs,
    ):
        super(Cochlea, self).__init__(1,channels)
        self.gtfb=gt.GTFB(1,channels,fr_hz=fr,fs=fs,bf=bf)

    def forward(self, inp):
        self.gtfb.reset()
        out = []
        for t in range(inp.size(1)):
            temp = self.gtfb(inp[:,t])
            out.append(temp)
        return torch.stack(out,dim=1)


class InnerHairCell(Basic):
    def __init__(self,
                 window,
    ):
        super(InnerHairCell, self).__init__(1,1)
        self.window = window

    def forward(self, inp):

        inp = (inp > 0) * inp

        out = []
        for head in range(0, inp.size(1), self.window):
            temp = torch.mean(inp[:, head:(head + self.window), :], dim=1)
            out.append(temp.unsqueeze(1))
        out = torch.cat(out, dim=1)
        return out


class AuditoryNerve(Basic):
    def __init__(self,
                 in_features,
                 out_features,
                 uth
    ):
        super(AuditoryNerve, self).__init__(in_features,out_features)
        self.uth=uth
        self.sleaky = snn.Leaky(beta=0.95,threshold=uth)

    def forward(self,x,mem):

        # spk, mem = self.LIF(x,mem,uth=self.uth)
        spk, mem = self.sleaky(x,mem)
        return spk,mem



class E_Linear(Basic):
    def __init__(
            self,
            in_features,
            out_features,
    ):
        super(E_Linear, self).__init__(in_features, out_features)
        self.w = Parameter(torch.empty((out_features)))
        self.reset_parameters()

    def forward(self, inp):
        out = inp * self.w
        return out


class InferiorColliculus(Basic):
    def __init__(
            self,
            in_features,
            out_features,
    ):
        super(InferiorColliculus, self).__init__(in_features, out_features)
        self.sleaky = snn.Leaky(beta=0.95)
        self.elinear = E_Linear(in_features, out_features)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, inp, mem=None):
        if mem == None:
            out = self.leaky_relu(inp + self.elinear(inp))
            return out
        else:
            spk, mem = self.sleaky(inp + self.elinear(inp), mem)
            return spk, mem


class OSF(Basic):
    def __init__(
            self,
            in_features,
            out_features,
            std,
    ):
        super(OSF, self).__init__(in_features, out_features)
        # self.sleaky = snn.Leaky(beta=0.95)
        # self.w_0 = self.reset_mask(std_0)
        self.w = self.set_w(std)
        self.sleaky = snn.Leaky(beta=0.95,threshold=0.5)
        # plt.plot(self.w_1[20,:])

    def forward(self, inp, mem):
        I = torch.matmul(self.w, inp.unsqueeze(2)).squeeze(2)
        spk, mem = self.sleaky(inp-I,mem)
        return spk,mem


class LateralInhibition(Basic):
    def __init__(
            self,
            in_features,
            out_features,
            std,
    ):
        super(LateralInhibition, self).__init__(in_features, out_features)
        self.w = self.set_w_li(std)
        self.sleaky = snn.Leaky(beta=0.95,threshold=0.5)
        # plt.plot(self.w_1[20,:])

    def forward(self, inp, mem):
        I = torch.matmul(self.w, inp.unsqueeze(-1)).squeeze(-1)
        spk, mem = self.sleaky(inp-I,mem)
        return spk, mem


class AudioCortex(Basic):
    def __init__(
            self,
            in_features,
            out_features,
    ):
        super(AudioCortex, self).__init__(in_features, out_features)
        self.sleaky = snn.Leaky(beta=0.95)
        self.flinear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, inp, mem):
        spk, mem = self.sleaky(self.flinear(inp), mem)
        return spk, mem
