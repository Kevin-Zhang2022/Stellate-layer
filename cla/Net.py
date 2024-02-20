import matplotlib.pyplot as plt
import torch.nn as nn
from cla.Layer import AuditoryNerve,InferiorColliculus,AudioCortex,Cochlea,InnerHairCell,OSF

import torch
from scipy import stats
import math
import numpy as np


class Net(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.cochlea = Cochlea(channels=in_features, fs=kwargs['gtf_fs'],fr=kwargs['gtf_fr'],bf=kwargs['gtf_bf'])
        self.ihc = InnerHairCell(window=kwargs['ihc_win'])
        self.an = AuditoryNerve(in_features, in_features,uth=kwargs['an_uth'])
        self.osf_b = OSF(in_features,in_features,std=kwargs['steli_std'][0])
        self.osf_m = OSF(in_features,in_features,std=kwargs['steli_std'][1])
        self.osf_n = OSF(in_features,in_features,std=kwargs['steli_std'][2])


        self.ic = InferiorColliculus(in_features,in_features)
        self.ac = AudioCortex(in_features,out_features)
        self.snr_an=0
        self.snr_osf_b=0
        self.snr_osf_m = 0
        self.snr_osf_n = 0


    def ini_sm(self,inp):
        return torch.zeros_like(inp[:,0,:],dtype=torch.float32)


    def forward(self, inp):
        spk_ac_rec = []
        mem_ac_rec = []
        spk_an_rec=[]
        spk_osf_b_rec=[]  # off
        spk_osf_m_rec=[]  # off
        spk_osf_n_rec=[]  # off


        # self.reset_sm()

        inp = self.cochlea(inp)
        inp = self.ihc(inp)

        # mem_an= self.ini_sm(inp)
        mem_an = self.an.sleaky.init_leaky()
        mem_osf_b = self.osf_b.sleaky.init_leaky()  # off
        mem_osf_m = self.osf_m.sleaky.init_leaky()  # off
        mem_osf_n = self.osf_n.sleaky.init_leaky()  # off
        mem_ic = self.ic.sleaky.init_leaky()
        mem_ac = self.ac.sleaky.init_leaky()

        for t in range(inp.size(1)):
            spk_an, mem_an = self.an(inp[:, t, :],mem_an)

            spk_osf_b, mem_osf_b = self.osf_b(spk_an,mem_osf_b)  # off
            spk_osf_m, mem_osf_m = self.osf_m(spk_osf_b,mem_osf_m)  # off
            spk_osf_n, mem_osf_n = self.osf_n(spk_osf_m,mem_osf_n)  # off


            spk_ic, mem_ic = self.ic(spk_osf_n,mem_ic)  # off
            spk_ac, mem_ac = self.ac(spk_ic,mem_ac)

            spk_an_rec.append(spk_an)
            spk_osf_b_rec.append(spk_osf_b)  # off
            spk_osf_m_rec.append(spk_osf_m)  # off
            spk_osf_n_rec.append(spk_osf_n)  # off


            spk_ac_rec.append(spk_ac)
            mem_ac_rec.append(mem_ac)
        self.snr_an = self.get_snr(spk_an_rec)
        self.snr_osf_b = self.get_snr(spk_osf_b_rec)  # off
        self.snr_osf_m = self.get_snr(spk_osf_m_rec)  # off
        self.snr_osf_n = self.get_snr(spk_osf_n_rec)  # off

        out = (torch.stack(spk_ac_rec, dim=1), torch.stack(mem_ac_rec, dim=1))

        ########## do not delete!!!!!!!!!!!!!!   make figure for paper
        # fig=0
        # fontsize=8
        # spk_an_spe = torch.sum(torch.stack(spk_an_rec,dim=1),dim=1)[fig].detach().numpy()[30:50]
        # spk_osf_n_spe = torch.sum(torch.stack(spk_osf_n_rec, dim=1), dim=1)[fig].detach().numpy()[30:50]
        # fig = plt.figure(figsize=(8, 6))
        # fig.show()
        # axe = fig.add_subplot(1, 1, 1)
        # axe.plot(spk_an_spe,label='Without stellate \n$K_a$=1.06',linewidth=1.5,color='r')
        # axe.plot(spk_osf_n_spe,label='With stellate \n$K_b$=2.96',linewidth=1.5,color='b')
        # axe.set_xticks(ticks=np.arange(0,20,2.5),labels=np.arange(164,304,17.5))
        # axe.set_xlabel('Frequency(Hz)',fontsize=fontsize*1.5)  # 164-303
        # axe.set_ylabel('Spike Count',fontsize=fontsize*1.5)  # 164-303
        # axe.grid('on')
        # axe.legend(loc='upper right', fontsize=fontsize * 1.5)
        # fig.subplots_adjust(left=0.086, bottom=0.088, right=0.98, top=0.98, wspace=0.27, hspace=0.4)
        # fig.show()
        # filename = 'fig/E/E Correlation Analysis/E Kurtosis Change/E Kurtosis Change.pdf'
        # fig.savefig(filename, bbox_inches='tight')
        ########## do not delete!!!!!!!!!!!!!!   make figure for paper
        return out

    @staticmethod
    def get_snr(inp):

        inp = torch.sum(torch.stack(inp, dim=1),dim=1)

        k_lis = []
        for b in range(inp.size(0)):
            inp_b = inp[b].detach().numpy()
            win=int(0.1*inp_b.shape[0])
            for t in range(0,inp_b.shape[0],win):
                k = stats.kurtosis(inp_b[t:t+win])
                if math.isnan(k):
                    pass
                else:
                    k_lis.append(k)
        return torch.mean(torch.tensor(k_lis))
        # snr_list = []
        # for b in range(inp.size(0)):
        #     inp_b = inp[b]
        #     inp_b_mean = torch.mean(inp_b)
        #
        #     ind_larger = torch.where(inp_b>=inp_b_mean)
        #     ind_smaller = torch.where(inp_b<inp_b_mean)
        #     if len(ind_larger[0]) and len(ind_smaller[0]):
        #         sigal = torch.sum(inp_b[ind_larger])
        #         noise = torch.sum(inp_b[ind_smaller])
        #         snr_list.append(sigal/(noise+1))
        #     else:
        #         snr_list.append(torch.tensor(0,dtype=torch.float32))
        # return torch.mean(torch.tensor(snr_list))











