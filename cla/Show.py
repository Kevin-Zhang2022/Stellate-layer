import matplotlib.pyplot as plt
import numpy as np
import torch

class Show(object):
    def __init__(self):
        pass

    @classmethod
    def acc(cls,inp,label):
        # if mode in['std_snn','my_snn']:
        inp =inp[0]
        _, predicted = inp.sum(dim=1).max(1)
        # _, predicted = out.max(1)
        acc = np.mean((label == predicted).detach().numpy())
            # loss_val = loss_val.detach().numpy()[0]
        # else:
        # plt.plot(inp[0,:,8].detach().numpy())
        #     _, predicted = inp.max(1)
        #     acc = np.mean((label == predicted).detach().numpy())
        return acc

    @classmethod
    def loss_val(cls,loss,out,label):
        # if mode in ['std_snn','my_snn']:
        out = out[1]
        loss_val = torch.zeros(1)
        for t in range(out.shape[1]):
            loss_val += loss(out[:, t, :], label)
        # plt.imshow(out,aspect='auto')
        # else:
        #     loss_val = loss(out, label)
        return loss_val

    @classmethod
    def print_process(cls,i,acc,epoch,loss_val,rep=0,process=''):
        print(process, f' i:{i:d}', f"epoch: {epoch:d}", f"acc: {acc:.2f}", f"rep: {rep:d}",
              f"loss: {loss_val.item():.2f}")

