import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


#@title Plotting Settings
def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max1=1.25, ylim_max2=1.25):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,6), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

    # Plot input current
    ax[0].plot(cur, c="tab:orange")
    ax[0].set_ylim([0, 0.5])
    ax[0].set_xlim([0, 200])
    ax[0].set_ylabel("$I$",fontsize=12)
    if title:
        ax[0].set_title(title)

    # Plot membrane potential
    ax[1].plot(mem)
    ax[1].set_ylim([0, ylim_max2])
    ax[1].set_ylabel("$U$",fontsize=12)
    if thr_line:
        ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[2], s=400, c="black", marker="|")
    if vline:
        ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha=0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
    # plt.ylabel("Output spikes")
    plt.ylabel("$S$",fontsize=12)
    # plt.text("$S$")
    plt.yticks([])

    plt.show()

    plt.subplots_adjust(left=0.09,top=0.98,right=0.98,bottom=0.08)

    filename = '../fig/M/M LIF/M Mechanism of LIF layer.pdf'
    fig.savefig(filename,bbox_inches='tight')

# def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title):
#   # Generate Plots
#   fig, ax = plt.subplots(3, figsize=(8,7), sharex=True,
#                         gridspec_kw = {'height_ratios': [1, 1, 0.4]})
#
#   # Plot input spikes
#   splt.raster(spk_in[:,0], ax[0], s=0.03, c="black")
#   ax[0].set_ylabel("Input Spikes")
#   ax[0].set_title(title)
#
#   # Plot hidden layer spikes
#   splt.raster(spk1_rec.reshape(num_steps, -1), ax[1], s = 0.05, c="black")
#   ax[1].set_ylabel("Hidden Layer")
#
#   # Plot output spikes
#   splt.raster(spk2_rec.reshape(num_steps, -1), ax[2], c="black", marker="|")
#   ax[2].set_ylabel("Output Spikes")
#   ax[2].set_ylim([0, 10])
#
#   plt.show()



num_steps = 200

# Initialize inputs and outputs
cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.3), 0)  # increased current
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = [mem]
spk_rec = [spk_out]

lif2 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3)

# neuron simulation
for step in range(num_steps):
  spk_out, mem = lif2(cur_in[step], mem)
  mem_rec.append(mem)
  spk_rec.append(spk_out)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)


# plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max2=1.3,
#                  title="Lapicque Neuron Model With Periodic Firing")
plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max2=1.3)


a=10