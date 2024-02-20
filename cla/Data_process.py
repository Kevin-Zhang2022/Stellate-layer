import matplotlib.pyplot as plt
import librosa
import os
import csv
import sys
import numpy as np
import soundfile as sf
import torch

class Data_process(object):
    def __init__(
            self,
    ):
        super().__init__()
        # self.in_feature = in_feature
        # self.out_feature = out_feature
        # self.mode = mode
        # self.out_af = out_af
    @staticmethod
    def create_datalist(path2audio):
        dirs = os.listdir(path2audio)
        path_split = path2audio.split('/')
        head = ['class_name', 'class_id']

        dir_lv2 = os.path.join(path_split[0], path_split[1]).replace('\\', '/')

        f_label = open(os.path.join(dir_lv2, 'label_list.csv'), 'w', encoding='utf-8', newline='')
        f_all = open(os.path.join(dir_lv2, 'all_list.csv'), 'w', encoding='utf-8', newline='')
        f_train = open(os.path.join(dir_lv2, 'train_list.csv'), 'w', encoding='utf-8', newline='')
        f_test = open(os.path.join(dir_lv2, 'test_list.csv'), 'w', encoding='utf-8', newline='')

        # this is label list
        writer_label = csv.writer(f_label)
        writer_label.writerow(head)
        class_dic = {}
        for dir_id, dir in enumerate(dirs):
            writer_label.writerow([dir, dir_id])
            if dir not in class_dic.keys():
                class_dic[dir] = dir_id
        f_label.close()

        # now let's do training list
        writer_all = csv.writer(f_all)
        path = path2audio
        if not os.path.exists(path):
            # print("error: folder \"", path, "\" does not exit!")
            print('error: folder \"', path, '\" does not exit!')
            sys.exit()

        writer_all.writerow(['path', 'class_id'])
        for root, dirs, files in os.walk(path):
            for file in files:
                path = os.path.join(root, file).replace('\\', '/')
                temp = path.split('/')
                class_id = class_dic[temp[-2]]
                writer_all.writerow([path, class_id])
                # print(path,'\t',class_id)
        f_all.close()

        f_all = open(os.path.join(dir_lv2, 'all_list.csv'), 'r', encoding='utf-8', newline='')
        lines = f_all.readlines()
        f_all.close()

        index = np.arange(len(lines) - 1)
        np.random.shuffle(index)

        train_index = index[0:int(0.8 * (len(lines) - 1))]
        test_index = index[int(0.8 * (len(lines) - 1)):]

        writer_train = csv.writer(f_train)
        writer_train.writerow(head)
        for i in range(len(train_index)):
            temp = lines[train_index[i] + 1].split(',')
            writer_train.writerow([temp[0], temp[1].replace('\r\n', '')])
        f_train.close()

        writer_test = csv.writer(f_test)
        writer_test.writerow(head)
        for i in range(len(test_index)):
            temp = lines[test_index[i] + 1].split(',')
            writer_test.writerow([temp[0], temp[1].replace('\r\n', '')])
        f_test.close()

    @classmethod
    def pick_peaks(cls,inp):
        dif1_0 = inp[:, 1:-1, :]-inp[:, 0:-2, :]
        dif1_2 = inp[:, 1:-1, :]-inp[:, 2:, :]
        temp = torch.zeros(inp.size(0), 1, inp.size(2))

        dif = (dif1_0>0)*(dif1_2>0)
        dif = torch.cat([temp,dif],dim=1)
        dif = torch.cat([dif,temp],dim=1)
        out = dif*inp
        return out

    @staticmethod
    def rms_amp(inp,window,stride):
        out = torch.zeros(inp.size(0),int((inp.size(1)-window)/stride),inp.size(2))
        for i in range(0,inp.size(1),stride):
            temp = torch.std_mean(inp[:,i:i+window,:])
        rms_redhot = librosa.feature.rms(inp, frame_length=10, hop_length=10)[0]
        return rms_redhot

    @staticmethod
    def ave_amp(inp, window, stride):
        # out = torch.zeros(inp.size(0), int((inp.size(1) - window) / stride), inp.size(2))
        out = []
        for i in range(0, inp.size(1), stride):
            a = torch.mean(abs(inp[:, i:i + window, :]), dim=1)
            out.append(a.unsqueeze(1))
        out = torch.cat(out, dim=1)

        # max,_ = torch.max(out,dim=1)
        # plt.plot(max[15])
        # 0,1,5,15
        return out

    @classmethod
    def env_amp(cls,inp,window,stride):
        # window = (inp.size(1)/cf).astype(int)
        # a = np.int_(inp.size(1)/cf)
        apm=[]
        a,_ = torch.max(inp[17,:,:],dim=0)
        # plt.plot(cf,a)
        plt.figure()
        plt.plot(a)

        # 17 0 8
        sf.write('stereo_file.wav', torch.sum(inp[8,:,125:200],dim=1), 10000, 'PCM_24')
        ps('stereo_file.wav')

        a = inp[[0,8,17],:,:]
        max, _ = torch.max(a,dim=1)
        mask = torch.mean(max,dim=0)
        plt.plot((mask-mask.min())/(mask.max()-mask.min()))

        torch.mean(inp[[0,8,17],:,:], dim=0)


        # ipd.audio(inp[0,:,193], rate=10000)
        # librosa.write('tone_220.wav', inp[0,:,193], 10000)

        # stride = ((inp.size(1)-window)/out_len).astype(int)
        # for j in range(0, inp.size(2)):
        #     for i in range(0, inp.size(1), stride):
        #         temp,_ = torch.max(inp[:,i:i+window[j],j], dim=1,keepdim=True)
        #         amp.append(temp)
        # peaks = cls.pick_peaks(inp)
        # plt.plot(peaks[0,:,25])

        # for b in range(inp.size)
        #     for j in range(inp.size(2)):
        #         ind = torch.where(inp[0,:,0]>0)
        #         a = inp[ind[0],ind[1],0]


        # return torch.cat(amp,dim=1)




        # a = torch.cat(amp,dim=1)
        # plt.plot(a[0,:,25])
    @classmethod
    def scale(cls,inp):
        out = inp/torch.max(abs(inp),dim=1)[0].unsqueeze(1)
        return out

    @classmethod
    def normalize(cls,inp):
        out = torch.layer_norm(inp, normalized_shape=[inp.size(1),])
        return out

    @classmethod
    def gtfb(cls,inp, band_width, channels, sam_rate):
        start_band = gt.hertz_to_erbscale(band_width[0])
        end_band = gt.hertz_to_erbscale(band_width[1])
        GTF_bank = gt.GammatoneFilterbank(samplerate=sam_rate, bandwidth_factor=0.05, order=4, startband=start_band,
                                          endband=end_band, density=(end_band-start_band)/(channels-0.05), normfreq=0)
        out = []
        for b in range(inp.shape[0]):
            # plt.plot(inp[0,:])
            results = GTF_bank.analyze(inp[b, :])
            temp = []
            # IHC_R_real = []
            for (band, state) in results:
                temp.append(np.real(band))
            temp = np.array(temp).transpose()
            out.append(temp)
        out = np.array(out)
        # plt.plot(out[0, 25, :])
        # temp[np.where(temp<=2e-5)] = 2e-5

        # out = np.array(out)
        # out1 = torch.tensor(out, dtype=torch.float32).reshape(out.shape[0], 1, out.shape[1], out.shape[2])
        # mask = out1>0
        # out2 = out1*mask

        # m = temp
        # a = np.max(m, axis=1)
        # plt.plot(m)
        # plt.show()
        # for i in range(55,60):
        #     plt.plot(m[i, :])
        # plt.plot(m[23,:]) # 58.6 107.8 149.7

        return torch.tensor(out, dtype=torch.float32), GTF_bank.centerfrequencies
    #
    # @classmethod
    # # def plot_max_byclass(cls, inp, label):
    # #

    @classmethod
    def average_ccmat(cls,inp):
        mean=[]
        for i in range(inp.shape[0]):
            temp = inp[i,:,:].detach().numpy()
            std_x = np.std(temp,axis=1)
            std_x = std_x.reshape((std_x.shape[0], 1))
            std_mat = np.matmul(std_x,std_x.transpose())
            cc_mat = np.cov(temp)/(std_mat+1e-10)
            mean.append(np.mean(cc_mat))
            # plt.imshow(cc_mat,aspect='auto')
        return np.mean(mean)
        # mean_before = np.mean(mean_before)
        # mean_after = np.mean(mean_after)
        # print(mean_before)
        # print(mean_after)




# y = [3,5,9,7,18,16,6,5,9,10]
