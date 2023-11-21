# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:49:19 2022
Code from: georgeretsi

"""
import os

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import datetime
#import pandas as pd

from os.path import isfile

def min_max_normalize(loader):

    acc_min, gyr_min, hrm_min = 1e4 * torch.ones(1,3), 1e4 * torch.ones(1,3), 1e4 * torch.ones(1,2)
    acc_max, gyr_max, hrm_max = -1e4 * torch.ones(1,3), -1e4 * torch.ones(1,3), -1e4 * torch.ones(1,2)
    for iter_idx, (acc, gyr, hrm, extra, id) in enumerate(loader):

        acc_min = torch.min(acc_min, acc.view(-1, 3).min(0)[0])
        gyr_min = torch.min(gyr_min, gyr.view(-1, 3).min(0)[0])
        hrm_min = torch.min(hrm_min, hrm.view(-1, 2).min(0)[0])

        acc_max = torch.max(acc_max, acc.view(-1, 3).max(0)[0])
        gyr_max = torch.max(gyr_max, gyr.view(-1, 3).max(0)[0])
        hrm_max = torch.max(hrm_max, hrm.view(-1, 2).max(0)[0])

    return (acc_min, acc_max), (gyr_min, gyr_max), (hrm_min, hrm_max)

class ELoader(Dataset):

    def __init__(self, path_db, set='all', use_interval=-1,augmentation=False, skipsleep=False, minutes=10, separate=False, age=False, gender=False):

        self.minutes = minutes
        self.age = age
        self.gender = gender
        self.aug = augmentation
        self.skipsleep = skipsleep
        self.use_interval = use_interval

        if age:
            age_info = np.loadtxt('ages.txt', dtype=str)
            self.age_dict = {a[0]:float(a[-1]) for a in age_info}
        if gender:
            age_info = np.loadtxt('ages.txt', dtype=str)
            gd = {'m': 1, 'f':0}
            self.gender_dict = {a[0]:gd[a[1]] for a in age_info}

        if separate:
            separate_sets(path_db, r=0.75)
        if set == 'train':
            set_file = os.path.join(path_db, 'train.txt')
            #if not isfile(set_file):
            valid_set = np.loadtxt(set_file, dtype=str)
        elif set == 'test':
            valid_set = np.loadtxt(os.path.join(path_db, 'test.txt'), dtype=str)
        elif set == 'all':
            valid_set = None
        else:
            print('problem in loader')
            return


        #(acc_min, acc_max), (gyr_min, gyr_max), (hrm_min, hrm_max) = min_max_normalize(train_loader)
        #acc_min, acc_max, gyr_min, gyr_max, hrm_min, hrm_max = -19.6*torch.ones(3), 19.6*torch.ones(3), -573.0*torch.ones(3), 573.0*torch.ones(3), torch.zeros(2), torch.tensor([240.0, 2000.0])
        #acc_min, acc_max = acc_min.view(1, 3), acc_max.view( 1, 3)
        #gyr_min, gyr_max = gyr_min.view(1, 3), gyr_max.view(1, 3)
        #hrm_min, hrm_max = hrm_min.view(1, 2), hrm_max.view(1, 2)


        if valid_set is None:
            files = []
            cnt=0
            for user in os.listdir(path_db):
                for name in os.listdir(os.path.join(path_db, user)):
                    files += [(os.path.join(path_db, user, name), cnt, user)]
                cnt += 1
        else:
            files = [(f[0], int(f[1]), f[2]) for f in valid_set]
            cnt = max([f[1] for f in files]) + 1

        #files = [(f[0], int(f[1]), f[2]) for f in valid_set]
        #cnt = max([f[1] for f in files]) + 1

        self.nclasses = cnt
        self.files = files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):

        data = np.load(self.files[index][0], allow_pickle=True).item()

        # !!!!!!!!!!!!!!
        #if self.skipsleep:
        #    while data["sleep"] > 0:
        #        data = np.load(self.files[np.random.randint(0, self.__len__())][0], allow_pickle=True).item()

        #[{"acc":acc, "gyr":gyr, "hrm":hrm, "sleep":sleep, "step":step, "interval":(t, t+tinterval), "id":user}]
        acc, gyr, hrm = torch.zeros(self.minutes * 60 * 20, 3), torch.zeros(self.minutes * 60 * 20, 3), torch.zeros(self.minutes * 60 * 5, 2)

        #try:


        tacc = torch.clamp(torch.from_numpy(np.asarray(data["acc"], dtype=np.float)), -19.6, 19.6) / (2 * 19.6)
        if tacc.size(0) > acc.size(0):
            acc = tacc[:acc.size(0)]
        else:
            ts = int(.5 * (acc.size(0) - tacc.size(0)))
            acc[ts: ts+tacc.size(0)] = tacc

        tgyr = torch.clamp(torch.from_numpy(np.asarray(data["gyr"], dtype=np.float)), -573.0, 573.0) / (2 * 573.0)
        if tgyr.size(0) > gyr.size(0):
            gyr = tgyr[:gyr.size(0)]
        else:
            ts = int(.5 * (gyr.size(0) - tgyr.size(0)))
            gyr[ts: ts+tgyr.size(0)] = tgyr

        thrm = torch.from_numpy(np.asarray(data["hrm"], dtype=np.float))
        thrm[:,0] = torch.clamp(thrm[:,0], 0.0, 240.0) / 240.0
        thrm[:,1] = torch.clamp(thrm[:,1], 0.0, 2000.0) / 2000.0
        if thrm.size(0) > hrm.size(0):
            hrm = thrm[:hrm.size(0)]
        else:
            ts = int(.5 * (hrm.size(0) - thrm.size(0)))
            hrm[ts: ts+thrm.size(0)] = thrm
        #except:
        #    print(self.files[index][0])

        if self.use_interval > 0:
            tstart = np.random.randint(0, acc.size(0) - self.use_interval * 20)
            tend = tstart + self.use_interval * 20
            acc = acc[tstart:tend]
            gyr = gyr[tstart:tend]
            hrm = hrm[tstart//4:tend//4]
        
        # freq aug
        if self.aug:
            for i in range(acc.size(1)):
                acc[:, i] = torch.from_numpy(freq_augmentation(acc[:, i].squeeze().numpy(), 600))
            acc = acc * (np.random.uniform() > .10)
            for i in range(gyr.size(1)):
                gyr[:, i] = torch.from_numpy(freq_augmentation(gyr[:, i].squeeze().numpy(), 600))
            gyr = gyr * (np.random.uniform() > .10)
            #for i in range(hrm.size(1)):
            #    hrm[:, i] = torch.from_numpy(freq_augmentation(hrm[:, i].squeeze().numpy(), 200))
            hrm = hrm * (np.random.uniform() > .01)


        extra = (data["sleep"], data["step"], data["interval"][0].strftime("%m/%d/%Y, %H:%M:%S"))
        #target = data["id"]
        if self.age:
            target = torch.Tensor([self.age_dict[self.files[index][2]]]).float()
        elif self.gender:
            target = self.gender_dict[self.files[index][2]]
        else:
            target = self.files[index][1]

        #return spectogram(acc.float().permute(1,0)), spectogram(gyr.float().permute(1,0)), spectogram_hrm(hrm.float().permute(1,0)), extra, target
        return acc.float(), gyr.float(), hrm.float(), extra, target


def freq_augmentation(x, window_size):

    f, t, Zxx = scipy.signal.stft(x, nperseg=window_size)


    mgn, phase = np.abs(Zxx), np.angle(Zxx)
    #mx = mgn.max()
    mx = .1

    bb = np.random.randint(50, 200)
    mgn *= 1 + .2 * (np.random.uniform(size=mgn.shape[:]) > .75) * np.random.normal(size=mgn.shape[:])
    sm = mgn[mgn < mx / bb]
    mgn[mgn < mx / bb] *= (np.random.uniform(size=sm.shape[:]) > .85) * np.random.uniform(size=sm.shape[:])
    mgn += mgn.max() / 100 * (np.random.uniform(size=mgn.shape[:]) > .75) * np.random.normal(size=mgn.shape[:])


    Zxx = mgn * np.exp(1j*phase)

    _, xrec = scipy.signal.istft(Zxx)

    return xrec