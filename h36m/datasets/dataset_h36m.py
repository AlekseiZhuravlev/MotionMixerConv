
from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
import utils.data_utils as data_utils
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import time

import os 

'''
adapted from
https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/h36motion3d.py
'''


class H36M_Dataset(Dataset):

    def __init__(self,data_dir,input_n,output_n,skip_rate, actions=None, split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir,'h3.6m/dataset')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.sample_rate = 2
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n

        subs = [[1, 6, 7, 8, 9], [11], [5]]

        # acts = data_utils.define_actions(actions)
        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = actions


        # subs = np.array([[1], [11], [5]])
        # acts = ['walking']
        # 32 human3.6 joint name:
        joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
                      "RightForeArm",
                      "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]

        subs = subs[split]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]

                # print(f'processing action {action}, time {time.time():.2f}')

                if self.split <= 1:
                    for subact in [1, 2]:  # subactions
                        #print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                        ###############################################################################
                        # read action data, convert to xyz, and store in self.p3d
                        ###############################################################################

                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                        the_sequence = data_utils.readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(the_sequence[even_list, :])
                        the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        the_sequence[:, 0:6] = 0


                        p3d = data_utils.expmap2xyz_torch(the_sequence)

                        # self.p3d[(subj, action, subact)] = p3d.view(num_frames, -1).cpu().data.numpy()

                        # full sequence
                        # self.p3d[key].shape (1738, 96)
                        self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()

                        ###############################################################################
                        # find valid frames and store in self.data_idx
                        # valid frames = indices of start frames of training subsequences
                        # (e.g. start at frame 7, with seq length 20)
                        ###############################################################################

                        # possible starting frames
                        valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)

                        # tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)

                        # list of repeated keys, e.g. [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
                        tmp_data_idx_1 = [key] * len(valid_frames)

                        # possible starting frames
                        tmp_data_idx_2 = list(valid_frames)

                        # list of tuples, e.g. [(7, 0), (7, 10), (7, 20), (7, 30), (7, 40), (7, 50), (7, 60), (7, 70), (7, 80), (7, 90)]
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        key += 1
                else:
                    #print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))

                    ###############################################################################
                    # read subaction_1 data, convert to xyz, and store in self.p3d
                    ###############################################################################

                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    the_sequence1 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    the_seq1 = torch.from_numpy(the_sequence1).float().cuda()

                    the_seq1[:, 0:6] = 0

                    p3d1 = data_utils.expmap2xyz_torch(the_seq1)
                    # self.p3d[(subj, action, 1)] = p3d1.view(num_frames1, -1).cpu().data.numpy()
                    self.p3d[key] = p3d1.view(num_frames1, -1).cpu().data.numpy()

                    ###############################################################################
                    # read subaction_2 data, convert to xyz, and store in self.p3d
                    ###############################################################################

                    #print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    the_sequence2 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    the_seq2 = torch.from_numpy(the_sequence2).float().cuda()

                    the_seq2[:, 0:6] = 0

                    p3d2 = data_utils.expmap2xyz_torch(the_seq2)

                    # self.p3d[(subj, action, 2)] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    self.p3d[key + 1] = p3d2.view(num_frames2, -1).cpu().data.numpy()

                    # print("action:{}".format(action))
                    # print("subact1:{}".format(num_frames1))
                    # print("subact2:{}".format(num_frames2))

                    ###############################################################################
                    # find valid frames and store in self.data_idx
                    ###############################################################################

                    #  creates 128 + 128 sequences of length seq_len with random starting frames
                    # fs_sel2: [[ 995  996  997 ... 1027 1028 1029]
                    #  [ 372  373  374 ...  404  405  406]
                    #  [ 344  345  346 ...  376  377  378]
                    #  ...
                    #  [ 460  461  462 ...  492  493  494]
                    #  [ 957  958  959 ...  989  990  991]
                    #  [ 464  465  466 ...  496  497  498]] (128, 35)
                    # shapes: (128, 35), (128, 35)
                    fs_sel1, fs_sel2 = data_utils.find_indices_256(num_frames1, num_frames2, seq_len,
                                                                   input_n=self.in_n)
                    # starting frames of each sequence in fs_sel1 and fs_sel2
                    valid_frames = fs_sel1[:, 0]
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                    valid_frames = fs_sel2[:, 0]
                    tmp_data_idx_1 = [key + 1] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    key += 2

        # ignore constant joints and joints at same position with other joints
        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        #print (self.p3d[key][fs].shape)
        return self.p3d[key][fs]

