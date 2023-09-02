import torch
import json
import logging
import pandas as pd
import numpy as np

class DatasetAISxyz(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_n, output_n, skip_rate, actions, smoothing_alpha):
        self.data_dir = data_dir
        self.input_n = input_n
        self.output_n = output_n

        # TODO (Aleksei): skip_rate is not used
        self.skip_rate = skip_rate
        self.actions = actions
        self.smoothing_alpha = smoothing_alpha

        self.sequences = [] # (action_id, starting frame, ending frame)

        self.action_data = dict()
        for action in actions:
            self.read_action(action)

    def read_action(self, action):
        print('*********************************')
        print('Reading action {}'.format(action))

        # read the json file
        in_file_name = self.data_dir + '/' + action + '.json'
        with open(in_file_name, 'r') as f:
            pose_data = json.load(f)

        # check that there is only one person id in the action
        person_ids = set()
        for frame in pose_data:
            if frame['person']['id'] not in person_ids:
                person_ids.add(frame['person']['id'])
        assert len(person_ids) == 1, 'More than one person in action {}'.format(action)

        # parse joint coordinates from frames
        frame_unsuccessful = []  # 0 for successful, 1 for failed
        joint_coords_full = []
        for i, frame in enumerate(pose_data[::self.skip_rate]):
            try:
                # all joints were successfully detected
                joint_coords_per_interval = self.process_frame(frame)
                joint_coords_full.append(joint_coords_per_interval)
                frame_unsuccessful.append(0)
            except RuntimeError as e:
                # some joints were not detected (score == 0)
                print(f'Failed to process frame {i} in action {action}: {e}')
                joint_coords_full.append(
                    np.full([19], np.nan)
                )
                frame_unsuccessful.append(1)
                continue

        # print failed frames
        print(f'Total n of frames', len(pose_data))
        print(f'Failed to process {sum(frame_unsuccessful)} frames')
        print('indices of failed frames:', end=' ')
        for i, frame in enumerate(frame_unsuccessful):
            if frame == 1:
                print(i, end=', ')
        print()

        joint_coords_smoothed = self.apply_smoothing(joint_coords_full)
        # save joint coordinates
        self.action_data[action] = joint_coords_smoothed

        # generate sequences
        self.generate_sequences(action, joint_coords_full, frame_unsuccessful)

    def generate_sequences(self, action, joint_coords_full, frame_unsuccessful):
        for i in range(len(joint_coords_full) - self.input_n - self.output_n):

            # check that all frames in the sequence are successful
            if sum(frame_unsuccessful[i:i + self.input_n + self.output_n]) != 0:
                continue
            else:
                # (action_id, starting frame, ending frame)
                self.sequences.append((action, i, i + self.input_n + self.output_n))
        print('Current n of sequences:', len(self.sequences))

    def process_frame(self, frame):
        NUM_KPS_USED = 19

        # check that there are 21 or 27 keypoints
        if len(frame['person']['keypoints']) not in {21, 27}:
            print('Expected 21 or 27 keypoints, got {}'.format(len(frame['person']['keypoints'])))
        assert len(frame['person']['keypoints']) == 21 or len(frame['person']['keypoints']) == 27, 'Expected 21 keypoints, got {}'.format(len(frame['person']['keypoints']))

        joint_coords = []
        for kp_idx in range(NUM_KPS_USED):
            kp = frame['person']['keypoints'][kp_idx]

            # check for failed keypoints, this frame will be skipped
            if kp['score'] == 0:
                raise RuntimeError(f'Keypoint {kp_idx} has score 0')

            # convert coordinates to other coordinate system
            joint_pos_processed = self.process_joint_pos(kp['pos'])
            joint_coords.append(joint_pos_processed)

        joint_coords = torch.tensor(joint_coords)

        # print('joint_coords.shape', joint_coords.shape)
        joint_coords = self.remove_global_rot_transl(joint_coords)
        joint_coords = joint_coords.flatten()
        # print('joint_coords.shape', joint_coords.shape)
        # exit()


        return joint_coords

    def process_joint_pos(self, joint_pos):
        # TODO (Aleksei): ask Simon about this
        return joint_pos

    def remove_global_rot_transl(self, joint_coords):

        root_joint = joint_coords[8] # MidHip
        neck_joint = joint_coords[1] # Neck
        lhip_joint = joint_coords[12] # LHip
        rhip_joint = joint_coords[9] # RHip

        dir_up = (neck_joint - root_joint) / torch.norm(neck_joint - root_joint)
        dir_right = (rhip_joint - lhip_joint) / torch.norm(rhip_joint - lhip_joint)
        dir_forward = torch.cross(dir_up, dir_right) / torch.norm(torch.cross(dir_up, dir_right))

        # dir_right and dir_forward may not be orthogonal,
        # but since they lie in the same plane, we can make them orthogonal
        dir_right = torch.cross(dir_forward, dir_up) / torch.norm(torch.cross(dir_forward, dir_up))

        # assert that the coordinate axes are orthogonal
        assert torch.abs(torch.dot(dir_up, dir_right)) < 1e-6
        assert torch.abs(torch.dot(dir_up, dir_forward)) < 1e-6
        assert torch.abs(torch.dot(dir_right, dir_forward)) < 1e-6

        new_basis_mat = torch.stack([dir_right, dir_forward, dir_up])
        new_basis_mat_inv = torch.inverse(new_basis_mat)

        # remove global translation
        joint_coords_local = joint_coords - root_joint

        # remove global rotation
        joint_coords_local_rotated = torch.matmul(new_basis_mat, joint_coords_local.transpose(0, 1)).transpose(0, 1)
        return joint_coords_local_rotated


    def apply_smoothing(self, joint_coords_full):
        """
        Applies exponential smoothing to joint coordinates
        """

        # create pandas dataframe
        df = pd.DataFrame(joint_coords_full).astype(np.float32)

        # apply smoothing
        df_smooth = df.ewm(alpha=self.smoothing_alpha, axis=0, ignore_na=False).mean()

        # return df as tensor
        return torch.tensor(df_smooth.values)


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        action, start_frame, end_frame = self.sequences[item]

        # joint_coords = torch.stack(self.action_data[action][start_frame:end_frame])
        joint_coords = self.action_data[action][start_frame:end_frame]
        return joint_coords


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--integers', default=1, type=int, nargs='+',
                        help='an integer for the accumulator')

    args = parser.parse_args()

    # add argument data_dir to args
    args.data_dir = "/home/azhuavlev/Desktop/Data/CUDA_lab/VisionLabSS23_3DPoses"

    print(args)
    exit(0)








    dataset = DatasetAISxyz(
        data_dir="/home/azhuavlev/Desktop/Data/CUDA_lab/VisionLabSS23_3DPoses",
        input_n=10,
        output_n=10,
        skip_rate=2,
        actions=[
            "2021-08-04-singlePerson_003",
            "2021-08-04-singlePerson_001",
            "2022-05-26_2persons_002",
            "2021-08-04-singlePerson_000",
            "2022-05-26_2persons_000",
            "2022-05-26_2persons_003",
            "2022-05-26_2persons_001",
            "2021-08-04-singlePerson_002",
        ],
        smoothing_alpha=0.2
    )
    # print names of all json files in "/home/azhuavlev/Desktop/Data/CUDA_lab/VisionLabSS23_3DPoses"
    # import os
    # for file_name in os.listdir("/home/azhuavlev/Desktop/Data/CUDA_lab/VisionLabSS23_3DPoses"):
    #     if file_name.endswith(".json"):
    #         print(f'"{file_name[:-5]}",')

    # # plot joint_coords as 3d plot on one subplot, and direction vectors on another subplot
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure(figsize=(10, 10))
    #
    # ax = fig.add_subplot(121, projection='3d')
    # # ax.set_box_aspect([1,1,1])
    # ax.scatter(joint_coords_local[:, 0], joint_coords_local[:, 1], joint_coords_local[:, 2])
    #
    # KPS_PARENT = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16]
    #
    # xs = joint_coords_local[:, 0]
    # ys = joint_coords_local[:, 1]
    # zs = joint_coords_local[:, 2]
    #
    # print(joint_coords_local.shape)
    #
    # j = 0
    # # ax.scatter(xs, ys,
    # #            zs)
    # # print(xs)
    #
    # for kp_idx in range(1, len(xs)):
    #     # print(kp_idx)
    #     lxs = torch.stack([xs[KPS_PARENT[kp_idx]], xs[kp_idx]])
    #     lys = torch.stack([ys[KPS_PARENT[kp_idx]], ys[kp_idx]])
    #     lzs = torch.stack([zs[KPS_PARENT[kp_idx]], zs[kp_idx]])
    #
    #     ax.plot(lxs, lys, lzs)
    #
    # ax.quiver(0, 0, 0, dir_right[0], dir_right[1], dir_right[2], color='r')
    # ax.quiver(0, 0, 0, dir_up[0], dir_up[1], dir_up[2], color='g')
    # ax.quiver(0, 0, 0, dir_forward[0], dir_forward[1], dir_forward[2],
    #           color='b')
    #
    # # set xyz axes to be equal
    # ax.set_xlim3d([-1, 1])
    # ax.set_ylim3d([-1, 1])
    # ax.set_zlim3d([-1, 1])
    #
    # #########################################################################
    #
    # ax = fig.add_subplot(122, projection='3d')
    # # ax.set_box_aspect([1,1,1])
    # ax.scatter(joint_coords_local_rotated[:, 0], joint_coords_local_rotated[:, 1], joint_coords_local_rotated[:, 2])
    #
    # KPS_PARENT = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16]
    #
    # xs = joint_coords_local_rotated[:, 0]
    # ys = joint_coords_local_rotated[:, 1]
    # zs = joint_coords_local_rotated[:, 2]
    #
    # for kp_idx in range(1, len(xs)):
    #     # print(kp_idx)
    #     lxs = torch.stack([xs[KPS_PARENT[kp_idx]], xs[kp_idx]])
    #     lys = torch.stack([ys[KPS_PARENT[kp_idx]], ys[kp_idx]])
    #     lzs = torch.stack([zs[KPS_PARENT[kp_idx]], zs[kp_idx]])
    #
    #     ax.plot(lxs, lys, lzs)
    #
    # # set xyz axes to be equal
    # ax.set_xlim3d([-1, 1])
    # ax.set_ylim3d([-1, 1])
    # ax.set_zlim3d([-1, 1])
    #
    # plt.show()
    #
    #
    # exit(0)