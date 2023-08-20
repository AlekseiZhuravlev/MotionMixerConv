import torch
import json
import logging

class DatasetAISxyz(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_n, output_n, skip_rate, actions):
        self.data_dir = data_dir
        self.input_n = input_n
        self.output_n = output_n

        # TODO (Aleksei): skip_rate is not used
        self.skip_rate = skip_rate
        self.actions = actions

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
        # TODO (Aleksei): ask Simon about this
        person_ids = set()
        for frame in pose_data:
            if frame['person']['id'] not in person_ids:
                person_ids.add(frame['person']['id'])
        assert len(person_ids) == 1, 'More than one person in action {}'.format(action)

        # parse joint coordinates from frames
        frame_unsuccessful = []  # 0 for successful, 1 for failed
        joint_coords_full = []
        for i, frame in enumerate(pose_data):
            try:
                # all joints were successfully detected
                joint_coords_per_interval = self.process_frame(frame)
                joint_coords_full.append(joint_coords_per_interval)
                frame_unsuccessful.append(0)
            except RuntimeError:
                # some joints were not detected (score == 0)
                joint_coords_full.append(None)
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

        # save joint coordinates
        self.action_data[action] = joint_coords_full

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
        # TODO (Aleksei): ask Simon about this
        return joint_coords

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        action, start_frame, end_frame = self.sequences[item]

        joint_coords = torch.stack(self.action_data[action][start_frame:end_frame])
        return joint_coords


if __name__ == '__main__':
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
        ]
    )
    # print names of all json files in "/home/azhuavlev/Desktop/Data/CUDA_lab/VisionLabSS23_3DPoses"
    # import os
    # for file_name in os.listdir("/home/azhuavlev/Desktop/Data/CUDA_lab/VisionLabSS23_3DPoses"):
    #     if file_name.endswith(".json"):
    #         print(f'"{file_name[:-5]}",')