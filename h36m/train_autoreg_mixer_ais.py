import torch
import os

import sys

USER_NAME = 'a'
if USER_NAME == 'a':
    sys.path.append('/home/azhuavlev/PycharmProjects/MotionMixerConv')
elif USER_NAME == 'v':
    sys.path.append('/home/user/bornhaup/FinalProject/MotionMixerConv')
else:
    raise ValueError('User not defined')

from conv_mixer.datasets.dataset_ais_xyz import DatasetAISxyz
from h36m.utils.data_utils import define_actions
from torch.utils.data import DataLoader
from h36m.mlp_mixer import MlpMixer
from h36m.conv_mixer_model import ConvMixer
import torch.optim as optim
import numpy as np
import argparse
from h36m.utils.utils_mixer import delta_2_gt, mpjpe_error, euler_error, auc_pck_metric, joint_angle_error
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# sys.path.append('/home/azhuavlev/PycharmProjects/MotionMixerConv/conv_mixer')
from conv_mixer.utils.visualization_helpers_ais import visualize_batch_ais

from h36m import train_autoreg_mixer_h36m as train_autoreg_mixer_h36m

import time

import matplotlib as mpl

mpl.use('Agg')


def get_log_dir(out_dir):
    dirs = [x[0] for x in os.walk(out_dir)]
    if len(dirs) < 2:
        log_dir = os.path.join(out_dir, 'exp0')
        os.mkdir(log_dir)
    else:
        log_dir = os.path.join(out_dir, 'exp%i' % (len(dirs) - 1))
        os.mkdir(log_dir)

    return log_dir


def train(model, model_name, args):
    # log_dir = get_log_dir(args.root)
    log_dir = os.path.join(args.save_path, model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        time.sleep(5)
        raise ValueError('The directory already exists. Please, change the name of the model', log_dir)

    tb_writer = SummaryWriter(log_dir=log_dir)
    print('Save data of the run in: %s' % log_dir)

    # device is provided by the user in the command line
    device = args.dev

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)

    train_loss, val_loss, test_loss = [], [], []

    # different dataset classes and different joint dimensions are used for the two loss types
    if args.loss_type == 'mpjpe':

        dataset = DatasetAISxyz(
            data_dir="/home/azhuavlev/Desktop/Data/CUDA_lab/VisionLabSS23_3DPoses",
            input_n=args.input_n_dataset,
            output_n=args.output_n_dataset,
            skip_rate=args.skip_rate,
            actions=[
                '2021-08-04-singlePerson_000',
                '2021-08-04-singlePerson_001',
                # '2021-08-04-singlePerson_002',
                '2021-08-04-singlePerson_003',
                '2022-05-26_2persons_000',
                # '2022-05-26_2persons_001',
                # '2022-05-26_2persons_002',
                '2022-05-26_2persons_003'
            ],
            smoothing_alpha=0.15
        )
        vald_dataset = DatasetAISxyz(
            data_dir="/home/azhuavlev/Desktop/Data/CUDA_lab/VisionLabSS23_3DPoses",
            input_n=args.input_n_dataset,
            output_n=args.output_n_dataset,
            skip_rate=args.skip_rate,
            actions=[
                # '2021-08-04-singlePerson_000',
                # '2021-08-04-singlePerson_001',
                # '2021-08-04-singlePerson_002',
                # '2021-08-04-singlePerson_003',
                # '2022-05-26_2persons_000',
                '2022-05-26_2persons_001',
                # '2022-05-26_2persons_002',
                # '2022-05-26_2persons_003'
            ],
            smoothing_alpha=0.15
        )
        # joints_ais = [
        #     "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee",
        #     "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"
        # ]
        # ignore Nose, MidHip, RHip, LHip, REye, LEye, REar, LEar

        dim_all = np.arange(57)
        # dim_used = dim_all

        joints_to_ignore = [1, 8, 9, 12, 15, 16, 17, 18]
        dims_to_ignore = np.concatenate(
            (np.array(joints_to_ignore) * 3, np.array(joints_to_ignore) * 3 + 1, np.array(joints_to_ignore) * 3 + 2))
        dim_used = np.setdiff1d(dim_all, dims_to_ignore)

    elif args.loss_type == 'angle':
        raise NotImplementedError()

    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_worker, pin_memory=True)
    vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_worker, pin_memory=True)

    if args.loss_type == 'mpjpe':
        metrics = {'auc_pck': [], 'mpjpe': []}

    for epoch in range(args.n_epochs):
        print('Run epoch: %i' % epoch)
        running_loss = 0
        n = 0
        model.train()
        for cnt, batch in enumerate(data_loader):
            batch = batch.to(device)
            batch_dim = batch.shape[0]
            n += batch_dim

            optimizer.zero_grad()

            if epoch < args.n_epochs_teacher_forcing:
                loss, _ = train_autoreg_mixer_h36m.autoregressive_process_batch(batch, model, args, dim_used, teacher_forcing=True)
            else:
                loss, _ = train_autoreg_mixer_h36m.autoregressive_process_batch(batch, model, args, dim_used, teacher_forcing=False)

            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad)

            optimizer.step()

            running_loss += loss * batch_dim

        train_loss.append(running_loss.detach().cpu() / n)
        model.eval()
        with torch.no_grad():
            running_loss = 0
            n = 0
            for cnt, batch in enumerate(vald_loader):
                batch = batch.to(device)
                batch_dim = batch.shape[0]
                n += batch_dim

                loss, _ = train_autoreg_mixer_h36m.autoregressive_process_batch(batch, model, args, dim_used, teacher_forcing=False)

                running_loss += loss * batch_dim
            val_loss.append(running_loss.detach().cpu() / n)
        if args.use_scheduler:
            scheduler.step()

        if args.loss_type == 'mpjpe':
            test_mpjpe_loss, test_auc_pck = test_mpjpe(model, args, model_name, save_results=False)

            metrics['auc_pck'].append(test_auc_pck)
            metrics['mpjpe'].append(test_mpjpe_loss)

            test_loss.append(test_mpjpe_loss)

        tb_writer.add_scalar('loss/train', train_loss[-1].item(), epoch)
        tb_writer.add_scalar('loss/val', val_loss[-1].item(), epoch)
        tb_writer.add_scalar('loss/test', test_loss[-1].item(), epoch)

        if args.loss_type == 'mpjpe':
            tb_writer.add_scalar('metrics/auc_pck', metrics['auc_pck'][-1], epoch)
            tb_writer.add_scalar('metrics/mpjpe', metrics['mpjpe'][-1], epoch)
        elif args.loss_type == 'angle':
            tb_writer.add_scalar('metrics/joint_angle', metrics['joint_angle'][-1], epoch)
            tb_writer.add_scalar('metrics/euler_angle', metrics['euler_angle'][-1], epoch)

        torch.save(model.state_dict(), os.path.join(log_dir, 'model.pt'))

    # (Aleksei) return the losses for optuna, lists of all losses
    return train_loss, val_loss, test_loss, metrics


def test_mpjpe(model, args, model_name, save_results):
    device = args.dev
    model.eval()

    auc_pck_accum = 0
    accum_loss = 0
    n_batches = 0  # number of batches for all the sequences

    if args.actions_to_consider == 'all':
        actions = [
            '2021-08-04-singlePerson_002',
            '2022-05-26_2persons_002'
        ]
    else:
        actions = [args.actions_to_consider]

    if args.loss_type == 'mpjpe':
        dim_all = np.arange(57)
        joints_to_ignore = [1, 8, 9, 12, 15, 16, 17, 18]
        dims_to_ignore = np.concatenate(
            (np.array(joints_to_ignore) * 3, np.array(joints_to_ignore) * 3 + 1, np.array(joints_to_ignore) * 3 + 2))
        dim_used = np.setdiff1d(dim_all, dims_to_ignore)

        # dim_used = dim_all

    elif args.loss_type == 'angle':
        raise NotImplementedError()

    sequences_to_save = {10, 20, 30}
    for action in actions:

        running_loss = 0
        n = 0
        if args.loss_type == 'mpjpe':
            dataset_test = DatasetAISxyz(
                data_dir="/home/azhuavlev/Desktop/Data/CUDA_lab/VisionLabSS23_3DPoses",
                input_n=args.input_n_dataset,
                output_n=args.output_n_dataset,
                skip_rate=args.skip_rate,
                actions=[
                    action
                ],
                smoothing_alpha=0.15
            )

        test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test,
                                 shuffle=False, num_workers=0, pin_memory=True)
        for cnt, batch in enumerate(test_loader):
            with torch.no_grad():

                batch = batch.to(device)
                batch_dim = batch.shape[0]
                n += batch_dim

                sequences_gt = batch[:, args.input_n_dataset:args.input_n_dataset +
                                                     args.output_n_dataset, dim_used]

                loss, sequences_predict = train_autoreg_mixer_h36m.autoregressive_process_batch(batch, model, args, dim_used,
                                                                           teacher_forcing=False)

            auc_pck_batch = auc_pck_metric(
                sequences_predict.view(-1, args.output_n_dataset, len(dim_used) // 3, 3) ,#/ 1000,
                sequences_gt.view(-1, args.output_n_dataset, len(dim_used) // 3, 3) ,#/ 1000
            )

            auc_pck_accum += auc_pck_batch * batch_dim
            running_loss += loss * batch_dim
            accum_loss += loss * batch_dim

            if save_results and cnt in sequences_to_save:

                # to visualize full skeleton, we joints that are ignored will be copied from ground truth
                batch_predicted = batch[10, args.input_n_dataset:args.input_n_dataset + args.output_n_dataset, :].clone()
                batch_predicted[:, dim_used] = sequences_predict[10]

                os.makedirs(os.path.join(args.save_path, model_name, 'visualization'), exist_ok=True)
                visualize_batch_ais(
                    batch_full=batch_predicted.cpu(),
                    save_path=os.path.join(args.save_path, model_name, 'visualization', f'{action}_{cnt}_10.gif'),
                    batch_gt=batch[10, args.input_n_dataset:args.input_n_dataset + args.output_n_dataset, :].cpu(),
                    batch_train=batch[10, 0:args.input_n_dataset, :].cpu()
                )

                # visualize_batch_ais(
                #     batch_full=sequences_predict[10].cpu(),
                #     save_path=os.path.join(args.save_path, model_name, 'visualization', f'{action}_{cnt}_10.gif'),
                #     batch_gt=sequences_gt[10].cpu(),
                #     batch_train=sequences_train[10].cpu()
                # )

        n_batches += n
        # total_pck += auc_pck_running
    print('overall average loss in mm is: %f' % (1000 * accum_loss / n_batches))
    print('auc_pck is:', auc_pck_accum / n_batches)
    return 1000 * accum_loss / n_batches, auc_pck_accum / n_batches
