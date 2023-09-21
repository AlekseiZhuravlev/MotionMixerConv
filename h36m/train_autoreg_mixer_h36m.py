
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

from h36m.datasets.dataset_h36m import H36M_Dataset
from h36m.datasets.dataset_h36m_ang import H36M_Dataset_Angle
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
from conv_mixer.utils.visualization_helpers_h3m import visualize_batch

import time
import shutil

import matplotlib as mpl
mpl.use('Agg')


def get_log_dir(out_dir):
    dirs = [x[0] for x in os.walk(out_dir)]
    if len(dirs ) < 2:
        log_dir = os.path.join(out_dir, 'exp0')
        os.mkdir(log_dir)
    else:
        log_dir = os.path.join(out_dir, 'exp%i'%(len(dirs)-1))
        os.mkdir(log_dir)

    return log_dir


def train_autoregressive(model, model_name, args):

    # log_dir = get_log_dir(args.root)
    log_dir = os.path.join(args.save_path, model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        # shutil.rmtree(log_dir)
        # os.makedirs(log_dir)
        time.sleep(5)
        raise ValueError('The directory already exists. Please, change the name of the model', log_dir)

    tb_writer = SummaryWriter(log_dir=log_dir)
    print('Save data of the run in: %s'%log_dir)

    # device is provided by the user in the command line
    device = args.dev

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)

    train_loss, val_loss, test_loss = [], [], []

    # different dataset classes and different joint dimensions are used for the two loss types
    if args.loss_type == 'mpjpe':
        dataset = H36M_Dataset(args.data_dir, args.input_n_dataset,
                        args.output_n_dataset, args.skip_rate, split=0)
        vald_dataset = H36M_Dataset(args.data_dir, args.input_n_dataset,
                            args.output_n_dataset, args.skip_rate, split=1)
        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

    elif args.loss_type == 'angle':
        # NOTE: for the angle loss, 48 dimensions are used
        dataset = H36M_Dataset_Angle(args.data_dir, args.input_n_dataset, args.output_n_dataset, 
                                    args.skip_rate, split=0)
        vald_dataset = H36M_Dataset_Angle(args.data_dir, args.input_n_dataset, 
                                    args.output_n_dataset, args.skip_rate, split=1)
        dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                        43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                        86])

    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_worker, pin_memory=True)
    vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_worker, pin_memory=True)

    if args.loss_type == 'mpjpe':
        metrics = {'auc_pck': [], 'mpjpe': []}
    elif args.loss_type == 'angle':
        metrics = {'joint_angle': [], 'euler_angle': []}
    
    for epoch in range(args.n_epochs):
        # assert that args.output_n_dataset is "big enough"
        print('Run epoch: %i'%epoch)
        running_loss = 0
        n = 0
        model.train()
        for cnt, batch in enumerate(tqdm(data_loader)):
            batch = batch.to(device)
            batch_dim = batch.shape[0]
            n += batch_dim

            optimizer.zero_grad()

            if epoch < args.n_epochs_teacher_forcing:
                loss, _ = autoregressive_process_batch(batch, model, args, dim_used, teacher_forcing=True)
            else:
                loss, _ = autoregressive_process_batch(batch, model, args, dim_used, teacher_forcing=False)

            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad)

            optimizer.step()

            running_loss += loss*batch_dim

        train_loss.append(running_loss.detach().cpu()/n)

        ##############################################################
        # Validation
        ##############################################################
        model.eval()

        print(">>>>>>>>>>> Validation <<<<<<<<<<<<")

        with torch.no_grad():
            running_loss = 0
            n = 0
            for cnt, batch in enumerate(vald_loader):
                batch = batch.to(device)
                batch_dim = batch.shape[0]
                n += batch_dim

                loss, _ = autoregressive_process_batch(batch, model, args, dim_used, teacher_forcing=False)
                running_loss += loss*batch_dim

                # break

            val_loss.append(running_loss.detach().cpu()/n)


        if args.use_scheduler:
            scheduler.step()

        if args.loss_type == 'mpjpe':
            test_mpjpe_loss, test_auc_pck = test_mpjpe_autoregressive(model, args, model_name, save_results=False)

            metrics['auc_pck'].append(test_auc_pck)
            metrics['mpjpe'].append(test_mpjpe_loss)

            test_loss.append(test_mpjpe_loss)
        elif args.loss_type == 'angle':
            test_euler_angle_loss, test_joint_angle_loss = test_angle_autoregressive(model, args)
            test_loss.append(test_euler_angle_loss)

            metrics['joint_angle'].append(test_joint_angle_loss)
            metrics['euler_angle'].append(test_euler_angle_loss)

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


def autoregressive_process_batch(batch, model, args, dim_used, teacher_forcing):
    # assert that args.output_n_dataset divides by args.step_window and is >= 1
    assert args.output_n_dataset % args.step_window == 0, "output_n_dataset does not divide by step_window"
    assert args.output_n_dataset // args.step_window >= 1, "output_n_dataset is smaller than step_window"

    # select loss function
    if args.loss_type == 'mpjpe':
        loss_fct = lambda pred, gt, out_n: mpjpe_error(pred, gt)
    elif args.loss_type == 'angle':
        loss_fct = lambda pred, gt, out_n: torch.mean(
            torch.sum(torch.abs(pred.reshape(-1, out_n, len(dim_used)) - gt), dim=2).view(-1))

    # select only used pose dimensions
    full_sequence = batch[:, :args.input_n_dataset + args.output_n_dataset, dim_used].clone()
    full_sequence_gt = batch[:, args.input_n_dataset:args.input_n_dataset + args.output_n_dataset, dim_used].clone()
    full_sequence_predict = torch.zeros_like(full_sequence_gt)

    # starting subsequence is the input data
    subsequence_train = full_sequence[:, 0: args.input_n_model, :]

    loss = torch.zeros(1, device=batch.device)

    # iterative prediction
    for start_frame_train in range(
            0,
            args.input_n_dataset + args.output_n_dataset - args.input_n_model - args.output_n_model + 1,
            args.step_window
    ):
        end_frame_train = start_frame_train + args.input_n_model
        end_frame_predict = end_frame_train + args.output_n_model

        if teacher_forcing:
            subsequence_train = full_sequence[:, start_frame_train: end_frame_train, :]

        # get the subsequence to train on and the gt
        subsequence_gt = full_sequence[:, end_frame_train: end_frame_predict, :]

        # predict next args.output_n_model steps
        subsequence_predict = model(subsequence_train)

        loss += loss_fct(subsequence_predict, subsequence_gt, args.output_n_model)

        # record the prediction
        full_sequence_predict[:, end_frame_train - args.input_n_model: end_frame_predict - args.input_n_model,
        :] = subsequence_predict.clone()

        if not teacher_forcing:
            # take last frames of input for the next iteration
            frames_to_take = args.input_n_model - args.step_window
            subsequence_reused = subsequence_train[:, -frames_to_take:, :]

            # print('subsequence_train', subsequence_train, subsequence_train.shape)
            # print('subsequence_reused', subsequence_reused, subsequence_reused.shape)
            # print('subsequence_predict', subsequence_predict, subsequence_predict.shape)
            # print('subsequence_gt', subsequence_gt, subsequence_gt.shape)


            # add the predicted subsequence to input for the next iteration
            subsequence_train = torch.cat((subsequence_reused, subsequence_predict), dim=1)

    # assert that loss is not nan
    assert not torch.isnan(loss), 'Loss is nan'

    return loss / (args.output_n_dataset // args.step_window), full_sequence_predict


def test_mpjpe_autoregressive(model, args, model_name, save_results):


    print('>>>>>>>>>>> Testing <<<<<<<<<<<<')

    device = args.dev
    model.eval()

    auc_pck_accum = 0
    accum_loss = 0
    n_batches = 0  # number of batches for all the sequences
    actions = define_actions(args.actions_to_consider)
    if args.loss_type == 'mpjpe':
        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                            26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                            46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                            75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    elif args.loss_type == 'angle':
        dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 
                            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 
                            56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86])
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate(
        (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate(
        (joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))


    sequences_to_save = {1, 3}

    for action in actions:

        running_loss = 0
        n = 0
        if args.loss_type == 'mpjpe':
            dataset_test = H36M_Dataset(args.data_dir, args.input_n_dataset,
                                    args.output_n_dataset, args.skip_rate, split=2, actions=[action])
        elif args.loss_type == 'angle':
            dataset_test = H36M_Dataset_Angle(args.data_dir, args.input_n_dataset,
                                    args.output_n_dataset, args.skip_rate, split=2, actions=[action])

        test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test,
                                 shuffle=False, num_workers=0, pin_memory=True)
        for cnt, batch in enumerate(test_loader):
            with torch.no_grad():
                batch = batch.to(device)
                batch_dim = batch.shape[0]
                n += batch_dim

                # create full sequences (without removing joints)
                all_joints_input = batch[:, :args.input_n_dataset, :].clone()
                all_joints_seq_gt = batch[:, args.input_n_dataset:args.input_n_dataset+args.output_n_dataset, :].clone()
                all_joints_seq = all_joints_seq_gt.clone()

                # ground truth with removed joints
                full_sequence_gt = all_joints_seq_gt[:, :, dim_used]

                # autoregressive
                loss, full_sequence_predict = autoregressive_process_batch(batch, model, args, dim_used, teacher_forcing=False)

                # insert back skipped joints to prediction
                all_joints_seq[:, :, dim_used] = full_sequence_predict

                # auc_pck
                auc_pck_batch = auc_pck_metric(
                    full_sequence_predict.view(-1, args.output_n_dataset, len(dim_used) // 3, 3) / 1000,
                    full_sequence_gt.view(-1, args.output_n_dataset, len(dim_used) // 3, 3) / 1000
                )

                # accumulate metrics
                auc_pck_accum += auc_pck_batch*batch_dim
                running_loss += loss * batch_dim
                accum_loss += loss * batch_dim


                # print('all_joints_seq', all_joints_seq)
                # print('all_joints_seq_gt', all_joints_seq_gt)
                # print('all_joints_input', all_joints_input)
                # exit(0)

                if save_results and cnt in sequences_to_save:
                    os.makedirs(os.path.join(args.save_path, model_name, 'visualization'), exist_ok=True)
                    visualize_batch(
                        batch_full=all_joints_seq[10].cpu(),
                        save_path=os.path.join(args.save_path, model_name, 'visualization', f'{action}_{cnt}_10.gif'),
                        batch_gt=all_joints_seq_gt[10].cpu(),
                        batch_train=all_joints_input[10].cpu()
                    )

        n_batches += n
        # total_pck += auc_pck_running
    print('overall average loss in mm is: %f'%(accum_loss/n_batches))
    print('auc_pck is:', auc_pck_accum/n_batches)

    return accum_loss/n_batches, auc_pck_accum/n_batches


def test_angle_autoregressive(model, args):

    device = args.dev
    model.eval()

    joint_angle_accum = 0
    accum_loss=0  
    n_batches=0 # number of batches for all the sequences
    actions=define_actions(args.actions_to_consider)
    dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                            43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                            86])

    for action in actions:

        # joint_angle_error_running = 0
        running_loss=0
        n=0
        dataset_test = H36M_Dataset_Angle(args.data_dir,args.input_n_dataset,args.output_n_dataset,args.skip_rate, split=2,actions=[action])

        # dataset_test = H36M_Dataset_Angle(args.data_dir,args.input_n_dataset,
        # args.output_n_frames_dataset,
        # args.skip_rate, split=2,actions=[action])
        #print('>>> Test dataset length: {:d}'.format(dataset_test.__len__()))

        test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
        for cnt, batch in enumerate(test_loader):
            with torch.no_grad():

                batch=batch.to(device)
                batch_dim=batch.shape[0]
                n+=batch_dim

                # create full sequences (without removing joints)
                all_joints_input = batch[:, :args.input_n_dataset, :].clone()
                all_joints_seq_gt = batch[:, args.input_n_dataset:args.input_n_dataset+args.output_n_dataset, :].clone()
                all_joints_seq = all_joints_seq_gt.clone()

                # ground truth with removed joints
                full_sequence_gt = all_joints_seq_gt[:, :, dim_used]

                # autoregressive
                loss, full_sequence_predict = autoregressive_process_batch(batch, model, args, dim_used, teacher_forcing=False)

                joint_angle_accum += joint_angle_error(full_sequence_predict, full_sequence_gt) * batch_dim
                running_loss += euler_error(full_sequence_predict, full_sequence_gt) * batch_dim
                accum_loss += euler_error(full_sequence_predict, full_sequence_gt) * batch_dim

        n_batches+=n
    print('overall average loss in euler angle is: '+str(accum_loss/n_batches))
    print('joint angle error is:', joint_angle_accum/n_batches)
    
    return accum_loss/n_batches, joint_angle_accum/n_batches


if __name__ == '__main__':
    # raise ValueError('This script is not supposed to be run directly. Use optuna_main.py instead.')

    parser = argparse.ArgumentParser(add_help=False) # Parameters for mpjpe
    
    if USER_NAME == "a":
        parser.add_argument('--data_dir', type=str,
                            default='/home/azhuavlev/Desktop/Data/CUDA_lab/VisionLabSS23_3DPoses',
                            help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
        parser.add_argument('--save_path',
                            default='/home/azhuavlev/Desktop/Results/CUDA_lab/Final_project/runs',
                            type=str, help='root path for the logging') #'./runs'
        parser.add_argument('--model_path', type=str,
                            default='/home/azhuavlev/Desktop/Results/CUDA_lab/Final_project/checkpoints',
                            help='directory with the models checkpoints ')
    elif USER_NAME =="v":
        parser.add_argument('--data_dir', type=str,
                            default='/home/user/bornhaup/FinalProject/VisionLabSS23_3DPoses',
                            help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
        parser.add_argument('--save_path',
                            default='/home/user/bornhaup/FinalProject/MotionMixerConv/runs',
                            type=str, help='root path for the logging') #'./runs'
        parser.add_argument('--model_path', type=str,
                            default='/home/user/bornhaup/FinalProject/MotionMixerConv/checkpoints',
                            help='directory with the models checkpoints ')  
    else: 
        raise ValueError('User not defined')


    ################################
    # Autoregressive settings
    ################################

    parser.add_argument('--input_n_model', type=int, default=10, help="number of model's input frames")
    parser.add_argument('--output_n_model', type=int, default=5, help="number of model's output frames")
    parser.add_argument('--input_n_dataset', type=int, default=10, help="number of ds's input frames")
    parser.add_argument('--output_n_dataset', type=int, default=25, help="number of ds's output frames")
    parser.add_argument('--step_window', type=int, default=5, help="step size for the sliding window")

    ###################################

    parser.add_argument('--skip_rate', type=int, default=1, choices=[1, 5], help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers in the dataloader')


    parser.add_argument('--activation', default='mish', type=str, required=False) 
    parser.add_argument('--r_se', default=8, type=int, required=False)

    parser.add_argument('--n_epochs', default=50, type=int, required=False)
    parser.add_argument('--batch_size', default=50, type=int, required=False)  
    parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
    parser.add_argument('--pin_memory', default=False, type=bool, required=False)
    parser.add_argument('--loader_workers', default=4, type=int, required=False)
    parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)
    parser.add_argument('--dev', default='cuda:0', type=str, required=False)
    parser.add_argument('--initialization', type=str, default='none', help='none, glorot_normal, glorot_uniform, hee_normal, hee_uniform')
    parser.add_argument('--use_scheduler', default=True, type=bool, required=False)
    parser.add_argument('--milestones', type=list, default=[15, 25, 35, 40], help='the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma correction to the learning rate, after reaching the milestone epochs')
    parser.add_argument('--clip_grad', type=float, default=None, help='select max norm to clip gradients')
    parser.add_argument('--actions_to_consider', default='all', help='Actions to visualize.Choose either all or a list of actions')
    parser.add_argument('--batch_size_test', type=int, default=50, help='batch size for the test set')
    parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'], help='choose data split to visualize from(train-val-test)')
    parser.add_argument('--loss_type', type=str, default='mpjpe', choices=['mpjpe', 'angle'])
    # parser.add_argument('--delta_x', type=bool, default=True, help='predicting the difference between 2 frames')

    args = parser.parse_args()

    if args.loss_type == 'mpjpe':
        parser_mpjpe = argparse.ArgumentParser(parents=[parser]) # Parameters for mpjpe
        parser_mpjpe.add_argument('--hidden_dim', default=192, type=int, required=False)
        parser_mpjpe.add_argument('--num_blocks', default=4, type=int, required=False)
        parser_mpjpe.add_argument('--channels_mlp_dim', default=8, type=int, required=False)
        parser_mpjpe.add_argument('--regularization', default=-1, type=float, required=False)
        parser_mpjpe.add_argument('--pose_dim', default=66, type=int, required=False)
        parser_mpjpe.add_argument('--delta_x', type=bool, default=False, help='predicting the difference between 2 frames')
        parser_mpjpe.add_argument('--lr', default=0.001, type=float, required=False)  
        args = parser_mpjpe.parse_args()
    
    elif args.loss_type == 'angle':
        parser_angle = argparse.ArgumentParser(parents=[parser]) # Parameters for angle
        parser_angle.add_argument('--hidden_dim', default=60, type=int, required=False) 
        parser_angle.add_argument('--num_blocks', default=3, type=int, required=False) 
        parser_angle.add_argument('--tokens_mlp_dim', default=40, type=int, required=False)
        parser_angle.add_argument('--channels_mlp_dim', default=60, type=int, required=False) 
        parser_angle.add_argument('--regularization', default=0.0, type=float, required=False)
        parser_angle.add_argument('--pose_dim', default=48, type=int, required=False)
        parser_angle.add_argument('--lr', default=1e-02, type=float, required=False)
        parser_angle.add_argument('--delta_x', type=bool, default=False,
                                  help='predicting the difference between 2 frames')
        args = parser_angle.parse_args()
    
    if args.loss_type == 'angle' and args.delta_x:
        raise ValueError('Delta_x and loss type angle cant be used together.')

    print(args)

    # model = MlpMixer(num_classes=args.pose_dim, # Number of input nodes
    #                  num_blocks=args.num_blocks, # Number of blocks
    #                  hidden_dim=args.hidden_dim, # dimPosEmb
    #                  tokens_mlp_dim=args.tokens_mlp_dim, #??
    #                  channels_mlp_dim=args.channels_mlp_dim, # ?? 
    #                  seq_len=args.input_n_dataset, # nTP
    #                  pred_len=args.output_n_dataset, # out_nTP
    #                  activation=args.activation, # activation function
    #                  mlp_block_type='normal',  # mlp_block_type
    #                  regularization=args.regularization, # regularization
    #                  input_size=args.pose_dim, # input_size
    #                  initialization='none', # Not used
    #                  r_se=args.r_se, # r_se
    #                  use_max_pooling=False, # use_max_pooling 
    #                  use_se=True) # use_se
    
    model = ConvMixer(num_blocks=args.num_blocks, 
                 dimPosIn=args.pose_dim,
                 dimPosEmb=args.hidden_dim,
                 dimPosOut=args.pose_dim, 
                 in_nTP=args.input_n_model,
                 out_nTP=args.output_n_model,

                  encoder_n_harmonic_functions=0,
                  encoder_omega0=0.1,

                 # out_nTP=args.output_n_frames_model = 5
                 conv_nChan=args.channels_mlp_dim, # TODO: Implement arg-parsing
                 conv1_kernel_shape=(5,5), # TODO: Implement arg-parsing

                 mode_conv="twice", # TODO: Implement arg-parsing

                 activation=args.activation,
                 regularization=args.regularization,  
                 use_se=True,
                 r_se=args.r_se, 
                 use_max_pooling=False)

    model = model.to(args.dev)

    print('total number of parameters of the network is: ' +
          str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    model_name = 'h36_3d_'+str(args.output_n_dataset)+'frames_ckpt_TeacherForcing'

    train_output = train_autoregressive(model, model_name, args)

    print('>>> Training finished', train_output)
    test_mpjpe_autoregressive(model, args, model_name=model_name, save_results=False)
