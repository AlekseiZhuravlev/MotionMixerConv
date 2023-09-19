
import torch
import os

import sys
USER_NAME = 'v'
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
        for cnt, batch in enumerate(data_loader):
            batch = batch.to(device)
            batch_dim = batch.shape[0]
            n += batch_dim

            optimizer.zero_grad()


            #######################
            # Autoregressive Code #
            #######################
            
            # assert that args.output_n_dataset is "big enough"
            assert args.output_n_dataset >= args.input_n_dataset - args.input_n_model 
            loss = torch.zeros(1).to(device)

            if args.loss_type == 'mpjpe':
                full_sequences_train = batch[:, 0 : args.input_n_dataset, dim_used].view(-1, args.input_n_dataset, args.pose_dim)
                full_sequences_gt = batch[:, args.input_n_dataset:args.input_n_dataset + args.output_n_dataset, dim_used].view(-1, args.output_n_dataset, args.pose_dim) # Shape [* , output_n_dataset, pose_dim]
                normalize_factor = 1.0/1000.0
                loss_fct = lambda pred, gt, out_n: mpjpe_error(pred, gt)
            elif args.loss_type == 'angle':
                full_sequences_train = batch[:, 0:args.input_n_dataset, dim_used].view(-1,args.input_n_dataset,len(dim_used))
                full_sequences_gt = batch[:, args.input_n_dataset:args.input_n_dataset+args.output_n_dataset, dim_used] # Shape [* , output_n_dataset, len(dim_used)]
                normalize_factor = 1.0
                loss_fct = lambda pred, gt, out_n: torch.mean(torch.sum(torch.abs(pred.reshape(-1,out_n,len(dim_used)) - gt), dim=2).view(-1))
                

            for cumul_steps in range(0, args.input_n_dataset - args.input_n_model, args.output_n_model): 
                # Apply Teacher forcing during training:
                subsequence_train = normalize_factor * full_sequences_train[:, cumul_steps : cumul_steps + args.input_n_model, :]
                subsequence_gt = full_sequences_gt[:, cumul_steps: cumul_steps + args.output_n_model, :]
                subsequences_predict = model(subsequence_train) # predict next args.output_n_model steps
                
                # compute error with gt and sum the loss
                loss += loss_fct(subsequences_predict, subsequence_gt, args.output_n_model)

            # TODO: Aleksei: Check how the loss is calculated to make sure it's consistent
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad)

            optimizer.step()

            running_loss += loss*batch_dim

        train_loss.append(running_loss.detach().cpu()/n)
        model.eval()
        with torch.no_grad():
            running_loss = 0
            n = 0
            for cnt, batch in enumerate(vald_loader):
                batch = batch.to(device)
                batch_dim = batch.shape[0]
                n += batch_dim
                assert args.output_n_dataset >= args.input_n_dataset - args.input_n_model, "Autoregression: Output of the Dataset is to small"
                loss = torch.zeros(1).to(device)

                if args.loss_type == 'mpjpe':
                    full_sequences_eval = batch[:, 0 : args.input_n_dataset, dim_used].view(-1, args.input_n_dataset, args.pose_dim)
                    full_sequences_gt = batch[:, args.input_n_dataset:args.input_n_dataset + args.output_n_dataset, dim_used].view(-1, args.output_n_dataset, args.pose_dim) # Shape [* , output_n_dataset, pose_dim]
                    normalize_factor = 1.0/1000.0
                    loss_fct = lambda pred, gt, out_n: mpjpe_error(pred, gt)
                elif args.loss_type == 'angle':
                    full_sequences_eval = batch[:, 0:args.input_n_dataset, dim_used].view(-1,args.input_n_dataset,len(dim_used))
                    full_sequences_gt = batch[:, args.input_n_dataset:args.input_n_dataset+args.output_n_dataset, dim_used] # Shape [* , output_n_dataset, len(dim_used)]
                    normalize_factor = 1.0
                    loss_fct = lambda pred, gt, out_n: torch.mean(torch.sum(torch.abs(pred.reshape(-1,out_n,len(dim_used)) - gt), dim=2).view(-1))
                    
                subsequence_eval = normalize_factor * full_sequences_eval[:, :args.input_n_model, :]
                elem_to_take = max(0, args.input_n_model - args.output_n_model) # 
                for cumul_steps in range(0, args.output_n_dataset, args.output_n_model): 
                    subsequence_gt = full_sequences_gt[:, cumul_steps: cumul_steps + args.output_n_model, :]
                    subsequences_predict = model(subsequence_eval) # predict next args.output_n_model steps
                    
                    # compute error with gt and sum the loss
                    loss += loss_fct(subsequences_predict, subsequence_gt, args.output_n_model)

                    subsequence_eval = torch.cat((subsequence_eval[:, -elem_to_take:, :], subsequences_predict), dim=1)

                running_loss += loss*batch_dim
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


def test_mpjpe_autoregressive(model, args, model_name, save_results):

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

                print('Batch shape', batch.shape)
                all_joints_input = batch.clone()[:, :args.input_n_dataset, :]
                all_joints_seq = batch.clone()[:, args.input_n_dataset:args.input_n_dataset+args.output_n_dataset, :]
                all_joints_seq_gt = batch.clone()[:, args.input_n_dataset:args.input_n_dataset+args.output_n_dataset, :] # Actually torch.zeros_like(batch[:,args...,:]) would do it as well

                assert not args.delta_x
                assert args.output_n_dataset >= args.input_n_dataset - args.input_n_model, "Autoregression: Output of the Dataset is to small"
                loss = torch.zeros(1).to(device)

                full_sequences_test = batch[:, 0 : args.input_n_dataset, dim_used].view(-1, args.input_n_dataset, args.pose_dim)
                full_sequences_gt = batch[:, args.input_n_dataset:args.input_n_dataset + args.output_n_dataset, dim_used].view(-1, args.output_n_dataset, args.pose_dim) # Shape [* , output_n_dataset, pose_dim]
                full_sequences_predict = torch.zeros_like(full_sequences_gt)

                subsequence_test = full_sequences_test[:, :args.input_n_model, :] / 1000
                elem_to_take = max(0, args.input_n_model - args.output_n_model) # 
                for cumul_steps in range(0, args.input_n_dataset - args.input_n_model, args.output_n_model): 
                    subsequence_gt = full_sequences_gt[:, cumul_steps: cumul_steps + args.output_n_model, :]
                    subsequences_predict = model(subsequence_test) # predict next args.output_n_model steps
                    full_sequences_predict[:, cumul_steps: cumul_steps + args.output_n_model, :] = subsequences_predict # Shape [* , output_n_dataset, pose_dim

                    # compute error with gt and sum the loss
                    loss += mpjpe_error(subsequences_predict, subsequence_gt)

                    subsequence_test = torch.cat((subsequence_test[:, -elem_to_take:, :], subsequences_predict), dim=1)

                all_joints_input[:, :, dim_used] = full_sequences_test * 1000
                all_joints_input[:, :, index_to_ignore] = all_joints_input[:, :, index_to_equal]

                all_joints_seq[:, :, dim_used] = full_sequences_predict
                all_joints_seq[:, :, index_to_ignore] = all_joints_seq[:, :, index_to_equal]

                all_joints_seq_gt[:, :, dim_used] = full_sequences_gt
                all_joints_seq_gt[:, :, index_to_ignore] = all_joints_seq_gt[:, :, index_to_equal]

                print('all_joints_seq.shape', all_joints_seq.shape)
                print('all_joints_seq_gt.shape', all_joints_seq_gt.shape)
                loss = mpjpe_error(all_joints_seq.view(-1, args.output_n_dataset, 33, 3), # all_joints_seq.view(-1, args.output_n_dataset, 32, 3)
                                   all_joints_seq_gt.view(-1, args.output_n_dataset, 33, 3) # all_joints_seq_gt.view(-1, args.output_n_dataset, 32, 3)
                                   )

                # print('sequences_predict.shape', sequences_predict.shape)
                # print(sequences_predict)
                #
                # print('all_joints_seq.shape', all_joints_seq.shape)
                # print(all_joints_seq)
                # exit(0)

                print('full_sequences_predict.shape', full_sequences_predict.shape)
                print('full_sequences_gt.shape', full_sequences_gt.shape)
                auc_pck_batch = auc_pck_metric(
                    full_sequences_predict.view(-1, args.output_n_dataset, 16, 3) / 1000, # full_sequences_predict.view(-1, args.output_n_dataset, 22, 3) / 1000,
                    full_sequences_gt.view(-1, args.output_n_dataset, 16, 3) / 1000 # full_sequences_gt.view(-1, args.output_n_dataset, 22, 3) / 1000
                )

                auc_pck_accum += auc_pck_batch*batch_dim
                running_loss += loss*batch_dim
                accum_loss += loss*batch_dim

                if save_results and cnt in sequences_to_save:

                    # print('batch_full', all_joints_seq[10].cpu().shape)
                    # print('batch_gt', all_joints_seq_gt[10].cpu().shape)
                    # print('batch_train', all_joints_input[10].cpu().shape)

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
    accum_loss=0  
    n_batches=0 # number of batches for all the sequences
    actions=define_actions(args.actions_to_consider)
    dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                            43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                            86])

    for action in actions:

        joint_angle_error_running = 0
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
                
                assert args.output_n_dataset >= args.input_n_dataset - args.input_n_model, "Autoregression: Output of the Dataset is to small"
                loss = 0

                full_sequences_test = batch[:, 0:args.input_n_dataset, dim_used].view(-1,args.input_n_dataset,len(dim_used))
                full_sequences_gt = batch[:, args.input_n_dataset:args.input_n_dataset+args.output_n_dataset, :] # Shape [* , output_n_dataset, angle_nums]
                all_joints_seq=batch.clone()[:, args.input_n_dataset : args.input_n_dataset + args.output_n_dataset, :] # [batch_size, args.output_n_dataset, angle_nums]

                subsequence_test = full_sequences_test[:, :args.input_n_model, :]
                elem_to_take_from_old_input = max(0, args.input_n_model - args.output_n_model) # 
                for cumul_steps in range(0, args.input_n_dataset - args.input_n_model, args.output_n_model): 
                    subsequence_gt = full_sequences_gt[:, cumul_steps: cumul_steps + args.output_n_model, :]
                    subsequences_predict = model(subsequence_test) # predict next args.output_n_model steps
                    
                    # compute error with gt and sum the loss
                    all_joints_subseq = all_joints_seq[:, cumul_steps: cumul_steps + args.output_n_model, :] 
                    all_joints_subseq[:, :, dim_used] = subsequences_predict
                    assert all_joints_subseq.shape == subsequence_gt.shape, "Shapes should be equal, but " +str(all_joints_subseq.shape) +" != " +str(subsequence_gt.shape)
                    loss += euler_error(all_joints_subseq, subsequence_gt)
                    joint_angle_error_running += joint_angle_error(all_joints_subseq, subsequence_gt) * batch_dim

                    # Setup next input (autoregressiveness)
                    subsequence_test = torch.cat((subsequence_test[:, -elem_to_take_from_old_input:, :], subsequences_predict), dim=1)
                # TODO: Check if joint_angle_error_running has to be devided by the number of for-loop-steps.
                
                running_loss+=loss*batch_dim
                accum_loss+=loss*batch_dim

        n_batches+=n
    print('overall average loss in euler angle is: '+str(accum_loss/n_batches))
    print('joint angle error is:', joint_angle_error_running/n_batches)
    
    return accum_loss/n_batches, joint_angle_error_running/n_batches


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


    parser.add_argument('--input_n_model', type=int, default=10, help="number of model's input frames")
    parser.add_argument('--output_n_model', type=int, default=5, help="number of model's output frames")
    parser.add_argument('--input_n_dataset', type=int, default=25, help="number of ds's input frames")
    parser.add_argument('--output_n_dataset', type=int, default=15, help="number of ds's output frames")
    parser.add_argument('--skip_rate', type=int, default=1, choices=[1, 5], help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers in the dataloader')


    parser.add_argument('--activation', default='mish', type=str, required=False) 
    parser.add_argument('--r_se', default=8, type=int, required=False)

    parser.add_argument('--n_epochs', default=2, type=int, required=False)
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
    parser.add_argument('--batch_size_test', type=int, default=256, help='batch size for the test set')
    parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'], help='choose data split to visualize from(train-val-test)')
    parser.add_argument('--loss_type', type=str, default='angle', choices=['mpjpe', 'angle'])
    # parser.add_argument('--delta_x', type=bool, default=True, help='predicting the difference between 2 frames')

    args = parser.parse_args()

    if args.loss_type == 'mpjpe':
        parser_mpjpe = argparse.ArgumentParser(parents=[parser]) # Parameters for mpjpe
        parser_mpjpe.add_argument('--hidden_dim', default=50, type=int, required=False)  
        parser_mpjpe.add_argument('--num_blocks', default=4, type=int, required=False)  
        parser_mpjpe.add_argument('--tokens_mlp_dim', default=20, type=int, required=False)
        parser_mpjpe.add_argument('--channels_mlp_dim', default=50, type=int, required=False)  
        parser_mpjpe.add_argument('--regularization', default=0.1, type=float, required=False)  
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

                 # out_nTP=args.output_n_frames_model = 5
                 conv_nChan=1, # TODO: Implement arg-parsing
                 conv1_kernel_shape=(1,3), # TODO: Implement arg-parsing
                 conv1_stride=(1,1), # TODO: Implement arg-parsing
                 conv1_padding=(0,1), # TODO: Implement arg-parsing
                 mode_conv="twice", # TODO: Implement arg-parsing
                 conv2_kernel_shape=None, # TODO: Implement arg-parsing
                 conv2_stride=None, # TODO: Implement arg-parsing
                 conv2_padding=None, # TODO: Implement arg-parsing
                 activation=args.activation,
                 regularization=args.regularization,  
                 use_se=True,
                 r_se=args.r_se, 
                 use_max_pooling=False)

    model = model.to(args.dev)

    print('total number of parameters of the network is: ' +
          str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    model_name = 'h36_3d_'+str(args.output_n_dataset)+'frames_ckpt'

    train_output = train_autoregressive(model, model_name, args)

    print('>>> Training finished', train_output)
    test_mpjpe_autoregressive(model, args, model_name=model_name, save_results=False)
