import torch
import os

import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import optuna
import sys

USER_NAME = 'v' # 'a' or 'v'
if USER_NAME == "a":
    sys.path.append('/home/azhuavlev/PycharmProjects/MotionMixerConv')
elif USER_NAME == "v":
    sys.path.append('/home/user/bornhaup/FinalProject/MotionMixerConv')

import h36m.train_mixer_h36m as train_mixer_h36m
from h36m.conv_mixer import ConvMixer
import shutil

class Objective:
    def __init__(self, study_dir):
        # Hold this implementation specific arguments as the fields of the class.
        self.study_dir = study_dir

        self.models_save_path = os.path.join(self.study_dir, 'models')


        if os.path.exists(self.models_save_path):
            # clear the folder
            print('Study directory already exists:', self.models_save_path)
            shutil.rmtree(self.models_save_path)
        os.makedirs(self.models_save_path)


    def parse_args(self):
        parser = argparse.ArgumentParser(add_help=False)  # Parameters for mpjpe

        ############################################################################
        # Directories
        ## Decide decide depending on current machine / user
        ############################################################################

        parser.add_argument('--user_name', type=str, default=USER_NAME, help='user name, a or v')
        if USER_NAME == "a":
            parser.add_argument('--data_dir', type=str,
                                default='/home/azhuavlev/Desktop/Data/CUDA_lab/VisionLabSS23_3DPoses',
                                help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
            parser.add_argument('--save_path',
                                # default='/home/azhuavlev/Desktop/Results/CUDA_lab/Final_project/runs',
                                default=self.models_save_path,
                                type=str, help='root path for logging and saving checkpoint')  # './runs'
            # parser.add_argument('--model_path', type=str,
            #                     default='/home/azhuavlev/Desktop/Results/CUDA_lab/Final_project/checkpoints',
            #                     help='directory with the models checkpoints ')
        elif USER_NAME == "v":
            parser.add_argument('--data_dir', type=str,
                                default='/home/user/bornhaup/FinalProject/VisionLabSS23_3DPoses',
                                help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
            parser.add_argument('--save_path', type=str,
                                default='/home/user/bornhaup/FinalProject/MotionMixerConv/runs',
                                help='root path for the logging and saving checkpoint') #'./runs'
            # parser.add_argument('--model_path', type=str,
            #                     default='/home/user/bornhaup/FinalProject/MotionMixerConv/checkpoints',
            #                     help='directory with the models checkpoints ')  # './checkpoints'
        else:
            raise ValueError('User not supported')
        
        ############################################################################
        # Dataset settings
        ############################################################################

        # sequence lengths
        parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
        parser.add_argument('--output_n', type=int, default=20, help="number of model's output frames")
        parser.add_argument('--skip_rate', type=int, default=1, choices=[1, 5],
                            help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
        parser.add_argument('--actions_to_consider', default='all',
                            help='Actions to visualize.Choose either all or a list of actions')

        # batch sizes
        parser.add_argument('--batch_size', default=50, type=int, required=False)
        parser.add_argument('--batch_size_test', type=int, default=50, help='batch size for the test set')

        # not important
        parser.add_argument('--num_worker', default=8, type=int, help='number of workers in the dataloader')
        parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
        parser.add_argument('--pin_memory', default=False, type=bool, required=False)
        parser.add_argument('--loader_workers', default=8, type=int, required=False)

        ############################################################################
        # Training settings
        ############################################################################

        # epochs / checkpoints
        parser.add_argument('--n_epochs', default=2, type=int, required=False)
        parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)

        # LR scheduler
        parser.add_argument('--use_scheduler', default=True, type=bool, required=False)
        parser.add_argument('--milestones', type=list, default=[15, 25, 35, 40],
                            help='the epochs after which the learning rate is adjusted by gamma')
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='gamma correction to the learning rate, after reaching the milestone epochs')

        # minor settings
        # parser.add_argument('--initialization', type=str, default='none',
        #                     help='none, glorot_normal, glorot_uniform, hee_normal, hee_uniform')
        parser.add_argument('--clip_grad', type=float, default=None, help='select max norm to clip gradients')
        parser.add_argument('--dev', default='cuda:0', type=str, required=False)

        # ? not used
        parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'],
                            help='choose data split to visualize from(train-val-test)')

        ############################################################################
        # Generic model settings
        ############################################################################
        parser.add_argument('--activation', default='mish', type=str, required=False)
        parser.add_argument('--r_se', default=8, type=int, required=False)

        ############################################################################
        # Loss type
        ############################################################################

        parser.add_argument('--loss_type', type=str, default='mpjpe', choices=['mpjpe', 'angle'])
        # parser.add_argument('--delta_x', type=bool, default=True, help='predicting the difference between 2 frames')

        args = parser.parse_args()

        ############################################################################
        # Specific arguments for each loss type
        ############################################################################

        parser_loss = argparse.ArgumentParser(parents=[parser])  # Parameters for loss
        if args.loss_type == 'mpjpe':
            # optimizable
            # parser_loss.add_argument('--dimPosEmb', default=50, type=int, required=False)
            # parser_loss.add_argument('--num_blocks', default=4, type=int, required=False)
            # parser_loss.add_argument('--tokens_mlp_dim', default=20, type=int, required=False)
            # parser_loss.add_argument('--channels_mlp_dim', default=50, type=int, required=False)
            # parser_loss.add_argument('--lr', default=0.001, type=float, required=False)
            # parser_loss.add_argument('--regularization', default=0.1, choices=[-1, 0, 0.1], type=float, required=False)

            # not optimizable
            parser_loss.add_argument('--pose_dim', default=66, type=int, required=False)
            parser_loss.add_argument('--delta_x', type=bool, default=False,
                                    help='predicting the difference between 2 frames')


        elif args.loss_type == 'angle':
            # optimizable
            # parser_loss.add_argument('--dimPosEmb', default=60, type=int, required=False)
            # parser_loss.add_argument('--num_blocks', default=3, type=int, required=False)
            # parser_loss.add_argument('--tokens_mlp_dim', default=40, type=int, required=False)
            # parser_loss.add_argument('--channels_mlp_dim', default=60, type=int, required=False)
            # parser_loss.add_argument('--lr', default=1e-02, type=float, required=False)
            # parser_loss.add_argument('--regularization', default=0.0, choices=[-1, 0, 0.1], type=float, required=False)

            # not optimizable
            parser_loss.add_argument('--pose_dim', default=48, type=int, required=False)
            parser_loss.add_argument('--delta_x', type=bool, default=False,
                                    help='predicting the difference between 2 frames')
        else:
            raise ValueError('Loss type not supported')
        
        
        ############################################################################
        # Parameters optimizable by optuna
        ############################################################################
        
        parser_loss.add_argument(
            '--dimPosEmb',
            default=50,
            type=int, required=False)
        parser_loss.add_argument(
            '--num_blocks',
            default=6,
            type=int, required=False)
        parser_loss.add_argument(
            '--channels_conv_blocks',
            default=1,
            type=int, required=False)
        parser_loss.add_argument(
            '--kernel1_x_Time',
            default=5,
            type=int, required=False)
        parser_loss.add_argument(
            '--kernel1_y_Pose',
            default=5, 
            type=int, required=False)
        parser_loss.add_argument(
            '--kernel2_x_Time',
            default=5,
            type=int, required=False)
        parser_loss.add_argument(
            '--kernel2_y_Pose',
            default=5, 
            type=int, required=False)
        parser_loss.add_argument(
            '--lr',
            default=1e-03,
            type=float, required=False)
        parser_loss.add_argument(
            '--regularization', # -1 for BatchNorm1d, 0 for no regularization, 0.1 for Dropout(0.1)
            default=-1,
            choices=[-1, 0, 0.1], type=float, required=False)
        parser_loss.add_argument(
            '--conv_mode',
            default='once',
            choices=['once', 'twice'], type=str, required=False)

        ############################################################################
        # Parse arguments
        ############################################################################
        args = parser_loss.parse_args()
        if args.loss_type == 'angle' and args.delta_x:
            raise ValueError('Delta_x and loss type angle cant be used together.')
        assert args.kernel1_x_Time <= args.input_n, "Kernel 1 has wrong size in x dim" # input_n == in_nTP
        assert args.kernel1_y_Pose <= args.dimPosEmb, "Kernel 1 has wrong size in y dim"
        assert args.kernel2_x_Time <= args.input_n, "Kernel 2 has wrong size in x dim" # input_n == in_nTP
        assert args.kernel2_y_Pose <= args.dimPosEmb, "Kernel 2 has wrong size in y dim"
        return args


    def overwrite_optuna_params(self, args, trial):
        args.n_epochs = trial.suggest_int('n_epochs', 1, 50)
        args.dimPosEmb = trial.suggest_int('dimPosEmb', 10, 100)
        args.num_blocks = trial.suggest_int('num_blocks', 1, 7)
        args.channels_conv_blocks = trial.suggest_int('channels_conv_blocks', 1, 10)
        args.kernel1_x_Time = trial.suggest_int('kernel1_x_Time', 1, args.input_n)
        args.kernel1_y_Pose = trial.suggest_int('kernel1_y_Pose', 1, args.dimPosEmb)
        args.lr = trial.suggest_float('lr', 1e-04, 1e-02)
        args.regularization = trial.suggest_categorical('regularization', [-1, 0, 0.1, 0.2])
        args.conv_mode = trial.suggest_categorical('conv_mode', ['once', 'twice'])
        if args.conv_mode == 'twice':
            args.kernel2_x_Time = trial.suggest_int('kernel2_x_Time', 1, args.input_n)
            args.kernel2_y_Pose = trial.suggest_int('kernel2_y_Pose', 1, args.dimPosEmb)
        return args, trial
    

    def __call__(self, trial):
        args = self.parse_args()
        args, trial = self.overwrite_optuna_params(args, trial)

        ############################################################################
        # Create model
        ############################################################################

        model = ConvMixer(num_blocks=args.num_blocks,
                            dimPosIn=args.pose_dim,
                            dimPosEmb=args.dimPosEmb,
                            dimPosOut=args.pose_dim,
                            in_nTP=args.input_n,
                            out_nTP=args.output_n,
                            conv_nChan=args.channels_conv_blocks,
                            conv1_kernel_shape=(args.kernel1_x_Time, args.kernel1_y_Pose),
                            conv1_stride=(1,1),
                            conv1_padding=None,
                            mode_conv=args.conv_mode,
                            conv2_kernel_shape=(args.kernel2_x_Time, args.kernel2_y_Pose),
                            conv2_stride=(1,1),
                            conv2_padding=None,
                            activation=args.activation,
                            regularization=args.regularization,
                            use_se=True,
                            r_se=args.r_se,
                            use_max_pooling=False,
                            encoder_n_harmonic_functions=64 # Is this flexible?
                        )
        model = model.to(args.dev)

        print(args)
        print('total number of parameters of the network is: ' +
              str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        model_name = f'h3.6m_{args.loss_type}_'\
                     f'hidden_dim={args.dimPosEmb}_'\
                     f'num_blocks={args.num_blocks}_' \
                     f'k1x={args.kernel1_x_Time}_'\
                     f'k1y={args.kernel1_y_Pose}_'\
                     f'conv_mode={args.conv_mode}_'\
                     f'k2x={args.kernel2_x_Time}_'\
                     f'k2y={args.kernel2_y_Pose}_'\
                     f'lr={args.lr:.4f}_'\
                     f'regularization={args.regularization}'

        ############################################################################
        # Train and evaluate model
        ############################################################################

        train_loss_list, val_loss_list, test_loss_list, metrics_dict = train_mixer_h36m.train(model, model_name, args)

        # I am not sure if this is necessary
        # test_loss_final = train_mixer_h36m.test_mpjpe(model, args)

        # IMPORTANT: we will optimize val_loss, and report train_loss and test_loss
        trial.set_user_attr("train_loss", train_loss_list[-1].item())
        trial.set_user_attr("val_loss", val_loss_list[-1].item())
        trial.set_user_attr("test_loss", test_loss_list[-1].item())

        for metric_name, metric_value in metrics_dict.items():
            trial.set_user_attr(metric_name, metric_value[-1].item())

        return val_loss_list[-1].item()


if __name__ == '__main__':

    if USER_NAME == "a":
        base_folder = f'/home/azhuavlev/Desktop/Results/CUDA_lab/Final_project/studies'
    elif USER_NAME == "v":
        base_folder = f'/home/user/bornhaup/FinalProject/MotionMixerConv/studies'
    else:
        raise ValueError('User not supported')
    study_name = 'example-study_out_nTP=20'

    study_path = base_folder + '/' + study_name
    if os.path.exists(study_path):
        # clear the folder
        print('Study directory already exists:', study_path)
        shutil.rmtree(study_path)
    os.makedirs(study_path)

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{base_folder}/{study_name}/results.db",
    )
    # To use the dashboard, run the following command:
    # optuna-dashboard sqlite:////home/azhuavlev/Desktop/Results/CUDA_lab/Final_project/studies/example-study/results.db
    # respectively: optuna-dashboard sqlite:////home/user/bornhaup/FinalProject/MotionMixerConv/studies/example-study_out_nTP=20/results.db
    # then: ssh -L 8080:127.0.0.1:8080 cuda4

    study.optimize(
        Objective(f'{base_folder}/{study_name}'),
        # direction="minimize",
        n_trials=50,
        timeout=60*60*12 # 12 hours
    )
    print('Number of finished trials:', len(study.trials))
    print(study.best_params)