import torch
import os

import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import optuna
import sys

sys.path.append('/home/azhuavlev/PycharmProjects/MotionMixerConv')
import h36m.train_mixer_h36m as train_mixer_h36m
from h36m.mlp_mixer import MlpMixer
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





    def __call__(self, trial):
        parser = argparse.ArgumentParser(add_help=False)  # Parameters for mpjpe

        ############################################################################
        # Directories
        ############################################################################

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

        ############################################################################
        # Dataset settings
        ############################################################################

        # sequence lengths
        parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
        parser.add_argument('--output_n', type=int, default=25, help="number of model's output frames")
        parser.add_argument('--skip_rate', type=int, default=1, choices=[1, 5],
                            help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
        parser.add_argument('--actions_to_consider', default='all',
                            help='Actions to visualize.Choose either all or a list of actions')

        # batch sizes
        parser.add_argument('--batch_size', default=50, type=int, required=False)
        parser.add_argument('--batch_size_test', type=int, default=50, help='batch size for the test set')

        # not important
        parser.add_argument('--num_worker', default=4, type=int, help='number of workers in the dataloader')
        parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
        parser.add_argument('--pin_memory', default=False, type=bool, required=False)
        parser.add_argument('--loader_workers', default=4, type=int, required=False)

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
        parser.add_argument('--initialization', type=str, default='none',
                            help='none, glorot_normal, glorot_uniform, hee_normal, hee_uniform')
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
            # parser_loss.add_argument('--hidden_dim', default=50, type=int, required=False)
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
            # parser_loss.add_argument('--hidden_dim', default=60, type=int, required=False)
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
            '--hidden_dim',
            default=trial.suggest_int('hidden_dim', 10, 100),
            type=int, required=False)
        parser_loss.add_argument(
            '--num_blocks',
            default=trial.suggest_int('num_blocks', 1, 7),
            type=int, required=False)
        parser_loss.add_argument(
            '--tokens_mlp_dim',
            default=trial.suggest_int('tokens_mlp_dim', 10, 100),
            type=int, required=False)
        parser_loss.add_argument(
            '--channels_mlp_dim',
            default=trial.suggest_int('channels_mlp_dim', 10, 100),
            type=int, required=False)
        parser_loss.add_argument(
            '--lr',
            default=trial.suggest_float('lr', 1e-04, 1e-02),
            type=float, required=False)
        parser_loss.add_argument(
            '--regularization', # -1 for BatchNorm1d, 0 for no regularization, 0.1 for Dropout(0.1)
            default=trial.suggest_categorical('regularization', [-1, 0, 0.1]),
            choices=[-1, 0, 0.1], type=float, required=False)

        ############################################################################
        # Parse arguments
        ############################################################################

        args = parser_loss.parse_args()
        if args.loss_type == 'angle' and args.delta_x:
            raise ValueError('Delta_x and loss type angle cant be used together.')

        ############################################################################
        # Create model
        ############################################################################

        model = MlpMixer(num_classes=args.pose_dim, num_blocks=args.num_blocks,
                         hidden_dim=args.hidden_dim, tokens_mlp_dim=args.tokens_mlp_dim,
                         channels_mlp_dim=args.channels_mlp_dim, seq_len=args.input_n,
                         pred_len=args.output_n, activation=args.activation,
                         mlp_block_type='normal', regularization=args.regularization,
                         input_size=args.pose_dim, initialization='none', r_se=args.r_se,
                         use_max_pooling=False, use_se=True)
        model = model.to(args.dev)

        print(args)
        print('total number of parameters of the network is: ' +
              str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        model_name = f'h3.6m_{args.loss_type}_hidden_dim={args.hidden_dim}_num_blocks={args.num_blocks}_' \
            f'tokens_mlp_dim={args.tokens_mlp_dim}_channels_mlp_dim={args.channels_mlp_dim}_' \
            f'lr={args.lr:.4f}_regularization={args.regularization}'

        ############################################################################
        # Train and evaluate model
        ############################################################################

        train_loss_list, val_loss_list, test_loss_list = train_mixer_h36m.train(model, model_name, args)

        # I am not sure if this is necessary
        # test_loss_final = train_mixer_h36m.test_mpjpe(model, args)

        # we will optimize val_loss
        return val_loss_list[-1].item()


if __name__ == '__main__':

    base_folder = f'/home/azhuavlev/Desktop/Results/CUDA_lab/Final_project/studies'
    study_name = 'example-study'

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
    # optuna-dashboard sqlite:////home/azhuavlev/Desktop/Results/CUDA_lab/Final_project/studies/example-study/results.db

    study.optimize(
        Objective(f'{base_folder}/{study_name}'),
        # direction="minimize",
        n_trials=5
    )
    print('Number of finished trials:', len(study.trials))
    print(study.best_params)