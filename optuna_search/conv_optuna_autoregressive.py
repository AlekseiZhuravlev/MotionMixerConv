import argparse
import numpy as np
import optuna
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy

USER_NAME = 'a'  # 'a' or 'v'
if USER_NAME == "a":
    sys.path.append('/home/azhuavlev/PycharmProjects/MotionMixerConv')
elif USER_NAME == "v":
    sys.path.append('/home/user/bornhaup/FinalProject/MotionMixerConv')

import h36m.train_autoreg_mixer_h36m as train_autoreg_mixer_h36m
import h36m.train_autoreg_mixer_ais as train_autoreg_mixer_ais
from h36m.conv_mixer_model import ConvMixer
import shutil


class Objective:
    def __init__(self, study_dir):
        # Hold this implementation specific arguments as the fields of the class.
        self.study_dir = study_dir

        self.models_save_path = os.path.join(self.study_dir, 'models')

        if os.path.exists(self.models_save_path):
            # clear the folder
            print('Study directory already exists:', self.models_save_path)
            # shutil.rmtree(self.models_save_path)
        else:
            os.makedirs(self.models_save_path, exist_ok=True)

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
                                default=self.models_save_path,
                                type=str, help='root path for logging and saving checkpoint')  # './runs'
        elif USER_NAME == "v":
            parser.add_argument('--data_dir', type=str,
                                default='/home/user/bornhaup/FinalProject/VisionLabSS23_3DPoses',
                                help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
            parser.add_argument('--save_path', type=str,
                                default='/home/user/bornhaup/FinalProject/MotionMixerConv/runs',
                                help='root path for the logging and saving checkpoint')  # './runs'
        else:
            raise ValueError('User not supported')

        ############################################################################
        # Dataset settings
        ############################################################################

        # sequence lengths
        parser.add_argument('--input_n_model', type=int, default=10, help="number of model's input frames")
        parser.add_argument('--output_n_model', type=int, default=5, help="number of model's output frames")
        parser.add_argument('--input_n_dataset', type=int, default=10, help="number of ds's input frames")
        parser.add_argument('--output_n_dataset', type=int, default=25, help="number of ds's output frames")
        parser.add_argument('--step_window', type=int, default=5, help="step size for the sliding window")
        parser.add_argument('--n_epochs_teacher_forcing', type=int, default=5, help="number of epochs to use teacher forcing")

        parser.add_argument('--skip_rate', type=int, default=1, choices=[1, 5],
                            help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
        parser.add_argument('--actions_to_consider', default='all',
                            help='Actions to visualize.Choose either all or a list of actions')

        # batch sizes
        parser.add_argument('--batch_size', default=50, type=int, required=False)
        parser.add_argument('--batch_size_test', type=int, default=50, help='batch size for the test set')

        # not important
        parser.add_argument('--num_worker', default=3, type=int, help='number of workers in the dataloader')
        parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
        parser.add_argument('--pin_memory', default=False, type=bool, required=False)
        parser.add_argument('--loader_workers', default=3, type=int, required=False)

        ############################################################################
        # Training settings
        ############################################################################

        # epochs / checkpoints

        parser.add_argument('--dataset_type', default='h36m', type=str, choices=['h36m', 'ais'], required=False)
        parser.add_argument('--n_epochs', default=50, type=int, required=False) # 50
        parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)

        # LR scheduler
        parser.add_argument('--lr', default=1e-03, type=float, required=False)
        parser.add_argument('--use_scheduler', default=True, type=bool, required=False)
        parser.add_argument('--milestones', type=list, default=[25, 40],
                            help='the epochs after which the learning rate is adjusted by gamma')
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='gamma correction to the learning rate, after reaching the milestone epochs')

        # minor settings
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

        parser.add_argument(
            '--regularization', # -1 for BatchNorm1d, 0 for no regularization, 0.1 for Dropout(0.1)
            default=-1.0,
            choices=[-1, 0, 0.1], type=float, required=False)

        ############################################################################
        # Parse arguments
        ############################################################################
        args = parser.parse_args()

        args.encoder_n_harmonic_functions = 0
        args.encoder_omega0 = 0

        return args


    def train_model_with_loss(self, args, trial, loss_type, pose_dim):

        ############################################################################
        # Train with mpjpe loss
        ############################################################################

        args.loss_type = loss_type
        args.delta_x = False
        args.pose_dim = pose_dim

        model = ConvMixer(
            # not optimizable
            dimPosIn=args.pose_dim,
            dimPosOut=args.pose_dim,
            in_nTP=args.input_n_model,
            out_nTP=args.output_n_model,

            # optimizable
            num_blocks=args.num_blocks,
            dimPosEmb=args.dimPosEmb,
            conv_nChan=args.channels_conv_blocks,
            conv1_kernel_shape=(args.kernel1_x_Time, args.kernel1_y_Pose),
            encoder_n_harmonic_functions=args.encoder_n_harmonic_functions,
            encoder_omega0=args.encoder_omega0,

            mode_conv="twice",
            activation=args.activation,
            regularization=args.regularization,
            use_se=True,
            r_se=args.r_se,
            use_max_pooling=False,
        ).to(args.dev)

        print(args)
        print('total number of parameters of the network is: ' +
              str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        model_name = f'h3.6m_autoregressive_{args.loss_type}_' \
                        f'input_n={args.input_n_model}_' \
                        f'output_n={args.output_n_model}_' \
                        f'skip_rate={args.skip_rate}_' \
                        f'actions_to_consider={args.actions_to_consider}_' \
                        f'num_blocks={args.num_blocks}_' \
                        f'regularization={args.regularization}_' \
                     f'hidden_dim={args.dimPosEmb}_' \
                     f'k1x={args.kernel1_x_Time}_' \
                     f'k1y={args.kernel1_y_Pose}_' \
                     f'channels_conv_blocks={args.channels_conv_blocks}_' \
                     f'encoder_n_harmonic_functions={args.encoder_n_harmonic_functions}_' \
                     f'encoder_omega0={args.encoder_omega0}_'

        train_loss_list, val_loss_list, test_loss_list, metrics_dict = \
            train_autoreg_mixer_h36m.train_autoregressive(model, model_name, args)

        # save gif of the predictions
        if args.loss_type == 'mpjpe':
            train_autoreg_mixer_h36m.test_mpjpe_autoregressive(model, args, model_name, save_results=True)

        # IMPORTANT: we will optimize val_loss, and report train_loss and test_loss
        trial.set_user_attr(f"train_loss_{loss_type}", train_loss_list[-1].item())
        trial.set_user_attr(f"val_loss_{loss_type}", val_loss_list[-1].item())
        trial.set_user_attr(f"test_loss_{loss_type}", test_loss_list[-1].item())

        # save metrics
        for metric_name, metric_value in metrics_dict.items():
            trial.set_user_attr(metric_name, metric_value[-1].item())

        # evaluate on each action separately
        for action in tqdm(["walking", "eating", "smoking", "discussion", "directions",
                       "greeting", "phoning", "posing", "purchases", "sitting",
                       "sittingdown", "takingphoto", "waiting", "walkingdog",
                       "walkingtogether"], desc='evaluating metrics on each action'):
            args_action = copy.deepcopy(args)
            args_action.actions_to_consider = action
            print(f'evaluating metrics on action {action}')

            if args.loss_type == 'mpjpe':
                action_mpjpe_loss, action_auc_pck = train_autoreg_mixer_h36m.test_mpjpe_autoregressive(model, args_action, model_name,
                                                                                save_results=False)
                trial.set_user_attr(f"{action}/mpjpe", action_mpjpe_loss.item())
                trial.set_user_attr(f"{action}/auc_pck", action_auc_pck.item())
            elif args.loss_type == 'angle':
                action_euler_angle_loss, action_joint_angle_loss = train_autoreg_mixer_h36m.test_angle_autoregressive(model, args_action)
                trial.set_user_attr(f"{action}/euler_angle", action_euler_angle_loss.item())
                trial.set_user_attr(f"{action}/joint_angle", action_joint_angle_loss.item())

        return test_loss_list[-1].item()


    def train_model_ais(self, args, trial, loss_type, pose_dim):

        ############################################################################
        # Train with mpjpe loss
        ############################################################################

        args.loss_type = loss_type
        args.delta_x = False
        args.pose_dim = pose_dim

        model = ConvMixer(
            # not optimizable
            dimPosIn=args.pose_dim,
            dimPosOut=args.pose_dim,
            in_nTP=args.input_n_model,
            out_nTP=args.output_n_model,

            # optimizable
            num_blocks=args.num_blocks,
            dimPosEmb=args.dimPosEmb,
            conv_nChan=args.channels_conv_blocks,
            conv1_kernel_shape=(args.kernel1_x_Time, args.kernel1_y_Pose),
            encoder_n_harmonic_functions=args.encoder_n_harmonic_functions,
            encoder_omega0=args.encoder_omega0,

            mode_conv="twice",
            activation=args.activation,
            regularization=args.regularization,
            use_se=True,
            r_se=args.r_se,
            use_max_pooling=False,
        ).to(args.dev)

        print(args)
        print('total number of parameters of the network is: ' +
              str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        model_name = f'ais_{args.loss_type}_' \
                        f'input_n={args.input_n_model}_' \
                        f'output_n={args.output_n_model}_' \
                        f'skip_rate={args.skip_rate}_' \
                        f'actions_to_consider={args.actions_to_consider}_' \
                        f'num_blocks={args.num_blocks}_' \
                        f'regularization={args.regularization}_' \
                     f'hidden_dim={args.dimPosEmb}_' \
                     f'k1x={args.kernel1_x_Time}_' \
                     f'k1y={args.kernel1_y_Pose}_' \
                     f'channels_conv_blocks={args.channels_conv_blocks}_' \
                     f'encoder_n_harmonic_functions={args.encoder_n_harmonic_functions}_' \
                     f'encoder_omega0={args.encoder_omega0}_'

        train_loss_list, val_loss_list, test_loss_list, metrics_dict = train_autoreg_mixer_ais.train(model, model_name, args)

        # save gif of the predictions
        if args.loss_type == 'mpjpe':
            train_autoreg_mixer_ais.test_mpjpe(model, args, model_name, save_results=True)

        # IMPORTANT: we will optimize val_loss, and report train_loss and test_loss
        trial.set_user_attr(f"train_loss_{loss_type}", train_loss_list[-1].item())
        trial.set_user_attr(f"val_loss_{loss_type}", val_loss_list[-1].item())
        trial.set_user_attr(f"test_loss_{loss_type}", test_loss_list[-1].item())

        # save metrics
        for metric_name, metric_value in metrics_dict.items():
            trial.set_user_attr(metric_name, metric_value[-1].item())

        # evaluate on each action separately
        for action in tqdm([
                '2021-08-04-singlePerson_000',
                '2021-08-04-singlePerson_001',
                '2021-08-04-singlePerson_002',
                '2021-08-04-singlePerson_003',
                '2022-05-26_2persons_000',
                '2022-05-26_2persons_001',
                '2022-05-26_2persons_002',
                '2022-05-26_2persons_003'
            ], desc='evaluating metrics on each action'):

            args_action = copy.deepcopy(args)
            args_action.actions_to_consider = action
            print(f'evaluating metrics on action {action}')

            if args.loss_type == 'mpjpe':
                action_mpjpe_loss, action_auc_pck = train_autoreg_mixer_ais.test_mpjpe(model, args_action, model_name,
                                                                                save_results=True)
                trial.set_user_attr(f"{action}/mpjpe", action_mpjpe_loss.item())
                trial.set_user_attr(f"{action}/auc_pck", action_auc_pck.item())

        return test_loss_list[-1].item()

    def __call__(self, trial):

        args = self.parse_args()
        args, trial = self.overwrite_optuna_params(args, trial)

        if args.dataset_type == 'h36m':
            test_loss_mpjpe = self.train_model_with_loss(args, trial, loss_type='mpjpe', pose_dim=66)
            test_loss_angle = self.train_model_with_loss(args, trial, loss_type='angle', pose_dim=48)
            return test_loss_mpjpe, test_loss_angle
        else:
            test_loss_mpjpe = self.train_model_ais(args, trial, loss_type='mpjpe', pose_dim=33)
            # test_loss_mpjpe = self.train_model_ais(args, trial, loss_type='mpjpe', pose_dim=57)
            return test_loss_mpjpe

    def overwrite_optuna_params(self, args, trial):
        args.dimPosEmb = trial.suggest_int('dimPosEmb', 192, 192, step=32)
        args.channels_conv_blocks = trial.suggest_int('channels_conv_blocks', 4, 4, step=4)
        args.kernel1_x_Time = trial.suggest_int('kernel1_x_Time', 1, 9, step=4)
        args.kernel1_y_Pose = trial.suggest_int('kernel1_y_Pose', 1, 9, step=4)
        args.num_blocks = trial.suggest_int('num_blocks', 6, 6, step=2)

        # disabled
        # args.encoder_n_harmonic_functions = trial.suggest_categorical('encoder_n_harmonic_functions', [0])
        # args.encoder_omega0 = trial.suggest_categorical('encoder_omega0', [0.1])
        return args, trial


if __name__ == '__main__':

    if USER_NAME == "a":
        base_folder = f'/home/azhuavlev/Desktop/Results/CUDA_lab/Final_project/studies'
    elif USER_NAME == "v":
        base_folder = f'/home/user/bornhaup/FinalProject/MotionMixerConv/studies'
    else:
        raise ValueError('User not supported')
    # study_name = 'h36m_reg=-1_out_nTP=10_skip=1_onlyKernels_once'
    study_name = 'h36m_twice_autoregressive_teacher5'
    # study_name = 'h36m_mlp_twice'

    study_path = base_folder + '/' + study_name
    if os.path.exists(study_path):
        # clear the folder
        print('Study directory already exists:', study_path)
        # shutil.rmtree(study_path)
    else:
        os.makedirs(study_path, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{base_folder}/{study_name}/results.db",
        directions=["minimize", "minimize"],
        # directions=["minimize"],
        load_if_exists=True,
        sampler=optuna.samplers.BruteForceSampler()
    )
    # To use the dashboard, run the following command:
    # optuna-dashboard sqlite:////home/azhuavlev/Desktop/Results/CUDA_lab/Final_project/studies/h36m_reg=-1_out_nTP=10_skip=1_fullSpace_testMetrics/results.db
    # respectively: optuna-dashboard sqlite:////home/user/bornhaup/FinalProject/MotionMixerConv/studies/example-study_out_nTP=20/results.db
    # then: ssh -L 8080:127.0.0.1:8080 cuda4

    study.optimize(
        Objective(f'{base_folder}/{study_name}'),
        # direction="minimize",
        n_trials=40,
        timeout=60 * 60 * 47,  # 47 hours,
        catch=(Exception,),
    )
    print('Number of finished trials:', len(study.trials))
    print('To use the dashboard, run the following command:')
    print(f'optuna-dashboard sqlite:////home/azhuavlev/Desktop/Results/CUDA_lab/Final_project/studies/{study_name}/results.db')
