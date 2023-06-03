# -*- coding: utf-8 -*-
import os
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt; plt.style.use(['science']); plt.rcParams.update({'font.size':16})
from multiprocessing import Process
import warnings; warnings.simplefilter('ignore')

import models
from methods import *
from util import set_cpu_num, seed_everything

    
def ID_subworker(args, tau, random_seed=729, is_print=False):
    '''
    A sub-pipeline of Time Scale Selection, with given param: tau and random_seed.

    Args:
        args (argparse): Namespace of ArgumentParser
        tau (float): time step for time-lagged autoencoder
        random_seed (int): random seed
        is_print (bool): whether print log to terminal

    Returns: None
    '''
    
    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(args.cpu_num)
    
    # train
    train_time_lagged(args.system, tau, args.id_epoch, is_print, random_seed, 
                      args.data_dir, args.id_log_dir, args.device, args.embed_dim)

    # test and calculating ID
    test_and_save_embeddings_of_time_lagged(args.system, tau, args.id_epoch, None, is_print, random_seed, 
                                            args.data_dir, args.id_log_dir, args.device, args.embed_dim)
    test_and_save_embeddings_of_time_lagged(args.system, tau, args.id_epoch, args.id_log_dir+f"tau_{tau}/seed{random_seed}", 
                                            is_print, random_seed, args.data_dir, args.id_log_dir, args.device, args.embed_dim)

 
def learn_subworker(args, n, random_seed=729, is_print=False, mode='train'):
    '''
    A sub-pipeline of Slow-Fast Dynamics Learning, with given param: random_seed.

    Args:
        args (argparse): Namespace of ArgumentParser
        n (int): tau_s / tau_1
        random_seed (int): random seed
        is_print (bool): whether print log to terminal
        mode (str): train or test

    Returns: None
    '''
    
    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(args.cpu_num)

    if mode == 'train':
        # train
        ckpt_path = args.id_log_dir + f'tau_{args.tau_s}/seed1/checkpoints/epoch-{args.id_epoch}.ckpt'
        train_slow_extract_and_evolve(args.system, args.tau_s, args.slow_dim, args.koopman_dim, args.tau_1, n, ckpt_path, is_print, random_seed, 
                                      args.learn_epoch, args.data_dir, args.learn_log_dir, args.device, args.alpha, args.embed_dim, args.fast)
    elif mode == 'test':
        # warm up
        test_evolve(args.system, args.tau_s, args.learn_epoch, args.slow_dim, args.koopman_dim, round(args.tau_1*1, 3), 1, is_print, random_seed, 
                    args.data_dir, args.learn_log_dir, args.device, args.embed_dim, args.fast)
        # test evolve
        for i in tqdm(range(1, 50+1)):
            delta_t = round(args.tau_1*i, 3)
            if args.system == '2S2F':
                MSE, RMSE, MAE, MAPE, c1_mae, c2_mae, duration = test_evolve(args.system, args.tau_s, args.learn_epoch, args.slow_dim, args.koopman_dim, delta_t, i, 
                                                                             is_print, random_seed, args.data_dir, args.learn_log_dir, args.device, args.embed_dim, args.fast)
                with open(args.result_dir+f'ours_evolve_test_{args.tau_s}.txt','a') as f:
                    f.writelines(f'{delta_t}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}, {c1_mae}, {c2_mae}, {duration}\n')
            elif args.system == '1S2F':
                MSE, RMSE, MAE, MAPE, duration = test_evolve(args.system, args.tau_s, args.learn_epoch, args.slow_dim, args.koopman_dim, delta_t, i, 
                                                             is_print, random_seed, args.data_dir, args.learn_log_dir, args.device, args.embed_dim, args.fast)
                with open(args.result_dir+f'ours_evolve_test_{args.tau_s}.txt','a') as f:
                    f.writelines(f'{delta_t}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}, {duration}\n')
    else:
        raise TypeError(f"Wrong mode of {mode}!")


def baseline_subworker(args, is_print=False, random_seed=729, mode='train'):
    '''
    A sub-pipeline of Baseline algorithms, with given param: random_seed.

    Args:
        args (argparse): Namespace of ArgumentParser
        random_seed (int): random seed
        is_print (bool): whether print log to terminal
        mode (str): train or test

    Returns: None
    '''

    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(1)

    if args.system == '2S2F':
        if args.model == 'lstm':
            model = models.LSTM(in_channels=1, feature_dim=4)
        elif args.model == 'tcn':
            model = models.TCN(input_size=4, output_size=4, num_channels=[32,16,8], kernel_size=3, dropout=0.1)
        elif args.model == 'neural_ode':
            model = models.NeuralODE(in_channels=1, feature_dim=4)
    elif args.system == '1S2F':
        if args.model == 'lstm':
            model = models.LSTM(in_channels=1, feature_dim=3)
        elif args.model == 'tcn':
            model = models.TCN(input_size=3, output_size=3, num_channels=[32,16,8], kernel_size=3, dropout=0.1)
        elif args.model == 'neural_ode':
            model = models.NeuralODE(in_channels=1, feature_dim=3)
        
    if mode == 'train':
        # train
        baseline_train(model, args.tau_s, args.tau_1, is_print, random_seed, 
                       args.baseline_epoch, args.data_dir, args.baseline_log_dir, args.device)
    else:
        # test evolve
        for i in tqdm(range(1, 50 + 1)):
            delta_t = round(args.tau_1*i, 3)
            if args.system == '2S2F':
                MSE, RMSE, MAE, MAPE, c1_mae, c2_mae, duration = baseline_test(model, args.system, args.tau_s, args.baseline_epoch, 
                                                                               delta_t, i, random_seed, args.data_dir, args.baseline_log_dir, args.device)
                with open(args.result_dir+f'{args.model}_evolve_test_{args.tau_s}.txt','a') as f:
                    f.writelines(f'{delta_t}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}, {c1_mae}, {c2_mae}, {duration}\n')
            elif args.system == '1S2F':
                MSE, RMSE, MAE, MAPE, duration = baseline_test(model, args.system, args.tau_s, args.baseline_epoch, 
                                                               delta_t, i, random_seed, args.data_dir, args.baseline_log_dir, args.device)
                with open(args.result_dir+f'{args.model}_evolve_test_{args.tau_s}.txt','a') as f:
                    f.writelines(f'{delta_t}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}, {duration}\n')
    

def Data_Generate(args):
    '''
    Data generation: simulation, processing, integration.

    Args:
        args (argparse): Namespace of ArgumentParser

    Returns: None
    '''
    
    # generate original data
    print('Generating original simulation data')
    generate_original_data(args.trace_num, args.total_t, args.dt, save=True, plot=False, parallel=args.parallel)

    # generate dataset for ID estimating
    print('Generating training data for ID estimating')
    for tau in np.arange(0., args.tau_N+args.tau_1, args.tau_1):
        tau = round(tau, 2)
        generate_dataset(args.trace_num, tau, None, is_print=True)
    
    # generate dataset for learning fast-slow dynamics
    print('Generating training data for learning fast-slow dynamics')
    n = int(args.tau_s/args.tau_1)
    generate_dataset(args.trace_num, args.tau_1, None, True, n) # traning data
    for i in range(1, 50+1):
        delta_t = round(args.tau_1*i, 3)
        generate_dataset(args.trace_num, delta_t, None, True) # testing data
    
    
def ID_Estimate(args):
    '''
    General process: serial or parallel execution of 'Time Scale Selection' with different given parameters.

    Args:
        args (argparse): Namespace of ArgumentParser

    Returns: None
    '''
    
    print('Estimating the ID per tau')
    
    # id estimate process
    T = np.arange(0., args.tau_N+args.tau_1, args.tau_1)
    workers = []
    for tau in T:
        tau = round(tau, 2)
        for random_seed in range(1, args.seed_num+1):
            if args.parallel: # multi-process to speed-up
                is_print = True if len(workers)==0 else False
                workers.append(Process(target=ID_subworker, args=(args, tau, random_seed, is_print), daemon=True))
                workers[-1].start()
            else:
                ID_subworker(args, tau, random_seed, True)
    # block
    while args.parallel and any([sub.exitcode==None for sub in workers]):
        pass

    # plot ID curve
    [plot_epoch_test_log(round(tau,2), args.id_epoch+1, args.embed_dim) for tau in T]
    plot_id_per_tau(T, np.arange(args.id_epoch-10, args.id_epoch+1, 1), args.embed_dim)
    
    if 'cuda' in args.device: torch.cuda.empty_cache()
    print('\nID Estimate Over')


def Learn_Slow_Fast(args, mode='train'):
    '''
    General process: serial or parallel execution of 'Slow-Fast Dynamics Learning' with different given parameters.

    Args:
        args (argparse): Namespace of ArgumentParser

    Returns: None
    '''
    
    os.makedirs(args.result_dir, exist_ok=True)
    print(f'{mode.capitalize()} the learning of slow and fast dynamics')
    
    # slow evolve sub-process
    n = int(args.tau_s/args.tau_1)
    workers = []
    for random_seed in range(1, args.seed_num+1):
        if args.parallel:
            is_print = True if len(workers)==0 else False
            workers.append(Process(target=learn_subworker, args=(args, n, random_seed, is_print, mode), daemon=True))
            workers[-1].start()
        else:
            learn_subworker(args, n, random_seed, True, mode)
    # block
    while args.parallel and any([sub.exitcode==None for sub in workers]):
        pass
    
    if 'cuda' in args.device: torch.cuda.empty_cache()
    print('\nSlow-Fast Evolve Over')


def Baseline(args, mode='train'):
    '''
    General process: serial or parallel execution of Baseline algorithms with different given parameters.

    Args:
        args (argparse): Namespace of ArgumentParser

    Returns: None
    '''

    os.makedirs(args.result_dir, exist_ok=True)
    print(f'Running the {args.model.upper()}')

    workers = []
    for random_seed in range(1, args.seed_num+1):
        if args.parallel:
            is_print = True if len(workers)==0 else False
            workers.append(Process(target=baseline_subworker, args=(args, is_print, random_seed, mode), daemon=True))
            workers[-1].start()
        else:
            baseline_subworker(args, True, random_seed, mode)
    # block
    while args.parallel and any([sub.exitcode==None for sub in workers]):
        pass

    if 'cuda' in args.device: torch.cuda.empty_cache()
    print(f'{args.model.upper()} running Over')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ours', help='Model: [ours, lstm, tcn, neural_ode]')
    parser.add_argument('--fast', type=int, default=1, help='Whether to learn fast dynamics')
    parser.add_argument('--system', type=str, default='2S2F', help='Dynamical System: [1S2F, 2S2F]')
    parser.add_argument('--phase', type=str, default='TimeSelection', help='Phase of whole pipeline: TimeSelection or LearnDynamics')
    parser.add_argument('--trace_num', type=int, default=200, help='Number of simulation trajectories')
    parser.add_argument('--total_t', type=float, default=5.1, help='Time length of each simulation trajectories')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step of each simulation trajectories')
    parser.add_argument('--tau_1', type=float, default=0.1, help='params for ID-driven Time Scale Selection')
    parser.add_argument('--tau_N', type=float, default=3.0, help='params for ID-driven Time Scale Selection')
    parser.add_argument('--tau_s', type=float, default=0.8, help='Approprate time scale for fast-slow separation')
    parser.add_argument('--slow_dim', type=int, default=2, help='Intrinsic dimension of slow dynamics')
    parser.add_argument('--koopman_dim', type=int, default=4, help='Dimension of Koopman invariable space')
    parser.add_argument('--id_epoch', type=int, default=100, help='Max training epoch of ID-driven Time Scale Selection')
    parser.add_argument('--learn_epoch', type=int, default=100, help='Max training epoch of Fast-Slow Learning')
    parser.add_argument('--alpha', type=float, default=0.1, help='Penalty coefficient of slow extracting loss')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension of Encoder_1')
    parser.add_argument('--baseline_epoch', type=int, default=100, help='Max training epoch of Baseline Algorithm')
    parser.add_argument('--seed_num', type=int, default=10, help='Multiple random seed for average')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--cpu_num', type=int, default=-1, help='Limit the available cpu number of each sub-processing, -1 means no limit')
    parser.add_argument('--parallel', help='Parallel running the whole pipeline by multi-processing', action='store_true')
    parser.add_argument('--data_dir', type=str, default='Data/2S2F/data/')
    parser.add_argument('--id_log_dir', type=str, default='logs/2S2F/TimeSelection/')
    parser.add_argument('--learn_log_dir', type=str, default='logs/2S2F/LearnDynamics/')
    parser.add_argument('--baseline_log_dir', type=str, default='logs/2S2F/LSTM/')
    parser.add_argument('--result_dir', type=str, default='Results/2S2F/')
    args = parser.parse_args()

    if args.system == '2S2F':
        from Data.generator_2s2f import generate_dataset, generate_original_data
        from util.plot_2s2f import plot_epoch_test_log, plot_id_per_tau
    elif args.system == '1S2F':
        from Data.generator_1s2f import generate_dataset, generate_original_data
        from util.plot_1s2f import plot_epoch_test_log, plot_id_per_tau

    if not args.parallel and args.cpu_num==1:
        print('Not recommand to limit the cpu num when non-parallellism!')
    
    # main pipeline
    Data_Generate(args)

    if args.model == 'ours':
        if args.phase == 'TimeSelection':
            ID_Estimate(args)
        elif args.phase == 'LearnDynamics':
            Learn_Slow_Fast(args, 'train')
            Learn_Slow_Fast(args, 'test')
    else:
        Baseline(args, 'train')
        Baseline(args, 'test')
