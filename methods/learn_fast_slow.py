# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import numpy as np
import scienceplots
import matplotlib.pyplot as plt; plt.style.use(['science']); plt.rcParams.update({'font.size':16})
import warnings;warnings.simplefilter('ignore')

import models
from Data.dataset import Dataset
    
    
def train_slow_extract_and_evolve(
        system,
        tau_s, 
        slow_dim, 
        koopman_dim, 
        delta_t, 
        n, 
        ckpt_path,
        is_print=False, 
        random_seed=729, 
        learn_max_epoch=100, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/LearnDynamics/', 
        device='cpu',
        alpha=0.1,
        embed_dim=64,
        fast=True
        ):
        
    # prepare
    data_filepath = data_dir + 'tau_' + str(delta_t)
    log_dir = log_dir + f'seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    assert koopman_dim>=slow_dim, f"Value Error, koopman_dim is smaller than slow_dim({koopman_dim}<{slow_dim})"
    if system == '2S2F':
        model = models.DynamicsEvolver(in_channels=1, feature_dim=4, embed_dim=embed_dim, slow_dim=slow_dim, 
                                       redundant_dim=koopman_dim-slow_dim, tau_s=tau_s, fast=fast, device=device)
    elif system == '1S2F':
        model = models.DynamicsEvolver(in_channels=1, feature_dim=3, embed_dim=embed_dim, slow_dim=slow_dim, 
                                       redundant_dim=koopman_dim-slow_dim, tau_s=tau_s, fast=fast, device=device)
    model.apply(models.weights_normal_init)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    
    # load pretrained time-lagged AE
    ckpt = torch.load(ckpt_path)
    model.encoder_1.load_state_dict(ckpt['encoder'])
    model = model.to(device)
    
    # training params
    lr = 0.001
    batch_size = 128
    weight_decay = 0.001
    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [{'params': model.encoder_2.parameters()},
         {'params': model.encoder_3.parameters()},
         {'params': model.decoder.parameters()}, 
         {'params': model.K_opt.parameters()},
         {'params': model.lstm.parameters()}],
        lr=lr, weight_decay=weight_decay) # not involve encoder_1 (freezen)
    
    # dataset
    train_dataset = Dataset(data_filepath, 'train', length=n)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = Dataset(data_filepath, 'val', length=n)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    train_loss = []
    val_loss = []
    lambda_curve = [[] for _ in range(slow_dim)]
    for epoch in range(1, learn_max_epoch+1):
        
        losses = [[],[],[],[]]
        
        # train
        model.train()
        [lambda_curve[i].append(model.K_opt.Lambda[i].detach().cpu()) for i in range(slow_dim) ]
        for input, _, internl_units in train_loader:
            
            input = model.scale(input.to(device)) # (batchsize,1,1,4)
            
            ###################################################
            # obs —— slow representation —— obs(reconstruction)
            #         |
            #      koopman
            ###################################################
            # obs ——> slow ——> koopman
            slow_var, embed = model.obs2slow(input)
            koopman_var = model.slow2koopman(slow_var)
            slow_obs = model.slow2obs(slow_var)
            _, embed_from_obs = model.obs2slow(slow_obs)
            
            adiabatic_loss = L1_loss(embed, embed_from_obs)
            slow_reconstruct_loss = MSE_loss(slow_obs, input)

            ################
            # n-step evolve
            ################
            fast_obs = input - slow_obs.detach()
            obs_evol_loss, slow_evol_loss, koopman_evol_loss = 0, 0, 0
            for i in range(1, len(internl_units)):
                
                unit = model.scale(internl_units[i].to(device)) # t+i
                
                #######################
                # slow component evolve
                #######################
                # obs ——> slow ——> koopman
                unit_slow_var, _ = model.obs2slow(unit)
                unit_koopman_var = model.slow2koopman(unit_slow_var)

                # slow evolve
                t = torch.tensor([delta_t * i], device=device) # delta_t
                unit_koopman_var_next = model.koopman_evolve(koopman_var, tau=t) # t ——> t + i*delta_t
                unit_slow_var_next = model.koopman2slow(unit_koopman_var_next)

                # koopman ——> slow ——> obs
                unit_slow_obs_next = model.slow2obs(unit_slow_var_next)
                
                #######################
                # fast component evolve
                #######################
                # fast obs evolve
                unit_fast_obs_next, _ = model.lstm_evolve(fast_obs, T=i) # t ——> t + i*delta_t
                
                ################
                # calculate loss
                ################
                # total obs evolve
                unit_obs_next = unit_slow_obs_next + unit_fast_obs_next
                
                # evolve loss
                koopman_evol_loss += MSE_loss(unit_koopman_var_next, unit_koopman_var)
                slow_evol_loss += MSE_loss(unit_slow_var_next, unit_slow_var)
                obs_evol_loss += MSE_loss(unit_obs_next, unit)
            
            ###########
            # optimize
            ###########
            all_loss = (slow_reconstruct_loss + alpha*adiabatic_loss) + (koopman_evol_loss + slow_evol_loss + obs_evol_loss) / n
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            
            # record loss
            losses[0].append(adiabatic_loss.detach().item())
            losses[1].append(slow_reconstruct_loss.detach().item())
            losses[2].append(slow_evol_loss.detach().item())
            losses[3].append(obs_evol_loss.detach().item())
        
        train_loss.append([np.mean(losses[0]), np.mean(losses[1]), np.mean(losses[2]), np.mean(losses[3])])
        
        # validate 
        with torch.no_grad():
            inputs = []
            slow_vars = []
            targets = []
            slow_obses = []
            slow_obses_next = []
            fast_obses = []
            fast_obses_next = []
            total_obses_next = []
            embeds = []
            embed_from_obses = []
            
            model.eval()
            for input, target, _ in val_loader:
                
                input = model.scale(input.to(device)) # (batchsize,1,channel_num,feature_dim)
                target = model.scale(target.to(device))
                
                # obs ——> slow ——> koopman
                slow_var, embed = model.obs2slow(input)
                koopman_var = model.slow2koopman(slow_var)
                slow_obs = model.slow2obs(slow_var)
                _, embed_from_obs = model.obs2slow(slow_obs)
                
                # koopman evolve
                t = torch.tensor([tau_s-delta_t], device=device)
                koopman_var_next = model.koopman_evolve(koopman_var, tau=t)
                slow_var_next = model.koopman2slow(koopman_var_next)
                slow_obs_next = model.slow2obs(slow_var_next)
                
                # fast obs evolve
                fast_obs = input - slow_obs
                fast_obs_next, _ = model.lstm_evolve(fast_obs, T=n)
                
                # total obs evolve
                total_obs_next = slow_obs_next + fast_obs_next

                # record results
                inputs.append(input.cpu())
                slow_vars.append(slow_var.cpu())
                targets.append(target.cpu())
                slow_obses.append(slow_obs.cpu())
                slow_obses_next.append(slow_obs_next.cpu())
                fast_obses.append(fast_obs.cpu())
                fast_obses_next.append(fast_obs_next.cpu())
                total_obses_next.append(total_obs_next.cpu())
                embeds.append(embed.cpu())
                embed_from_obses.append(embed_from_obs.cpu())
            
            # trans to tensor
            inputs = torch.concat(inputs, axis=0)
            slow_vars = torch.concat(slow_vars, axis=0)
            targets = torch.concat(targets, axis=0)
            slow_obses = torch.concat(slow_obses, axis=0)
            slow_obses_next = torch.concat(slow_obses_next, axis=0)
            fast_obses = torch.concat(fast_obses, axis=0)
            fast_obses_next = torch.concat(fast_obses_next, axis=0)
            total_obses_next = torch.concat(total_obses_next, axis=0)
            embeds = torch.concat(embeds, axis=0)
            embed_from_obses = torch.concat(embed_from_obses, axis=0)
            
            # cal loss
            adiabatic_loss = L1_loss(embeds, embed_from_obses)
            slow_reconstruct_loss = MSE_loss(slow_obses, inputs)
            evolve_loss = MSE_loss(total_obses_next, targets)
            all_loss = 0.5*slow_reconstruct_loss + 0.5*evolve_loss + 0.1*adiabatic_loss
            if is_print: print(f'\rTau[{tau_s}] | epoch[{epoch}/{learn_max_epoch}] | val: adiab_loss={adiabatic_loss:.5f}, \
                               recons_loss={slow_reconstruct_loss:.5f}, evol_loss={evolve_loss:.5f}', end='')
            
            val_loss.append(all_loss.detach().item())
            
            # plot per 5 epoch
            if epoch % 5 == 0:
                os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)
                
                # plot slow variable vs input
                plt.figure(figsize=(16,5+2*(slow_dim-1)))
                plt.title('Val Reconstruction Curve')
                for id_var in range(slow_dim):
                    for index, item in enumerate([f'c{k}' for k in range(targets.shape[-1])]):
                        plt.subplot(slow_dim, targets.shape[-1], index+1+targets.shape[-1]*(id_var))
                        plt.scatter(inputs[:,0,0,index], slow_vars[:, id_var], s=5)
                        plt.xlabel(item)
                        plt.ylabel(f'U{id_var+1}')
                plt.subplots_adjust(wspace=0.35, hspace=0.35)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_vs_input.pdf", dpi=300)
                plt.close()
                
                # plot slow variable
                plt.figure(figsize=(12,5+2*(slow_dim-1)))
                plt.title('Slow variable Curve')
                for id_var in range(slow_dim):
                    ax = plt.subplot(slow_dim, 1, 1+id_var)
                    if system == '2S2F':
                        ax.plot(inputs[:,0,0,0], label='c1')
                        ax.plot(inputs[:,0,0,1], label='c2')
                    elif system == '1S2F':
                        ax.plot(inputs[:,0,0,0], label='c1')
                    ax.plot(slow_vars[:, id_var], label=f'U{id_var+1}')
                    plt.xlabel(item)
                    ax.legend()
                plt.subplots_adjust(wspace=0.35, hspace=0.35)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_variable.pdf", dpi=300)
                plt.close()

                # plot observation and prediction
                for i, figname in enumerate(['fast_slow_obs', 'slow_predict', 'fast_predict', 'all_predict']):
                    plt.figure(figsize=(16,5))
                    for j, item in enumerate([f'c{k}' for k in range(targets.shape[-1])]):
                        ax = plt.subplot(1,targets.shape[-1],j+1)

                        # plot fast & slow observation reconstruction curve
                        if i == 0:
                            ax.plot(inputs[:,0,0,j], label='all_obs')
                            ax.plot(slow_obses[:,0,0,j], label='slow_obs')
                            ax.plot(fast_obses[:,0,0,j], label='fast_obs')
                        # plot slow observation one-step prediction curve
                        elif i == 1:
                            ax.plot(targets[:,0,0,j], label='all_true')
                            ax.plot(slow_obses_next[:,0,0,j], label='slow_predict')
                        # plot fast observation one-step prediction curve
                        elif i == 2:
                            ax.plot(targets[:,0,0,j], label='all_true')
                            ax.plot(fast_obses_next[:,0,0,j], label='fast_predict')
                        # plot total observation one-step prediction curve
                        else:
                            ax.plot(targets[:,0,0,j], label='all_true')
                            ax.plot(total_obses_next[:,0,0,j], label='all_predict')
                        
                        ax.set_title(item)
                        ax.legend()
                    plt.subplots_adjust(wspace=0.2)
                    plt.savefig(log_dir+f"/val/epoch-{epoch}/{figname}.pdf", dpi=300)
                    plt.close()
        
                # save model
                torch.save(model.state_dict(), log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    
    # plot loss curve
    train_loss = np.array(train_loss)
    plt.figure()
    for i, item in enumerate(['adiabatic','slow_reconstruct','koopman_evolve','total_evolve']):
        plt.plot(train_loss[:, i], label=item)
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.savefig(log_dir+'/train_loss_curve.pdf', dpi=300)
    np.save(log_dir+'/val_loss_curve.npy', val_loss)

    # plot Koopman Lambda curve
    plt.figure(figsize=(6,6))
    marker = ['o', '^', '+', 's', '*', 'x']
    for i in range(slow_dim):
        plt.plot(lambda_curve[i], marker=marker[i%len(marker)], markersize=6, label=rf'$\lambda_{i}$')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel(r'$\Lambda$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(log_dir+'/K_lambda_curve.pdf', dpi=300)
    np.savez(log_dir+'/K_lambda_curve.npz',lambda_curve=lambda_curve)


def test_evolve(
        system, 
        tau_s, 
        ckpt_epoch, 
        slow_dim,
        koopman_dim, 
        delta_t, 
        n, 
        is_print=False, 
        random_seed=729, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/LearnDynamics/', 
        device='cpu',
        embed_dim=64,
        fast=True
        ):
        
    # prepare
    data_filepath = data_dir + 'tau_' + str(delta_t)
    log_dir = log_dir + f'seed{random_seed}'

    # load model
    if system == '2S2F':
        model = models.DynamicsEvolver(in_channels=1, feature_dim=4, embed_dim=embed_dim, slow_dim=slow_dim, 
                                       redundant_dim=koopman_dim-slow_dim, tau_s=tau_s, fast=fast, device=device)
    elif system == '1S2F':
        model = models.DynamicsEvolver(in_channels=1, feature_dim=3, embed_dim=embed_dim, slow_dim=slow_dim, 
                                       redundant_dim=koopman_dim-slow_dim, tau_s=tau_s, fast=fast, device=device)
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    if is_print and delta_t==0.1:
        print('Koopman V:')
        print(model.K_opt.V.detach().cpu().numpy())
        print('Koopman Lambda:')
        print(model.K_opt.Lambda.detach().cpu().numpy())
    
    # dataset
    test_dataset = Dataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=False)
    
    # testing pipeline        
    with torch.no_grad():
        model.eval()

        # timing
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        duration = 0

        # only one iter
        for input, target in test_loader:

            starter.record()
            
            input = model.scale(input.to(device))
            target = model.scale(target.to(device))
        
            # obs ——> slow ——> koopman
            slow_var, _ = model.obs2slow(input)
            koopman_var = model.slow2koopman(slow_var)
            slow_obs = model.slow2obs(slow_var)
            
            # koopman evolve
            t = torch.tensor([delta_t], device=device)
            koopman_var_next = model.koopman_evolve(koopman_var, tau=t)
            slow_var_next = model.koopman2slow(koopman_var_next)
            slow_obs_next = model.slow2obs(slow_var_next)
            
            # fast obs evolve
            fast_obs = input - slow_obs
            fast_obs_next, _ = model.lstm_evolve(fast_obs, T=n)
            
            # total obs evolve
            total_obs_next = slow_obs_next + fast_obs_next

            ender.record()
            torch.cuda.synchronize()  # wait GPU task finished
            duration = starter.elapsed_time(ender) / len(test_dataset)
        
        inputs = model.descale(input).cpu()
        slow_obses = model.descale(slow_obs).cpu()
        slow_obses_next = model.descale(slow_obs_next).cpu()
        fast_obses_next = model.descale(fast_obs_next).cpu()
        slow_vars = slow_var.cpu()
        targets = target
        total_obses_next = total_obs_next
    
    # metrics
    pred = total_obses_next.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MAPE = np.mean(np.abs((pred - true) / true))
    
    targets = model.descale(targets)
    total_obses_next = model.descale(total_obses_next)
    pred = total_obses_next.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MSE = np.mean((pred - true) ** 2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(pred - true))
                
    os.makedirs(log_dir+f"/test/{delta_t}/", exist_ok=True)

    # plot slow extract from original data
    sample = 10
    plt.figure(figsize=(16,5))
    for j, item in enumerate([rf'$c_{k}$' for k in range(targets.shape[-1])]):
        ax = plt.subplot(1,targets.shape[-1],j+1)
        t = torch.range(0,len(inputs)-1) * 0.01
        ax.plot(t[::sample], inputs[::sample,0,0,j], label=r'$X$')
        ax.plot(t[::sample], slow_obses[::sample,0,0,j], marker="^", markersize=4, label=r'$X_s$')
        ax.legend()
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('t / s', fontsize=20)
        plt.ylabel(item, fontsize=20)
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_extract.pdf", dpi=300)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_extract.jpg", dpi=300)
    plt.close()

    # plot slow variable vs input
    sample = 4
    plt.figure(figsize=(16,5+2*(slow_dim-1)))
    for id_var in range(slow_dim):
        for index, item in enumerate([rf'$c_{k}$' for k in range(targets.shape[-1])]):
            plt.subplot(slow_dim, targets.shape[-1], index+1+targets.shape[-1]*(id_var))
            plt.scatter(inputs[::sample,0,0,index], slow_vars[::sample, id_var], s=2)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel(item, fontsize=20)
            plt.ylabel(rf'$u_{id_var+1}$', fontsize=20)
    plt.subplots_adjust(wspace=0.55, hspace=0.35)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_vs_input.pdf", dpi=150)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_vs_input.jpg", dpi=150)
    plt.close()

    for i, figname in enumerate(['slow_pred', 'fast_pred', 'total_pred']):
        plt.figure(figsize=(16,5))
        for j, item in enumerate([rf'$c_{k}$' for k in range(targets.shape[-1])]):
            ax = plt.subplot(1,targets.shape[-1],j+1)
            
            ax.plot(true[:,0,0,j], label='true')

            # plot slow observation prediction curve
            if i == 0:
                ax.plot(slow_obses_next[:,0,0,j], label='predict')
            # plot fast observation prediction curve
            elif i == 1 and fast:
                ax.plot(fast_obses_next[:,0,0,j], label='predict')
            # plot total observation prediction curve
            elif i == 2:
                ax.plot(pred[:,0,0,j], label='predict')
            
            ax.set_title(item)
            ax.legend()
        plt.subplots_adjust(wspace=0.2)
        plt.savefig(log_dir+f"/test/{delta_t}/{figname}.pdf", dpi=300)
        plt.close()

    if system == '2S2F':
        c1_evolve_mae = torch.mean(torch.abs(slow_obses_next[:,0,0,0] - true[:,0,0,0]))
        c2_evolve_mae = torch.mean(torch.abs(slow_obses_next[:,0,0,1] - true[:,0,0,1]))
        return MSE, RMSE, MAE, MAPE, c1_evolve_mae.item(), c2_evolve_mae.item(), duration
    elif system == '1S2F':
        return MSE, RMSE, MAE, MAPE, duration
