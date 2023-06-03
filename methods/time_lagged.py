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
from util.intrinsic_dimension import eval_id_embedding


def train_time_lagged(
        system,
        tau, 
        max_epoch, 
        is_print=False, 
        random_seed=729, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/TimeSelection/', 
        device=torch.device('cpu'),
        embed_dim=64
        ):
    
    # prepare
    data_filepath = data_dir + 'tau_' + str(tau)
    log_dir = log_dir + 'tau_' + str(tau) + f'/seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)
    
    # init model
    if system == '2S2F':
        model = models.TimeLaggedAE(in_channels=1, feature_dim=4, embed_dim=embed_dim)
    elif system == '1S2F':
        model = models.TimeLaggedAE(in_channels=1, feature_dim=3, embed_dim=embed_dim)
    model.apply(models.weights_normal_init)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    model.to(device)
    
    # training params
    lr = 0.001
    batch_size = 128
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()

    # dataset
    train_dataset = Dataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = Dataset(data_filepath, 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    losses = []
    loss_curve = []
    for epoch in range(1, max_epoch+1):
        
        # train
        model.train()
        for input, target in train_loader:
            input = model.scale(input.to(device)) # (batchsize,1,channel_num,feature_dim)
            target = model.scale(target.to(device))
            
            output, _ = model.forward(input)
            
            loss = loss_func(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.detach().item())
            
        loss_curve.append(np.mean(losses))
        
        # validate
        with torch.no_grad():
            targets = []
            outputs = []
            
            model.eval()
            for input, target in val_loader:
                input = model.scale(input.to(device))
                target = model.scale(target.to(device))
            
                output, _ = model.forward(input)
                outputs.append(output.cpu())
                targets.append(target.cpu())
                
            targets = torch.concat(targets, axis=0)
            outputs = torch.concat(outputs, axis=0)
            mse = loss_func(outputs, targets)
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] val-MSE={mse:.5f}', end='')
        
        # save each epoch model
        model.train()
        torch.save({'model': model.state_dict(), 'encoder': model.encoder.state_dict(),}, log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
        
    # plot loss curve
    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel('epoch')
    plt.title('Train MSELoss Curve')
    plt.savefig(log_dir+'/loss_curve.pdf', dpi=300)
    np.save(log_dir+'/loss_curve.npy', loss_curve)
        

def test_and_save_embeddings_of_time_lagged(
        system,
        tau, 
        max_epoch, 
        checkpoint_filepath=None, 
        is_print=False, 
        random_seed=729, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/TimeSelection/', 
        device=torch.device('cpu'),
        embed_dim=64
        ):
    
    # prepare
    data_filepath = data_dir + 'tau_' + str(tau)
    
    # testing params
    batch_size = 128
    loss_func = nn.MSELoss()
    
    # init model
    if system == '2S2F':
        model = models.TimeLaggedAE(in_channels=1, feature_dim=4, embed_dim=embed_dim)
    elif system == '1S2F':
        model = models.TimeLaggedAE(in_channels=1, feature_dim=3, embed_dim=embed_dim)
    if checkpoint_filepath is None: # not trained
        model.apply(models.weights_normal_init)
        model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
        model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)

    # dataset
    test_dataset = Dataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # testing pipeline
    fp = open(log_dir + 'tau_' + str(tau) + '/test_log.txt', 'a')
    for ep in range(1,max_epoch):
        
        # load weight file
        epoch = ep
        if checkpoint_filepath is not None:
            epoch = ep + 1
            ckpt_path = checkpoint_filepath + f"/checkpoints/" + f'epoch-{epoch}.ckpt'
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
        model = model.to(device)
        model.eval()
        
        all_embeddings = []
        test_outputs = np.array([])
        test_targets = np.array([])
        var_log_dir = log_dir + 'tau_' + str(tau) + f'/seed{random_seed}/test/epoch-{epoch}'
        os.makedirs(var_log_dir, exist_ok=True)
        
        # testing
        with torch.no_grad():
            for input, target in test_loader:
                input = model.scale(input.to(device)) # (batchsize,1,1,4)
                target = model.scale(target.to(device))
                
                output, embeddings = model.forward(input)
                
                # save the embedding vectors
                for embedding in embeddings:
                    all_embeddings.append(embedding.cpu().numpy())

                test_outputs = output.cpu() if not len(test_outputs) else torch.concat((test_outputs, output.cpu()), axis=0)
                test_targets = target.cpu() if not len(test_targets) else torch.concat((test_targets, target.cpu()), axis=0)
                                
            # test mse
            mse_ = []
            for i in range(test_outputs.shape[-1]):
                mse_.append(loss_func(test_outputs[:,0,0,i], test_targets[:,0,0,i]))
        
        # plot
        plt.figure(figsize=(16,5))
        for j in range(test_outputs.shape[-1]):
            data = []
            for i in range(len(test_outputs)):
                data.append([test_outputs[i,0,0,j], test_targets[i,0,0,j]])
            ax = plt.subplot(1,test_outputs.shape[-1],j+1)
            ax.set_title('test_'+f'c{j+1}')
            ax.plot(np.array(data)[:,1], label='true')
            ax.plot(np.array(data)[:,0], label='predict')
            ax.legend()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
        plt.savefig(var_log_dir+"/result.pdf", dpi=300)
        plt.close()

        # save embedding
        np.save(var_log_dir+'/embedding.npy', all_embeddings)
        
        # calculae ID
        def cal_id_embedding(method='MLE', is_print=False):
            # eval_id_embedding(var_log_dir, method=method, is_print=is_print, max_point=100)
            eval_id_embedding(var_log_dir, method=method, is_print=False, max_point=100)
            dims = np.load(var_log_dir+f'/id_{method}.npy')
            return np.mean(dims)
        MLE_id = cal_id_embedding('MLE', is_print)

        # logging
        if system == '2S2F':
            fp.write(f"{tau},{random_seed},{mse_[0]},{mse_[1]},{mse_[2]},{mse_[3]},{epoch},{MLE_id}\n")
        elif system == '1S2F':
            fp.write(f"{tau},{random_seed},{mse_[0]},{mse_[1]},{mse_[2]},{epoch},{MLE_id}\n")
        fp.flush()

        if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}] | MSE: {loss_func(test_outputs, test_targets):.6f} | MLE={MLE_id:.1f}   ', end='')
        
        if checkpoint_filepath is None: break
        
    fp.close()
