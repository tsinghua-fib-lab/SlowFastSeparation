# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
import scienceplots
import matplotlib.pyplot as plt; plt.style.use(['science']); plt.rcParams.update({'font.size':16})
from Data.dataset import Dataset


def baseline_train(
        model,
        tau_s, 
        tau_1, 
        is_print=False, 
        random_seed=729,
        max_epoch=50,
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/lstm/', 
        device='cpu'
        ):
        
    # prepare
    data_filepath = data_dir + 'tau_' + str(tau_1)
    log_dir = log_dir + f'tau_{tau_s}/seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    model.to(device)
    
    # training params
    lr = 0.001
    batch_size = 128
    weight_decay = 0.001
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # dataset
    train_dataset = Dataset(data_filepath, 'train', length=int(tau_s/tau_1))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = Dataset(data_filepath, 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    train_loss = []
    best_loss = 1.
    for epoch in range(1, max_epoch+1):
        
        losses = []
        
        # train
        model.train()
        counter = 0
        for input, _, internl_units in train_loader:
            counter += 1
            
            input = model.scale(input.to(device)) # (batchsize, 1, channel_num, feature_dim)

            loss = 0
            t = [0.]
            for i in range(1, len(internl_units)):
                
                unit = model.scale(internl_units[i].to(device)) # t+i
                
                if model.__class__.__name__ == 'LSTM':
                    output = model(input, device)
                    for _ in range(1, i):
                        output = model(output, device)
                    loss += MSE_loss(output, unit)
                elif model.__class__.__name__ == 'TCN':
                    output = model(input)
                    for _ in range(1, i):
                        input = torch.concat([input[:,1:], output[:,0].unsqueeze(1)], dim=1)
                        output = model(input)
                    loss += MSE_loss(output, unit)
                elif model.__class__.__name__ == 'NeuralODE':
                    t = torch.tensor([0., tau_1], device=device)
                    output = model(input, t)[:, -1:]
                    for _ in range(1, i):
                        output = model(output, t)[:, -1:]
                    loss += MSE_loss(output, unit)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # record loss
            losses.append(loss.detach().item())

        train_loss.append(np.mean(losses[0]))
        
        # validate
        with torch.no_grad():
            targets = []
            outputs = []
            
            model.eval()
            for input, target in val_loader:
                
                input = model.scale(input.to(device)) # (batchsize,1,channel_num,feature_dim)
                target = model.scale(target.to(device))
                
                if model.__class__.__name__ == 'LSTM':
                    output = model(input, device)
                elif model.__class__.__name__ == 'TCN':
                    output = model(input)
                elif model.__class__.__name__ == 'NeuralODE':
                    t = torch.tensor([0., tau_s-tau_1], device=device)
                    output = model(input, t)[:, -1:]

                # record results
                outputs.append(output.cpu())
                targets.append(target.cpu())
            
            # trans to tensor
            outputs = torch.concat(outputs, axis=0)
            targets = torch.concat(targets, axis=0)
            
            # cal loss
            loss = MSE_loss(outputs, targets)
            if is_print: print(f'\rTau[{tau_s}] | epoch[{epoch}/{max_epoch}] | val-mse={loss:.5f}', end='')
                        
            # plot per epoch
            if epoch % 1 == 0:
                
                os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)
                
                # plot total infomation one-step prediction curve
                plt.figure(figsize=(16,4))
                for j, item in enumerate([f'c{i+1}' for i in range(outputs.shape[-1])]):
                    ax = plt.subplot(1,outputs.shape[-1],j+1)
                    ax.set_title(item)
                    plt.plot(targets[:,0,0,j], label='true')
                    plt.plot(outputs[:,0,0,j], label='predict')
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/predict.pdf", dpi=300)
                plt.close()
        
            # record best model
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()

    # save model
    torch.save(best_model, log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    if is_print: print(f'\nsave best model at {log_dir}/checkpoints/epoch-{epoch}.ckpt (val best_loss={best_loss})')
    
    # plot loss curve
    train_loss = np.array(train_loss)
    plt.figure()
    plt.plot(train_loss)
    plt.xlabel('epoch')
    plt.title('Training Loss Curve')
    plt.savefig(log_dir+'/train_loss_curve.pdf', dpi=300)


def baseline_test(
        model, 
        system,
        tau_s, 
        ckpt_epoch,
        delta_t, 
        n, 
        random_seed=729, 
        data_dir='Data/2S2F/data/', 
        log_dir='logs/2S2F/lstm/', 
        device='cpu'
        ):
        
    # prepare
    data_filepath = data_dir + 'tau_' + str(delta_t)
    log_dir = log_dir + f'tau_{tau_s}/seed{random_seed}'
    os.makedirs(log_dir+f"/test/", exist_ok=True)

    # load model
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    # dataset
    test_dataset = Dataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=False)
    
    # testing pipeline        
    with torch.no_grad():

        # timing
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        duration = 0

        model.eval()
        for input, target in test_loader:

            starter.record()

            input = model.scale(input.to(device))
            target = model.scale(target.to(device))

            if model.__class__.__name__ == 'LSTM':
                output = model(input, device)
                for _ in range(1, n):
                    output = model(output, device)
            elif model.__class__.__name__ == 'TCN':
                output = model(input)
                for _ in range(1, n):
                    input = torch.concat([input[:,1:], output[:,0].unsqueeze(1)], dim=1)
                    output = model(input)
            elif model.__class__.__name__ == 'NeuralODE':
                t = torch.tensor([0., delta_t/n], dtype=torch.float32, device=device)
                output = model(input, t)[:, -1:]
                for _ in range(1, n):
                    output = model(output, t)[:, -1:]

            ender.record()
            torch.cuda.synchronize()  # wait GPU task finished
            duration = starter.elapsed_time(ender) / len(test_dataset)

        targets = target
        outputs = output
        
    # metrics
    pred = outputs.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MAPE = np.mean(np.abs((pred - true) / true))
    targets = model.descale(targets)
    outputs = model.descale(outputs)
    pred = outputs.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MSE = np.mean((pred - true) ** 2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(pred - true))
    
    # plot total infomation prediction curve
    plt.figure(figsize=(16,4))
    for j, item in enumerate([f'c{i+1}' for i in range(outputs.shape[-1])]):
        ax = plt.subplot(1,outputs.shape[-1],j+1)
        ax.set_title(item)
        plt.plot(true[:,0,0,j], label='true')
        plt.plot(pred[:,0,0,j], label='predict')
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(log_dir+f"/test/predict_{delta_t}.pdf", dpi=300)
    plt.close()

    if system == '2S2F':
        c1_evolve_mae = np.mean(np.abs(pred[:,0,0,0] - true[:,0,0,0]))
        c2_evolve_mae = np.mean(np.abs(pred[:,0,0,1] - true[:,0,0,1]))
        return MSE, RMSE, MAE, MAPE, c1_evolve_mae, c2_evolve_mae, duration
    elif system == '1S2F':
        return MSE, RMSE, MAE, MAPE, duration