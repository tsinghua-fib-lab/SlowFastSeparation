import numpy as np
import pandas as pd
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt; plt.style.use(['science']); plt.rcParams.update({'font.size':16})


def plot_epoch_test_log(tau, max_epoch, embed_dim):

    class MSE():
        def __init__(self, tau):
            self.tau = tau
            self.mse_c1 = [[] for _ in range(max_epoch)]
            self.mse_c2 = [[] for _ in range(max_epoch)]
            self.mse_c3 = [[] for _ in range(max_epoch)]
            self.mse_c4 = [[] for _ in range(max_epoch)]
            self.MLE_id = [[] for _ in range(max_epoch)]

    fp = open(f'logs/2S2F/TimeSelection/embed_dim_{embed_dim}/tau_{tau}/test_log.txt', 'r')
    items = []
    for line in fp.readlines():
        tau = float(line[:-1].split(',')[0])
        seed = int(line[:-1].split(',')[1])
        mse_c1 = float(line[:-1].split(',')[2])
        mse_c2 = float(line[:-1].split(',')[3])
        mse_c3 = float(line[:-1].split(',')[4])
        mse_c4 = float(line[:-1].split(',')[5])
        epoch = int(line[:-1].split(',')[6])
        MLE_id = float(line[:-1].split(',')[7])

        find = False
        for M in items:
            if M.tau == tau:
                M.mse_c1[epoch].append(mse_c1)
                M.mse_c2[epoch].append(mse_c2)
                M.mse_c3[epoch].append(mse_c3)
                M.mse_c4[epoch].append(mse_c4)
                M.MLE_id[epoch].append(MLE_id)
                find = True
                    
        if not find:
            M = MSE(tau)
            M.mse_c1[epoch].append(mse_c1)
            M.mse_c2[epoch].append(mse_c2)
            M.mse_c3[epoch].append(mse_c3)
            M.mse_c4[epoch].append(mse_c4)
            M.MLE_id[epoch].append(MLE_id)
            items.append(M)
    fp.close()

    for M in items:
        mse_c1_list = []
        mse_c2_list = []
        mse_c3_list = []
        mse_c4_list = []
        MLE_id_list = []
        for epoch in range(max_epoch):
            mse_c1_list.append(np.mean(M.mse_c1[epoch]))
            mse_c2_list.append(np.mean(M.mse_c2[epoch]))
            mse_c3_list.append(np.mean(M.mse_c3[epoch]))
            mse_c4_list.append(np.mean(M.mse_c4[epoch]))
            MLE_id_list.append(np.mean(M.MLE_id[epoch]))

    plt.figure(figsize=(12,9))
    plt.title(f'tau = {M.tau}')
    ax1 = plt.subplot(2,1,1)
    plt.xlabel('epoch')
    plt.ylabel('ID')
    plt.plot(range(max_epoch), MLE_id_list)
    ax2 = plt.subplot(2,1,2)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.plot(range(max_epoch), mse_c1_list, label='c1')
    plt.plot(range(max_epoch), mse_c2_list, label='c2')
    plt.plot(range(max_epoch), mse_c3_list, label='c3')
    plt.plot(range(max_epoch), mse_c4_list, label='c4')
    # plt.ylim((0., 1.05*max(np.max(mse_c1_list), np.max(mse_c2_list), np.max(mse_c3_list))))
    plt.legend()
    plt.savefig(f'logs/2S2F/TimeSelection/embed_dim_{embed_dim}/tau_{tau}/ID_per_epoch.pdf', dpi=300)
    plt.close()


def plot_id_per_tau(tau_list, id_epoch, embed_dim):

    id_per_tau = [[] for _ in tau_list]
    for i, tau in enumerate(tau_list):
        fp = open(f'logs/2S2F/TimeSelection/embed_dim_{embed_dim}/tau_{round(tau,2)}/test_log.txt', 'r')
        for line in fp.readlines():
            seed = int(line[:-1].split(',')[1])
            epoch = int(line[:-1].split(',')[6])
            MLE_id = float(line[:-1].split(',')[7])

            if epoch in id_epoch:
                id_per_tau[i].append([MLE_id])
    
    for i in range(len(tau_list)):
        id_per_tau[i] = np.mean(id_per_tau[i], axis=0)
    id_per_tau = np.array(id_per_tau)

    round_id_per_tau = []
    for id in id_per_tau:
        round_id_per_tau.append([round(id[0])])
    round_id_per_tau = np.array(round_id_per_tau)

    plt.figure(figsize=(6,6))
    for i, item in enumerate(['MLE']):
        plt.plot(tau_list, id_per_tau[:,i], marker="o", markersize=6, label="ID")
        plt.plot(tau_list, round_id_per_tau[:,i], marker="^", markersize=6, label="ID-rounding")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.xlabel(r'$\tau / s$', fontsize=18)
    plt.ylabel('Intrinsic dimensionality', fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f'logs/2S2F/TimeSelection/embed_dim_{embed_dim}/id_per_tau.pdf', dpi=300)


def plot_1s2f_autocorr():

    data = np.load('Data/2S2F/origin/1/data.npz')
    X = np.array(data['X'])[:, np.newaxis]
    Y = np.array(data['Y'])[:, np.newaxis]
    Z = np.array(data['Z'])[:, np.newaxis]

    data = pd.DataFrame(np.concatenate((X,Y,Z), axis=-1), columns=['X', 'Y', 'Z'])
    
    corrX, corrY, corrZ = [], [], []
    lag_list = np.arange(0, 600*2000, 2000)
    for lag in tqdm(lag_list):
        corrX.append(data['X'].autocorr(lag=lag))
        corrY.append(data['Y'].autocorr(lag=lag))
        corrZ.append(data['Z'].autocorr(lag=lag))
    
    plt.figure(figsize=(12,8))
    plt.plot(lag_list*5e-6, np.array(corrX), label='X')
    plt.plot(lag_list*5e-6, np.array(corrY), label='Y')
    plt.plot(lag_list*5e-6, np.array(corrZ), label='Z')
    plt.xlabel('time/s')
    plt.legend()
    plt.title('Autocorrelation')
    plt.savefig('corr.pdf', dpi=300)
    

def plot_2s2f_autocorr():

    simdata = np.load('Data/2S2F/origin/origin.npz')
    
    trace_num = 200
    corrC1, corrC2, corrC3, corrC4 = [[] for _ in range(trace_num)], [[] for _ in range(trace_num)], \
        [[] for _ in range(trace_num)], [[] for _ in range(trace_num)]
    for trace_id in range(trace_num):
        tmp = np.array(simdata['trace'])[trace_id]
        c1 = tmp[:,0][:,np.newaxis]
        c2 = tmp[:,1][:,np.newaxis]
        c3 = tmp[:,2][:,np.newaxis]
        c4 = tmp[:,3][:,np.newaxis]

        data = pd.DataFrame(np.concatenate((c1,c2,c3,c4), axis=-1), columns=['c1','c2','c3','c4'])
        
        lag_list = np.arange(0, 510, 10)
        for lag in tqdm(lag_list):
            corrC1[trace_id].append(data['c1'].autocorr(lag=lag))
            corrC2[trace_id].append(data['c2'].autocorr(lag=lag))
            corrC3[trace_id].append(data['c3'].autocorr(lag=lag))
            corrC4[trace_id].append(data['c4'].autocorr(lag=lag))
    
    corrC1 = np.mean(corrC1, axis=0)
    corrC2 = np.mean(corrC2, axis=0)
    corrC3 = np.mean(corrC3, axis=0)
    corrC4 = np.mean(corrC4, axis=0)

    plt.figure(figsize=(6,6))
    plt.rcParams.update({'font.size':16})
    plt.plot(lag_list*1e-2, np.array(corrC1), marker="o", markersize=6, label=r'$c_1$')
    plt.plot(lag_list*1e-2, np.array(corrC2), marker="^", markersize=6, label=r'$c_2$')
    plt.plot(lag_list*1e-2, np.array(corrC3), marker="D", markersize=6, label=r'$c_3$')
    plt.plot(lag_list*1e-2, np.array(corrC4), marker="*", markersize=6, label=r'$c_4$')
    plt.xlabel(r'$\tau/s$', fontsize=18)
    plt.ylabel('Autocorrelation coefficient', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    # plt.subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig('2s2f_autocorr.pdf', dpi=300)
    
    
def plot_evolve(length):
    
    our = open(f'Results/2S2F/ours_evolve_test_{1.0}.txt', 'r')
    lstm = open(f'Results/2S2F/lstm_evolve_test_{length}.txt', 'r')
    tcn = open(f'Results/2S2F/tcn_evolve_test_{length}.txt', 'r')
    ode = open(f'Results/2S2F/neural_ode_evolve_test_{length}.txt', 'r')
    
    our_data = [[] for seed in range(10)]
    lstm_data = [[] for seed in range(3)]
    tcn_data = [[] for seed in range(3)]
    ode_data = [[] for seed in range(3)]
    for i, data in enumerate([our, lstm, tcn, ode]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            mse = float(line.split(',')[2])
            rmse = float(line.split(',')[3])
            mae = float(line.split(',')[4])
            mape = float(line.split(',')[5])
            c1_mae = float(line.split(',')[6])
            c2_mae = float(line.split(',')[7])
            
            if i==0:
                our_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae]),c1_mae,c2_mae])
            elif i==1:
                lstm_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae])])
            elif i==2:
                tcn_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae])])
            elif i==3:
                ode_data[seed-1].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae])])
    
    our_data = np.mean(np.array(our_data), axis=0)
    lstm_data = np.mean(np.array(lstm_data), axis=0)
    tcn_data = np.mean(np.array(tcn_data), axis=0)
    ode_data = np.mean(np.array(ode_data), axis=0)
    
    plt.figure(figsize=(16,16))
    for i, item in enumerate(['mse', 'rmse', 'mae', 'mape']):
        ax = plt.subplot(2,2,i+1)
        ax.plot(our_data[:,0], our_data[:,i+1], label='our')
        ax.plot(lstm_data[:,0], lstm_data[:,i+1], label='lstm')
        ax.plot(tcn_data[:,0], tcn_data[:,i+1], label='tcn')
        ax.plot(ode_data[:,0], ode_data[:,i+1], label='ode')
        ax.set_title(item)
        ax.set_xlabel(r'\tau / s', fontsize=18)
        ax.legend()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    plt.savefig(f'Results/2S2F/evolve_test_{length}.pdf', dpi=300)

    for i, item in enumerate(['RMSE', 'MAPE']):
        plt.figure(figsize=(6,6))
        plt.rcParams.update({'font.size':16})
        ax = plt.subplot(1,1,1)
        ax.plot(our_data[::2,0], our_data[::2,2*(i+1)], marker="o", markersize=6, label='our')
        ax.plot(lstm_data[::2,0], lstm_data[::2,2*(i+1)], marker="^", markersize=6, label='lstm')
        ax.plot(tcn_data[::2,0], tcn_data[::2,2*(i+1)], marker="D", markersize=6, label='tcn')
        ax.plot(ode_data[::2,0], ode_data[::2,2*(i+1)], marker="+", markersize=6, label='ode')
        ax.set_xlabel(r'$\tau / s$', fontsize=18)
        ax.set_ylabel(item, fontsize=18)
        ax.legend()
        # from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        # axins = ax.inset_axes((0.16, 0.55, 0.47, 0.35))
        # axins.plot(our_data[:int(len(our_data)*0.1):1,0], our_data[:int(len(our_data)*0.1):1,2*(i+1)], marker="o", markersize=6, label='our')
        # axins.plot(lstm_data[:int(len(lstm_data)*0.1):1,0], lstm_data[:int(len(lstm_data)*0.1):1,2*(i+1)], marker="^", markersize=6, label='lstm')
        # axins.plot(tcn_data[:int(len(tcn_data)*0.1):1,0], tcn_data[:int(len(tcn_data)*0.1):1,2*(i+1)], marker="D", markersize=6, label='tcn')
        # axins.plot(ode_data[:int(len(ode_data)*0.1):1,0], ode_data[:int(len(ode_data)*0.1):1,2*(i+1)], marker="+", markersize=6, label='ode')
        # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(f'Results/2S2F/evolve_comp_{item}.pdf', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.rcParams.update({'font.size':16})
    ax.plot(our_data[::2,0], our_data[::2,5], marker="o", markersize=6, label='Our Model')
    ax.plot(lstm_data[::2,0], lstm_data[::2,5], marker="^", markersize=6, label='LSTM')
    ax.plot(tcn_data[::2,0], tcn_data[::2,5], marker="D", markersize=6, label='TCN')
    ax.plot(ode_data[::2,0], ode_data[::2,5], marker="+", markersize=6, label='Neural ODE')
    ax.legend()
    # from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # axins = ax.inset_axes((0.13, 0.43, 0.43, 0.3))
    # axins.plot(our_data[:int(len(our_data)*0.6):3,0], our_data[:int(len(our_data)*0.6):3,5], marker="o", markersize=6, label='Our Model')
    # axins.plot(lstm_data[:int(len(lstm_data)*0.6):3,0], lstm_data[:int(len(lstm_data)*0.6):3,5], marker="^", markersize=6, label='LSTM')
    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    plt.xlabel(r'$\tau / s$', fontsize=18)
    plt.ylabel('MAE', fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'Results/2S2F/slow_evolve_mae.pdf', dpi=300)

    plt.figure(figsize=(4,4))
    plt.rcParams.update({'font.size':16})
    plt.plot(our_data[:,0], our_data[:,3], marker="o", markersize=6, label=r'$overall$')
    plt.plot(our_data[:,0], our_data[:,6], marker="^", markersize=6, label=r'$c_1$')
    plt.plot(our_data[:,0], our_data[:,7], marker="D", markersize=6, label=r'$c_2$')
    plt.xlabel(r'$\tau / s$', fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'Results/2S2F/our_slow_evolve_mae.pdf', dpi=300)
    
    item = ['our','lstm','tcn', 'ode']
    for i, data in enumerate([our_data, lstm_data, tcn_data, ode_data]):
        print(f'{item[i]} | tau[{data[0,0]:.1f}] RMSE={data[0,2]:.4f}, MAE={data[0,3]:.4f}, MAPE={100*data[0,4]:.2f}% | \
              tau[{data[9,0]:.1f}] RMSE={data[9,2]:.4f}, MAE={data[9,3]:.4f}, MAPE={100*data[9,4]:.2f}% | \
              tau[{data[49,0]:.1f}] RMSE={data[49,2]:.4f}, MAE={data[49,3]:.4f}, MAPE={100*data[49,4]:.2f}%')


def plot_result_per_id():

    slow_id_list = [1,2,3,4,5]
    seed_list = [1,2,3,4,5]

    results = []
    for slow_id in slow_id_list:
        koopman_id = 5
        
        data = open(f'Results/2S2F/slow_{slow_id}_koopman_{koopman_id}/fast_1/ours_evolve_test_1.0.txt', 'r')
    
        tmp = [[] for seed in seed_list]
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            mse = float(line.split(',')[2])
            rmse = float(line.split(',')[3])
            mae = float(line.split(',')[4])
            mape = float(line.split(',')[5])
            c1_mae = float(line.split(',')[6])
            c2_mae = float(line.split(',')[7])
            duration = float(line.split(',')[8])
            
            if seed in seed_list:
                tmp[seed_list.index(seed)].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae]),c1_mae,c2_mae,duration])

        results.append(np.mean(tmp, axis=0))
    results = np.array(results)
    
    plt.figure(figsize=(16,16))
    for i, item in enumerate(['mse', 'rmse', 'mae', 'mape']):
        ax = plt.subplot(2,2,i+1)
        for slow_id in slow_id_list:
            ax.plot(results[slow_id-1,:,0], results[slow_id-1,:,i+1], label=f'slow ID = {slow_id}')
        ax.set_title(item)
        ax.set_xlabel('t / s')
        ax.legend()
    plt.savefig(f'Results/2S2F/error_per_slow_id.pdf', dpi=300)

    plt.figure(figsize=(12,5))
    for i, index in enumerate([0, 9, 49]):
        ax = plt.subplot(1,3,i+1)
        # ax.plot(results[:,index,1], label='mse')
        # ax.plot(results[:,index,2], label='rmse')
        # ax.plot(results[:,index,3], label='mae')
        ax.plot(results[:,index,4], label='mape')
        ax.plot(results[:,index,8], label='duration')
        ax.set_xlabel('slow ID')
        ax.set_title(f'tau = {results[0,index,0]}')
        ax.legend()
    plt.savefig(f'Results/2S2F/duration_per_slow_id.pdf', dpi=300)
    
    for slow_id, result in zip(slow_id_list, results):
        # print(f"slow id={slow_id} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}%, duration={result[0,8]:.5f}ms | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}%, duration={result[9,8]:.5f}ms | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%, duration={result[49,8]:.5f}ms")
        # print(f"slow id={slow_id} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}% | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}% | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%")
        print(f"slow id={slow_id} | tau[{result[0,0]:.1f}] MAPE={100*result[0,4]:.2f}% | tau[{result[9,0]:.1f}] MAPE={100*result[9,4]:.2f}% | tau[{result[49,0]:.1f}] MAPE={100*result[49,4]:.2f}%")


def plot_result_per_koopman_id():

    koopman_id_list = [2,3,4,5,6,7]
    seed_list = [1]

    results = []
    for koopman_id in koopman_id_list:
        slow_id = 2
    
        data = open(f'Results/2S2F/slow_{slow_id}_koopman_{koopman_id}/ours_evolve_test_1.0.txt', 'r')
    
        tmp = [[] for seed in seed_list]
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            mse = float(line.split(',')[2])
            rmse = float(line.split(',')[3])
            mae = float(line.split(',')[4])
            mape = float(line.split(',')[5])
            c1_mae = float(line.split(',')[6])
            c2_mae = float(line.split(',')[7])
            duration = float(line.split(',')[8])
            
            if seed in seed_list:
                tmp[seed_list.index(seed)].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae]),c1_mae,c2_mae,duration])

        results.append(np.mean(tmp, axis=0))
    results = np.array(results)
    
    plt.figure(figsize=(16,16))
    for i, item in enumerate(['mse', 'rmse', 'mae', 'mape']):
        ax = plt.subplot(2,2,i+1)
        for koopman_id in koopman_id_list:
            ax.plot(results[koopman_id-slow_id,:,0], results[koopman_id-slow_id,:,i+1], label=f'koopman ID = {koopman_id}')
        ax.set_title(item)
        ax.set_xlabel('t / s')
        ax.legend()
    plt.savefig(f'Results/2S2F/error_per_koopman_id.pdf', dpi=300)

    plt.figure(figsize=(12,5))
    for i, index in enumerate([0, 9, 49]):
        ax = plt.subplot(1,3,i+1)
        # ax.plot(results[:,index,1], label='mse')
        # ax.plot(results[:,index,2], label='rmse')
        # ax.plot(results[:,index,3], label='mae')
        ax.plot(results[:,index,4], label='mape')
        ax.plot(results[:,index,8], label='duration')
        ax.set_xlabel('koopman ID')
        ax.set_title(f'tau = {results[0,index,0]}')
        ax.legend()
    plt.savefig(f'Results/2S2F/duration_per_koopman_id.pdf', dpi=300)
    
    for koopman_id, result in zip(koopman_id_list, results):
        # print(f"koop id={koopman_id} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}%, duration={result[0,8]:.5f}ms | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}%, duration={result[9,8]:.5f}ms | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%, duration={result[49,8]:.5f}ms")
        # print(f"koop id={koopman_id} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}% | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}% | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%")
        print(f"koop id={koopman_id} | tau[{result[0,0]:.1f}] MAPE={100*result[0,4]:.2f}% | tau[{result[9,0]:.1f}] MAPE={100*result[9,4]:.2f}% | tau[{result[49,0]:.1f}] MAPE={100*result[49,4]:.2f}%")
    

def plot_whether_fast():
    
    data_fast = open(f'Results/2S2F/slow_{2}_koopman_{2}/fast_1/ours_evolve_test_1.0.txt', 'r')
    data_no_fast = open(f'Results/2S2F/slow_{2}_koopman_{2}/fast_0/ours_evolve_test_1.0.txt', 'r')
    data_lstm = open(f'Results/2S2F/lstm_evolve_test_0.8.txt', 'r')

    results = []
    seed_list = [1,2,3]

    tmp = [[] for seed in seed_list]
    for line in data_fast.readlines():
        tau = float(line.split(',')[0])
        seed = int(line.split(',')[1])
        mse = float(line.split(',')[2])
        rmse = float(line.split(',')[3])
        mae = float(line.split(',')[4])
        mape = float(line.split(',')[5])
        c1_mae = float(line.split(',')[6])
        c2_mae = float(line.split(',')[7])
        duration = float(line.split(',')[8])
        
        if seed in seed_list:
            tmp[seed_list.index(seed)].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae]),c1_mae,c2_mae,duration])
    results.append(np.mean(tmp, axis=0))

    tmp = [[] for seed in seed_list]
    for line in data_no_fast.readlines():
        tau = float(line.split(',')[0])
        seed = int(line.split(',')[1])
        mse = float(line.split(',')[2])
        rmse = float(line.split(',')[3])
        mae = float(line.split(',')[4])
        mape = float(line.split(',')[5])
        c1_mae = float(line.split(',')[6])
        c2_mae = float(line.split(',')[7])
        duration = float(line.split(',')[8])
        
        if seed in seed_list:
            tmp[seed_list.index(seed)].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae]),c1_mae,c2_mae,duration])
    results.append(np.mean(tmp, axis=0))

    tmp = [[] for seed in seed_list]
    for line in data_lstm.readlines():
        tau = float(line.split(',')[0])
        seed = int(line.split(',')[1])
        mse = float(line.split(',')[2])
        rmse = float(line.split(',')[3])
        mae = float(line.split(',')[4])
        mape = float(line.split(',')[5])
        c1_mae = float(line.split(',')[6])
        c2_mae = float(line.split(',')[7])
        duration = float(line.split(',')[8])
        
        if seed in seed_list:
            tmp[seed_list.index(seed)].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae]),c1_mae,c2_mae,duration])
    results.append(np.mean(tmp, axis=0))

    results = np.array(results)

    plt.figure(figsize=(9,4))
    ax = plt.subplot(1,2,1)
    ax.plot(results[0,:,0], results[0,:,4], label='with fast', marker='o', markersize=4)
    ax.plot(results[1,:,0], results[1,:,4], label='no fast', marker='^', markersize=4)
    ax.set_xlabel('tau/s', fontsize=16)
    ax.set_ylabel('MAPE', fontsize=16)
    ax.legend()
    ax = plt.subplot(1,2,2)
    ax.plot(results[0,:,0], results[0,:,8], label='with fast', marker='o', markersize=4)
    ax.plot(results[1,:,0], results[1,:,8], label='no fast', marker='^', markersize=4)
    ax.set_xlabel('tau/s', fontsize=16)
    ax.set_ylabel('Inference duration(ms)', fontsize=16)
    ax.legend()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f'Results/2S2F/comp_whether_fast.png', dpi=300)
    
    for i, result in enumerate(results):
        # print(f"{['fast   ', 'no fast'][i]} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}%, duration={result[0,8]:.5f}ms | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}%, duration={result[9,8]:.5f}ms | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%, duration={result[49,8]:.5f}ms")
        print(f"{['origin ', 'no fast', 'lstm   '][i]} | tau[{result[0,0]:.1f}] MAPE={100*result[0,4]:.2f}%, duration={result[0,8]:.5f}ms | tau[{result[9,0]:.1f}] MAPE={100*result[9,4]:.2f}%, duration={result[9,8]:.5f}ms | tau[{result[49,0]:.1f}] MAPE={100*result[49,4]:.2f}%, duration={result[49,8]:.5f}ms")


def plot_result_per_alpha():

    alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    seed_list = [1]

    results = []
    for alpha in alpha_list:
    
        data = open(f'Results/2S2F/alpha_{alpha}/ours_evolve_test_1.0.txt', 'r')
    
        tmp = [[] for seed in seed_list]
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            mse = float(line.split(',')[2])
            rmse = float(line.split(',')[3])
            mae = float(line.split(',')[4])
            mape = float(line.split(',')[5])
            c1_mae = float(line.split(',')[6])
            c2_mae = float(line.split(',')[7])
            duration = float(line.split(',')[8])
            
            if seed in seed_list:
                tmp[seed_list.index(seed)].append([tau,mse,rmse,mae,mape,np.mean([c1_mae,c2_mae]),c1_mae,c2_mae,duration])

        results.append(np.mean(tmp, axis=0))
    results = np.array(results)
    
    plt.figure(figsize=(16,16))
    for i, item in enumerate(['mse', 'rmse', 'mae', 'mape']):
        ax = plt.subplot(2,2,i+1)
        for j, alpha in enumerate(alpha_list):
            ax.plot(results[j,:,0], results[j,:,i+1], label=f'alpha = {alpha}')
        ax.set_title(item)
        ax.set_xlabel('t / s')
        ax.legend()
    plt.savefig(f'Results/2S2F/alpha_{alpha}/error_per_alpha.pdf', dpi=300)
    
    for alpha, result in zip(alpha_list, results):
        # print(f"alpha={alpha} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}%, duration={result[0,8]:.5f}ms | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}%, duration={result[9,8]:.5f}ms | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%, duration={result[49,8]:.5f}ms")
        print(f"alpha={alpha} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}% | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}% | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%")
        # print(f"alpha={alpha} | tau[{result[0,0]:.1f}] MAPE={100*result[0,4]:.2f}% | tau[{result[9,0]:.1f}] MAPE={100*result[9,4]:.2f}% | tau[{result[49,0]:.1f}] MAPE={100*result[49,4]:.2f}%")


if __name__ == '__main__':
    
    # plot_2s2f_autocorr()
    plot_evolve(0.8)
    # plot_result_per_id()
    # plot_result_per_koopman_id()
    # plot_whether_fast()
    # plot_result_per_alpha()