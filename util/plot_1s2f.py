import numpy as np
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']); plt.rcParams.update({'font.size':16})


def plot_epoch_test_log(tau, max_epoch, embed_dim):

    class MSE():
        def __init__(self, tau):
            self.tau = tau
            self.mse_x = [[] for _ in range(max_epoch)]
            self.mse_y = [[] for _ in range(max_epoch)]
            self.mse_z = [[] for _ in range(max_epoch)]
            self.LB_id = [[] for _ in range(max_epoch)]

    fp = open(f'logs/1S2F/TimeSelection/embed_dim_{embed_dim}/tau_{tau}/test_log.txt', 'r')
    items = []
    for line in fp.readlines():
        tau = float(line[:-1].split(',')[0])
        seed = int(line[:-1].split(',')[1])
        mse_x = float(line[:-1].split(',')[2])
        mse_y = float(line[:-1].split(',')[3])
        mse_z = float(line[:-1].split(',')[4])
        epoch = int(line[:-1].split(',')[5])
        LB_id = float(line[:-1].split(',')[6])

        find = False
        for M in items:
            if M.tau == tau:
                M.mse_x[epoch].append(mse_x)
                M.mse_y[epoch].append(mse_y)
                M.mse_z[epoch].append(mse_z)
                M.LB_id[epoch].append(LB_id)
                find = True
                    
        if not find:
            M = MSE(tau)
            M.mse_x[epoch].append(mse_x)
            M.mse_y[epoch].append(mse_y)
            M.mse_z[epoch].append(mse_z)
            M.LB_id[epoch].append(LB_id)
            items.append(M)
    fp.close()

    for M in items:
        mse_x_list = []
        mse_y_list = []
        mse_z_list = []
        LB_id_list = []
        MiND_id_list = []
        MADA_id_list = []
        PCA_id_list = []
        for epoch in range(max_epoch):
            mse_x_list.append(np.mean(M.mse_x[epoch]))
            mse_y_list.append(np.mean(M.mse_y[epoch]))
            mse_z_list.append(np.mean(M.mse_z[epoch]))
            LB_id_list.append(np.mean(M.LB_id[epoch]))

    plt.figure(figsize=(12,9))
    plt.title(f'tau = {M.tau}')
    ax1 = plt.subplot(2,1,1)
    plt.xlabel('epoch')
    plt.ylabel('ID')
    plt.plot(range(max_epoch), LB_id_list)
    ax2 = plt.subplot(2,1,2)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.plot(range(max_epoch), mse_x_list, label='x')
    plt.plot(range(max_epoch), mse_y_list, label='y')
    plt.plot(range(max_epoch), mse_z_list, label='z')
    # plt.ylim((0., 1.05*max(np.max(mse_x_list), np.max(mse_y_list), np.max(mse_z_list))))
    plt.legend()
    plt.savefig(f'logs/1S2F/TimeSelection/tau_{tau}/ID_per_epoch.pdf', dpi=300)
    plt.close()


def plot_id_per_tau(tau_list, id_epoch, embed_dim):

    id_per_tau = [[] for _ in tau_list]
    for i, tau in enumerate(tau_list):
        fp = open(f'logs/1S2F/TimeSelection/embed_dim_{embed_dim}/tau_{round(tau,2)}/test_log.txt', 'r')
        for line in fp.readlines():
            seed = int(line[:-1].split(',')[1])
            epoch = int(line[:-1].split(',')[5])
            LB_id = float(line[:-1].split(',')[6])

            if epoch in id_epoch:
                id_per_tau[i].append([LB_id])
    
    for i in range(len(tau_list)):
        id_per_tau[i] = np.mean(id_per_tau[i], axis=0)
    id_per_tau = np.array(id_per_tau)

    round_id_per_tau = []
    for id in id_per_tau:
        round_id_per_tau.append([round(id[0])])
    round_id_per_tau = np.array(round_id_per_tau)

    plt.figure(figsize=(6,6))
    plt.rcParams.update({'font.size':16})
    for i, item in enumerate(['MLE']):
        plt.plot(tau_list, id_per_tau[:,i], marker="o", markersize=6, label="ID")
        plt.plot(tau_list, round_id_per_tau[:,i], marker="^", markersize=6, label="ID-rounding")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.xlabel(r'$\tau / s$', fontsize=18)
    plt.ylabel('Intrinsic dimensionality', fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f'logs/1S2F/TimeSelection/embed_dim_{embed_dim}/id_per_tau.pdf', dpi=300)


def plot_1s2f_autocorr(totol_p, trace_num=20):

    corrX, corrY, corrZ = [[] for _ in range(trace_num)], [[] for _ in range(trace_num)], [[] for _ in range(trace_num)]
    for i in range(1, trace_num+1):

        data = np.load(f'Data/1S2F/origin-total_25.1/{i}/data.npz')
        X = np.array(data['X'])
        Y = np.array(data['Y'])
        Z = np.array(data['Z'])
        
        lag_list = np.arange(0, int(totol_p), 30)
        from tqdm import tqdm
        for lag in tqdm(lag_list):
            if lag == 0:
                corrX[i-1].append(1)
                corrY[i-1].append(1)
                corrZ[i-1].append(1)
            else:
                corrX[i-1].append(np.corrcoef(X[:-lag], X[lag:])[0,1])
                corrY[i-1].append(np.corrcoef(Y[:-lag], Y[lag:])[0,1])
                corrZ[i-1].append(np.corrcoef(Z[:-lag], Z[lag:])[0,1])
    
    corrX = np.mean(corrX, axis=0)
    corrY = np.mean(corrY, axis=0)
    corrZ = np.mean(corrZ, axis=0)

    plt.figure(figsize=(6,6))
    plt.plot(lag_list*1e-2, corrX, marker="o", markersize=6, label=r'$X$')
    plt.plot(lag_list*1e-2, corrY, marker="^", markersize=6, label=r'$Y$')
    plt.plot(lag_list*1e-2, corrZ, marker="D", markersize=6, label=r'$Z$')
    plt.xlabel(r'$\tau/s$', fontsize=18)
    plt.ylabel('Autocorrelation coefficient', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    # plt.subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig('1s2f_autocorr.pdf', dpi=300)


def plot_evolve(length):
    
    our = open(f'Results/1S2F/ours_evolve_test_{length}.txt', 'r')
    lstm = open(f'Results/1S2F/lstm_evolve_test_{length}.txt', 'r')
    tcn = open(f'Results/1S2F/tcn_evolve_test_{length}.txt', 'r')
    ode = open(f'Results/1S2F/neural_ode_evolve_test_{length}.txt', 'r')
    
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
            
            if i==0:
                our_data[seed-1].append([tau,mse,rmse,mae,mape])
            elif i==1:
                lstm_data[seed-1].append([tau,mse,rmse,mae,mape])
            elif i==2:
                tcn_data[seed-1].append([tau,mse,rmse,mae,mape])
            elif i==3:
                ode_data[seed-1].append([tau,mse,rmse,mae,mape])
    
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
    plt.savefig(f'Results/1S2F/evolve_test_{length}.pdf', dpi=300)
    
    item = ['our','lstm','tcn', 'ode']
    for i, data in enumerate([our_data, lstm_data, tcn_data, ode_data]):
        print(f'{item[i]} | tau[{data[0,0]:.1f}] RMSE={data[0,2]:.4f}, MAE={data[0,3]:.4f}, MAPE={100*data[0,4]:.2f}% | \
              tau[{data[9,0]:.1f}] RMSE={data[9,2]:.4f}, MAE={data[9,3]:.4f}, MAPE={100*data[9,4]:.2f}% | \
              tau[{data[49,0]:.1f}] RMSE={data[49,2]:.4f}, MAE={data[49,3]:.4f}, MAPE={100*data[49,4]:.2f}%')


def plot_result_per_slow_id():

    slow_id_list = [1,2,3,4,5]
    seed_list = [1,2,3,4,5]

    results = []
    for slow_id in slow_id_list:
        koopman_id = 5
    
        data = open(f'Results/1S2F/slow_{slow_id}_koopman_{koopman_id}/fast_1/ours_evolve_test_3.0.txt', 'r')
    
        tmp = [[] for seed in seed_list]
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            mse = float(line.split(',')[2])
            rmse = float(line.split(',')[3])
            mae = float(line.split(',')[4])
            mape = float(line.split(',')[5])
            duration = float(line.split(',')[6])
            
            if seed in seed_list:
                tmp[seed_list.index(seed)].append([tau,mse,rmse,mae,mape,duration])

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
    plt.savefig(f'Results/1S2F/error_per_slow_id.pdf', dpi=300)

    plt.figure(figsize=(12,5))
    for i, index in enumerate([0, 9, 49]):
        ax = plt.subplot(1,3,i+1)
        # ax.plot(results[:,index,1], label='mse')
        # ax.plot(results[:,index,2], label='rmse')
        # ax.plot(results[:,index,3], label='mae')
        ax.plot(results[:,index,4], label='mape')
        ax.plot(results[:,index,5], label='duration')
        ax.set_xlabel('slow ID')
        ax.set_title(f'tau = {results[0,index,0]}')
        ax.legend()
    plt.savefig(f'Results/1S2F/duration_per_slow_id.pdf', dpi=300)
    
    for slow_id, result in zip(slow_id_list, results):
        # print(f"slow id={slow_id} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}%, duration={result[0,5]:.5f}ms | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}%, duration={result[9,5]:.5f}ms | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%, duration={result[49,5]:.5f}ms")
        # print(f"slow id={slow_id} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}% | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}% | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%")
        print(f"slow id={slow_id} | tau[{result[0,0]:.1f}] MAPE={100*result[0,4]:.2f}% | tau[{result[9,0]:.1f}] MAPE={100*result[9,4]:.2f}% | tau[{result[49,0]:.1f}] MAPE={100*result[49,4]:.2f}%")


def plot_result_per_koopman_id():

    koopman_id_list = [1,2,3,4,5,6]
    seed_list = [1,2,3]

    results = []
    for koopman_id in koopman_id_list:
        slow_id = 1
    
        data = open(f'Results/1S2F/slow_{slow_id}_koopman_{koopman_id}/ours_evolve_test_3.0.txt', 'r')
    
        tmp = [[] for seed in seed_list]
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            mse = float(line.split(',')[2])
            rmse = float(line.split(',')[3])
            mae = float(line.split(',')[4])
            mape = float(line.split(',')[5])
            duration = float(line.split(',')[6])
            
            if seed in seed_list:
                tmp[seed_list.index(seed)].append([tau,mse,rmse,mae,mape,duration])

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
    plt.savefig(f'Results/1S2F/eror_per_koopman_id.pdf', dpi=300)

    plt.figure(figsize=(12,5))
    for i, index in enumerate([0, 9, 49]):
        ax = plt.subplot(1,3,i+1)
        # ax.plot(results[:,index,1], label='mse')
        # ax.plot(results[:,index,2], label='rmse')
        # ax.plot(results[:,index,3], label='mae')
        ax.plot(results[:,index,4], label='mape')
        ax.plot(results[:,index,5], label='duration')
        ax.set_xlabel('koopman ID')
        ax.set_title(f'tau = {results[0,index,0]}')
        ax.legend()
    plt.savefig(f'Results/1S2F/duration_per_koopman_id.pdf', dpi=300)
    
    for koopman_id, result in zip(koopman_id_list, results):
        # print(f"koop id={koopman_id} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}%, duration={result[0,5]:.5f}ms | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}%, duration={result[9,5]:.5f}ms | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%, duration={result[49,5]:.5f}ms")
        # print(f"koop id={koopman_id} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}% | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}% | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%")
        print(f"koop id={koopman_id} | tau[{result[0,0]:.1f}] MAPE={100*result[0,4]:.2f}% | tau[{result[9,0]:.1f}] MAPE={100*result[9,4]:.2f}% | tau[{result[49,0]:.1f}] MAPE={100*result[49,4]:.2f}%")


def plot_whether_fast():
    
    data_fast = open(f'Results/1S2F/slow_{1}_koopman_{1}/fast_1/ours_evolve_test_3.0.txt', 'r')
    data_no_fast = open(f'Results/1S2F/slow_{1}_koopman_{1}/fast_0/ours_evolve_test_3.0.txt', 'r')
    data_lstm = open(f'Results/1S2F/lstm_evolve_test_3.0.txt', 'r')

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
        duration = float(line.split(',')[6])
        
        if seed in seed_list:
            tmp[seed_list.index(seed)].append([tau,mse,rmse,mae,mape,duration])
    results.append(np.mean(tmp, axis=0))

    tmp = [[] for seed in seed_list]
    for line in data_no_fast.readlines():
        tau = float(line.split(',')[0])
        seed = int(line.split(',')[1])
        mse = float(line.split(',')[2])
        rmse = float(line.split(',')[3])
        mae = float(line.split(',')[4])
        mape = float(line.split(',')[5])
        duration = float(line.split(',')[6])
        
        if seed in seed_list:
            tmp[seed_list.index(seed)].append([tau,mse,rmse,mae,mape,duration])
    results.append(np.mean(tmp, axis=0))

    seed_list = [1,2,3,4,5,6,7,8,9,10]
    tmp = [[] for seed in seed_list]
    for line in data_lstm.readlines():
        tau = float(line.split(',')[0])
        seed = int(line.split(',')[1])
        mse = float(line.split(',')[2])
        rmse = float(line.split(',')[3])
        mae = float(line.split(',')[4])
        mape = float(line.split(',')[5])
        duration = float(line.split(',')[6])
        
        if seed in seed_list:
            tmp[seed_list.index(seed)].append([tau,mse,rmse,mae,mape,duration])
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
    ax.plot(results[0,:,0], results[0,:,5], label='with fast', marker='o', markersize=4)
    ax.plot(results[1,:,0], results[1,:,5], label='no fast', marker='^', markersize=4)
    ax.set_xlabel('tau/s', fontsize=16)
    ax.set_ylabel('Inference duration(ms)', fontsize=16)
    ax.legend()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f'Results/1S2F/comp_whether_fast.png', dpi=300)
    
    for i, result in enumerate(results):
        # print(f"{['fast   ', 'no fast'][i]} | tau[{result[0,0]:.1f}] RMSE={result[0,2]:.4f}, MAE={result[0,3]:.4f}, MAPE={100*result[0,4]:.2f}%, duration={result[0,8]:.5f}ms | tau[{result[9,0]:.1f}] RMSE={result[9,2]:.4f}, MAE={result[9,3]:.4f}, MAPE={100*result[9,4]:.2f}%, duration={result[9,8]:.5f}ms | tau[{result[49,0]:.1f}] RMSE={result[49,2]:.4f}, MAE={result[49,3]:.4f}, MAPE={100*result[49,4]:.2f}%, duration={result[49,8]:.5f}ms")
        print(f"{['origin ', 'no fast', 'lstm   '][i]} | tau[{result[0,0]:.1f}] MAPE={100*result[0,4]:.2f}%, duration={result[0,5]:.5f}ms | tau[{result[9,0]:.1f}] MAPE={100*result[9,4]:.2f}%, duration={result[9,5]:.5f}ms | tau[{result[49,0]:.1f}] MAPE={100*result[49,4]:.2f}%, duration={result[49,5]:.5f}ms")


if __name__ == '__main__':
    
    # plot_1s2f_autocorr(totol_p=15*100)
    plot_evolve(3.0)
    # plot_result_per_slow_id()
    # plot_result_per_koopman_id()
    # plot_whether_fast()
