import os
import numpy as np
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
from multiprocessing import Process
import warnings;warnings.simplefilter('ignore')

from .gillespie import generate_origin


def findNearestPoint(data_t, start=0, object_t=10.0):
    '''Find the nearest time point to object time.'''

    index = start

    if index >= len(data_t):
        return index

    while not (data_t[index] <= object_t and data_t[index+1] > object_t):
        if index < len(data_t)-2:
            index += 1
        elif index == len(data_t)-2: # last one
            index += 1
            break
    
    return index


def time_discretization(seed, total_t, dt=None, is_print=False):
    '''
    Time-forward NearestNeighbor interpolate to discretizate the time.

    Args:
        seed (int): random seed
        total_t (float): simulation time
        dt (float): time unit
        is_print (bool): whether print log to terminal

    Returns: None
    '''

    data = np.load(f'Data/1S2F/origin/{seed}/origin.npz')
    data_t = data['t']
    data_X = data['X']
    data_Y = data['Y']
    data_Z = data['Z']

    dt = 5e-6 if dt is None else dt
    current_t = 0.0
    index = 0
    t, X, Y, Z = [], [], [], []
    while current_t < total_t:
        index = findNearestPoint(data_t, start=index, object_t=current_t)
        t.append(current_t)
        X.append(data_X[index])
        Y.append(data_Y[index])
        Z.append(data_Z[index])

        current_t += dt

        if is_print == 1: print(f'\rSeed[{seed}] interpolating {current_t:.6f}/{total_t}', end='')

    plt.figure(figsize=(16,5))
    # plt.title(f'dt = {dt}')
    plt.subplot(1,3,1)
    plt.plot(t, X, label=r'$X$')
    plt.xlabel(r'$t / s$', fontsize=18)
    plt.ylabel(r'$X$', fontsize=18)
    plt.subplot(1,3,2)
    plt.plot(t, Y, label=r'$Y$')
    plt.xlabel(r'$t / s$', fontsize=18)
    plt.ylabel(r'$Y$', fontsize=18)
    plt.subplot(1,3,3)
    plt.plot(t, Z, label=r'$Z$')
    plt.xlabel(r'$t / s$', fontsize=18)
    plt.ylabel(r'$Z$', fontsize=18)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.9,
        bottom=0.15,
        wspace=0.2
    )
    plt.savefig(f'Data/1S2F/origin/{seed}/data.pdf', dpi=300)

    np.savez(f'Data/1S2F/origin/{seed}/data.npz', dt=dt, t=t, X=X, Y=Y, Z=Z)


def generate_original_data(trace_num, total_t, dt, save=True, plot=False, parallel=False):
    '''
    Generate simulation data.

    Args:
        trace_num (int): num of simulation trajectories
        total_t (float): simulation time
        dt (float): time unit
        parallel (bool): parallel or serial execution

    Returns: None
    '''

    os.makedirs('Data/1S2F/origin', exist_ok=True)

    # generate original data by gillespie algorithm
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/1S2F/origin/{seed}/origin.npz'):
            IC = [np.random.randint(5,200), np.random.randint(5,100), np.random.randint(0,5000)]
            if parallel:
                subprocess.append(Process(target=generate_origin, args=(total_t, seed, IC), daemon=True))
                subprocess[-1].start()
            else:
                generate_origin(total_t, seed, IC)
    while any([subp.exitcode == None for subp in subprocess]):
        pass
    
    # time discretization by time-forward NearestNeighbor interpolate
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/1S2F/origin/{seed}/data.npz'):
            is_print = len(subprocess)==0
            if parallel:
                subprocess.append(Process(target=time_discretization, args=(seed, total_t, dt, is_print), daemon=True))
                subprocess[-1].start()
            else:
                time_discretization(seed, total_t, dt, is_print)
    while any([subp.exitcode == None for subp in subprocess]):
        pass

    print(f'save origin data form seed 1 to {trace_num} at Data/1S2F/origin/')
    
    
def generate_dataset(trace_num, tau, sample_num=None, is_print=False, sequence_length=None):
    '''
    Process simulation data.

    Args:
        trace_num (int): num of trajectories
        tau (float): time step
        sample_num (float): sample num of each trajectory
        is_print (bool): whether print log to terminal
        sequence_length (int): length of continuous time series

    Returns: None
    '''

    if (sequence_length is not None) and os.path.exists(f"Data/1S2F/data/tau_{tau}/train_{sequence_length}.npz") and \
        os.path.exists(f"Data/1S2F/data/tau_{tau}/val_{sequence_length}.npz") and \
            os.path.exists(f"Data/1S2F/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif (sequence_length is None) and os.path.exists(f"Data/1S2F/data/tau_{tau}/train.npz") and \
        os.path.exists(f"Data/1S2F/data/tau_{tau}/val.npz") and \
            os.path.exists(f"Data/1S2F/data/tau_{tau}/test.npz"):
        return

    # load original data
    if is_print: print('loading original trace data:')
    data = []
    iter = tqdm(range(1, trace_num+1)) if is_print else range(1, trace_num+1)
    for trace_id in iter:
        tmp = np.load(f"Data/1S2F/origin/{trace_id}/data.npz")
        dt = tmp['dt']
        X = np.array(tmp['X'])[:, np.newaxis, np.newaxis] # (sample_num, channel, feature_num)
        Y = np.array(tmp['Y'])[:, np.newaxis, np.newaxis]
        Z = np.array(tmp['Z'])[:, np.newaxis, np.newaxis]

        trace = np.concatenate((X, Y, Z), axis=-1)
        data.append(trace[np.newaxis])
    data = np.concatenate(data, axis=0)

    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')

    # save statistic information
    data_dir = f"Data/1S2F/data/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(data_dir + "/data_mean.txt", np.mean(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_std.txt", np.std(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_max.txt", np.max(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_min.txt", np.min(data, axis=(0,1)))
    np.savetxt(data_dir + "/tau.txt", [tau]) # Save the timestep

    # single-sample time steps
    if sequence_length is None:
        sequence_length = 2 if tau != 0. else 1
        seq_none = True
    else:
        seq_none = False
    
    ##################################
    # Create [train,val,test] dataset
    ##################################
    train_num = int(0.7*trace_num)
    val_num = int(0.1*trace_num)
    test_num = int(0.2*trace_num)
    trace_list = {'train':range(train_num), 'val':range(train_num,train_num+val_num), 'test':range(train_num+val_num,train_num+val_num+test_num)}
    for item in ['train','val','test']:
                
        # select trace num
        N_TRACE = len(trace_list[item])
        data_item = data[trace_list[item]]

        # subsampling
        step_length = int(tau/dt) if tau!=0. else 1

        # select sliding window index from N trace
        idxs_timestep = []
        idxs_ic = []
        for ic in range(N_TRACE):
            seq_data = data_item[ic]
            idxs = np.arange(0, np.shape(seq_data)[0]-step_length*(sequence_length-1), 1)
            for idx_ in idxs:
                idxs_ic.append(ic)
                idxs_timestep.append(idx_)

        # generator item dataset
        sequences = []
        parallel_sequences = [[] for _ in range(N_TRACE)]
        for bn in range(len(idxs_timestep)):
            idx_ic = idxs_ic[bn]
            idx_timestep = idxs_timestep[bn]
            tmp = data_item[idx_ic, idx_timestep : idx_timestep+step_length*(sequence_length-1)+1 : step_length]
            sequences.append(tmp)
            parallel_sequences[idx_ic].append(tmp)
            if is_print: print(f'\rtau[{tau}] sliding window for {item} data [{bn+1}/{len(idxs_timestep)}]', end='')
        if is_print: print()

        sequences = np.array(sequences) 
        if is_print: print(f'tau[{tau}]', f"{item} dataset (sequence_length={sequence_length})", np.shape(sequences))

        # keep sequences_length equal to sample_num
        if sample_num is not None:
            repeat_num = int(np.floor(N_TRACE*sample_num/len(sequences)))
            idx = np.random.choice(range(len(sequences)), N_TRACE*sample_num-len(sequences)*repeat_num, replace=False)
            idx = np.sort(idx)
            tmp1 = sequences[idx]
            tmp2 = None
            for i in range(repeat_num):
                if i == 0:
                    tmp2 = sequences
                else:
                    tmp2 = np.concatenate((tmp2, sequences), axis=0)
            sequences = tmp1 if tmp2 is None else np.concatenate((tmp1, tmp2), axis=0)
        if is_print: print(f'tau[{tau}]', f"after process", np.shape(sequences))

        # save
        if not seq_none:
            np.savez(data_dir+f'/{item}_{sequence_length}.npz', data=sequences)
        else:
            np.savez(data_dir+f'/{item}.npz', data=sequences)

        # plot
        if seq_none:
            plt.figure(figsize=(16,10))
            plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
            for i in range(3):
                ax = plt.subplot(3,1,i+1)
                ax.set_title(['X','Y','Z'][i])
                plt.plot(sequences[:, 0, 0, i])
            plt.subplots_adjust(left=0.05, bottom=0.05,  right=0.95,  top=0.95,  hspace=0.35)
            plt.savefig(data_dir+f'/{item}_input.pdf', dpi=300)

            plt.figure(figsize=(16,10))
            plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
            for i in range(3):
                ax = plt.subplot(3,1,i+1)
                ax.set_title(['X','Y','Z'][i])
                plt.plot(sequences[:, sequence_length-1, 0, i])
            plt.subplots_adjust(left=0.05, bottom=0.05,  right=0.95,  top=0.95,  hspace=0.35)
            plt.savefig(data_dir+f'/{item}_target.pdf', dpi=300)