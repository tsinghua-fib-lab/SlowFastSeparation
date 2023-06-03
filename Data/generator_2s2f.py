import os
import numpy as np
from tqdm import tqdm
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science']);plt.rcParams.update({'font.size':16})
from scipy.integrate import odeint
from pytorch_lightning import seed_everything
import warnings;warnings.simplefilter('ignore')


def system_4d(y0, t, para=(0.025,3)):
    '''2S2F ODE'''
    
    epsilon, omega =  para
    c1, c2, c3, c4 = y0
    
    dc1 = -c1
    dc2 = -2 * c2
    dc3 = -(c3-np.sin(omega*c1)*np.sin(omega*c2))/epsilon - c1*omega*np.cos(omega*c1)*np.sin(omega*c2) \
        - c2*omega*np.cos(omega*c2)*np.sin(omega*c1)
    dc4 = -(c4-1/((1+np.exp(-omega*c1))*(1+np.exp(-omega*c2))))/epsilon \
        - c1*omega*np.exp(-omega*c1)/((1+np.exp(-omega*c2))*((1+np.exp(-omega*c1))**2)) \
            - c2*omega*np.exp(-omega*c2)/((1+np.exp(-omega*c1))*((1+np.exp(-omega*c2))**2))
    
    return [dc1, dc2, dc3, dc4]


def generate_original_data(trace_num, total_t=5, dt=0.0001, save=True, plot=False, parallel=False):
    '''
    Generate simulation data.

    Args:
        trace_num (int): num of simulation trajectories
        total_t (float): simulation time
        dt (float): time unit
        parallel (bool): parallel or serial execution

    Returns: None
    '''
    
    def solve_1_trace(trace_id=0, total_t=5, dt=0.001):
        
        seed_everything(trace_id)
        
        y0 = [np.random.uniform(-3,3) for _ in range(4)]
        t  =np.arange(0, total_t, dt)
        
        sol = odeint(system_4d, y0, t)

        if plot:
            plt.figure()
            plt.plot(t, sol[:,0], label='c1')
            plt.plot(t, sol[:,1], label='c2')
            plt.plot(t, sol[:,2], label='c3')
            plt.plot(t, sol[:,3], label='c4')
            plt.legend()
            plt.savefig(f'Data/2S2F/origin/2s2f_{trace_id}.pdf', dpi=100)
        
        return sol
    
    if save and os.path.exists('Data/2S2F/origin/origin.npz'): return
    
    trace = []
    for trace_id in tqdm(range(1, trace_num+1)):
        sol = solve_1_trace(trace_id, total_t, dt)
        trace.append(sol)
    
    if save: 
        os.makedirs('Data/2S2F/origin', exist_ok=True)
        np.savez('Data/2S2F/origin/origin.npz', trace=trace, dt=dt, T=total_t)
        print(f'save origin data form seed 1 to {trace_num} at Data/2S2F/origin/')
    
    plot_c3_c4_trajectory() # plot c3,c4 trajectory

    return trace

    
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

    if (sequence_length is not None) and \
        os.path.exists(f"Data/2S2F/data/tau_{tau}/train_{sequence_length}.npz") and \
            os.path.exists(f"Data/2S2F/data/tau_{tau}/val_{sequence_length}.npz") and \
                os.path.exists(f"Data/2S2F/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif (sequence_length is None) and \
        os.path.exists(f"Data/2S2F/data/tau_{tau}/train.npz") and \
            os.path.exists(f"Data/2S2F/data/tau_{tau}/val.npz") and \
                os.path.exists(f"Data/2S2F/data/tau_{tau}/test.npz"):
        return
    
    # load original data
    if is_print: print('loading original trace data:')
    tmp = np.load(f"Data/2S2F/origin/origin.npz")
    dt = tmp['dt']
    data = np.array(tmp['trace'])[:trace_num,:,np.newaxis] # (trace_num, time_length, channel, feature_num)
    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')

    # save statistic information
    data_dir = f"Data/2S2F/data/tau_{tau}"
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
        if is_print: print(f'tau[{tau}]', f"{item} dataset (sequence_length={sequence_length}, step_length={step_length})", np.shape(sequences))

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
                plt.plot(sequences[:,0,0,0], label='c1')
                plt.plot(sequences[:,0,0,1], label='c2')
                plt.plot(sequences[:,0,0,2], label='c3')
                plt.plot(sequences[:,0,0,3], label='c4')
                plt.legend()
                plt.savefig(data_dir+f'/{item}_input.pdf', dpi=300)

                plt.figure(figsize=(16,10))
                plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
                plt.plot(sequences[:,sequence_length-1,0,0], label='c1')
                plt.plot(sequences[:,sequence_length-1,0,1], label='c2')
                plt.plot(sequences[:,sequence_length-1,0,2], label='c3')
                plt.plot(sequences[:,sequence_length-1,0,3], label='c4')
                plt.legend()
                plt.savefig(data_dir+f'/{item}_target.pdf', dpi=300)


def plot_c3_c4_trajectory():
    
    c1, c2 = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))

    omega = 3
    c3 = np.sin(omega*c1)*np.sin(omega*c2)
    c4 = 1/((1+np.exp(-omega*c1))*(1+np.exp(-omega*c2)))

    y0 = [1.,1.,2.,1.5]
    t  =np.arange(0, 5.1, 0.05)
    sol = odeint(system_4d, y0, t)
    c1_trace = sol[:, 0]
    c2_trace = sol[:, 1]
    c3_trace = sol[:, 2]
    c4_trace = sol[:, 3]
    
    for i, (c, trace) in enumerate(zip([c3,c4], [c3_trace,c4_trace])):
        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(111,projection='3d')

        # plot the slow manifold and c3,c4 trajectory
        ax.scatter(c1, c2, c, marker='.', color='k', label=rf'Points on slow-manifold surface')
        ax.plot(c1_trace, c2_trace, trace, linewidth=2, color="r", label=rf'Solution  trajectory')
        ax.scatter(c1_trace[::2], c2_trace[::2], trace[::2], linewidth=2, color="b", marker='o')
        ax.set_xlabel(r"$c_2$", labelpad=10, fontsize=18)
        ax.set_ylabel(r"$c_1$", labelpad=10, fontsize=18)
        ax.set_zlim(0, 2)
        ax.text2D(0.85, 0.65, rf"$c_{2+i+1}$", fontsize=18, transform=ax.transAxes)
        # ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(False)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.view_init(elev=25., azim=-60.) # view direction: elve=vertical angle ,azim=horizontal angle
        plt.tick_params(labelsize=16)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.zaxis.set_major_locator(plt.MultipleLocator(1))
        if i == 1:
            plt.legend()
        plt.subplots_adjust(bottom=0., top=1.)
        plt.savefig(f"Data/2S2F/origin/c{2+i+1}.pdf", dpi=300)
        plt.close()