#! -*- coding: utf-8 -*-

import os
import time
import numpy as np
from tqdm import tqdm
from scipy.special import comb
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything


class Reaction:
    '''Chemcial reaction'''

    def __init__(self, rate=0., num_lefts=None, num_rights=None):

        self.rate = rate
        assert len(num_lefts) == len(num_rights)
        self.num_lefts = np.array(num_lefts)
        self.num_rights = np.array(num_rights)
        self.num_diff = self.num_rights - self.num_lefts

    def combine(self, n, s):
        return np.prod(comb(n, s))

    def propensity(self, n):
        return self.rate * self.combine(n, self.num_lefts)


class System:
    '''Gillespie Algorithm'''

    def __init__(self, num_elements):

        assert num_elements > 0
        self.num_elements = num_elements
        self.reactions = []

        self.noise_t = 0

    def add_reaction(self, rate=0., num_lefts=None, num_rights=None):

        assert len(num_lefts) == self.num_elements
        assert len(num_rights) == self.num_elements
        self.reactions.append(Reaction(rate, num_lefts, num_rights))

    def evolute(self, steps=None, total_t=None, IC=None, seed=1):

        self.t = [0]

        if IC is None:
            self.n = [np.array([100, 40, 2500])]
        else:
            self.n = [np.array(IC)]

        if steps is not None:
            for i in tqdm(range(steps)):
                A = np.array([rec.propensity(self.n[-1]) for rec in self.reactions])
                A0 = A.sum()
                A /= A0
                t0 = -np.log(np.random.random())/A0
                self.t.append(self.t[-1] + t0)
                d = np.random.choice(self.reactions, p=A)
                self.n.append(self.n[-1] + d.num_diff)
        else:
            total_t = 10 if total_t is None else total_t
            while self.t[-1] < total_t:
                A = np.array([rec.propensity(self.n[-1]) for rec in self.reactions])
                A0 = A.sum()
                A /= A0
                t0 = -np.log(np.random.random())/A0
                self.t.append(self.t[-1] + t0)
                d = np.random.choice(self.reactions, p=A)
                self.n.append(self.n[-1] + d.num_diff)
                if seed == 1:
                    print(f'\rSeed[{seed}] time: {self.t[-1]:.3f}/{total_t}s | X={self.n[-1][0]}, Y={self.n[-1][1]}, Z={self.n[-1][2]}', end='')

    def reset(self, IC):

        self.t = [0]
        
        if IC is None:
            self.n = [np.array([100, 40, 2500])]
        else:
            self.n = [np.array(IC)]
        


def generate_origin(total_t=None, seed=729, IC=[100,40,2500]):
    '''
    Simulate the chemical reaction by Gillespie algorithm.

    Args:
        total_t (float): simulation time
        seed (int): random seed
        IC (list[int]): initial conditions [X0,Y0,Z0]

    Returns:
        float: The difference of a and b.
    '''

    time.sleep(1.0)

    seed_everything(seed)
    os.makedirs(f'Data/1S2F/origin/{seed}/', exist_ok=True)

    num_elements = 3
    system = System(num_elements)

    # X, Y, Z
    system.add_reaction(1000, [1, 0, 0], [1, 0, 1])
    system.add_reaction(1, [0, 1, 1], [0, 1, 0])
    system.add_reaction(40, [0, 0, 0], [0, 1, 0])
    system.add_reaction(1, [0, 1, 0], [0, 0, 0])
    system.add_reaction(1, [0, 0, 0], [1, 0, 0])

    system.evolute(total_t=total_t, seed=seed, IC=IC)

    t = system.t
    X = [i[0] for i in system.n]
    Y = [i[1] for i in system.n]
    Z = [i[2] for i in system.n]

    plt.figure(figsize=(16,4))
    ax1 = plt.subplot(1,3,1)
    ax1.set_title('X')
    plt.plot(t, X, label='X')
    plt.xlabel('time / s')
    ax2 = plt.subplot(1,3,2)
    ax2.set_title('Y')
    plt.plot(t, Y, label='Y')
    plt.xlabel('time / s')
    ax3 = plt.subplot(1,3,3)
    ax3.set_title('Z')
    plt.plot(t, Z, label='Z')
    plt.xlabel('time / s')

    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.9,
        bottom=0.15,
        wspace=0.2
    )
    plt.savefig(f'Data/1S2F/origin/{seed}/origin.pdf', dpi=500)
    
    # calculate average dt
    digit = f'{np.average(np.diff(t)):.20f}'.count("0")
    avg = np.round(np.average(np.diff(t)), digit)

    np.savez(f'Data/1S2F/origin/{seed}/origin.npz', t=t, X=X, Y=Y, Z=Z, dt=avg)
