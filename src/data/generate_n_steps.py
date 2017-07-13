from .generator_traj import generate_traj, EmptyError
from .motion_type import random_rot
from ..features.prePostTools import traj_to_dist

import numpy as np


def generate_n_steps(N, nstep, ndim, sub=False, noise_level=0.25):

    add = 0
    if ndim == 3:
        add = 1
    size = nstep

    X_train = np.zeros((N, size, (5 + add)))
    if sub:

        Y_trains = np.zeros((N, size, 10))
        Y_train_cat = np.zeros((N, 27))
    else:
        Y_trains = np.zeros((N, size, 7))
        Y_train_cat = np.zeros((N, 12))
    Y_train_traj = []

    # 12
    for i in range(N):
        # for i in range(1000):

        # if i % 1000 == 0:
        #    print i
        sigma = max(np.random.normal(0.5, 1), 0.05)
        step = max(np.random.normal(1, 1), 0.2)
        tryagain = True
        while tryagain:
            try:

                clean = 4
                if size >= 50:
                    clean = 8

                clean = False
                """
                ModelN,Model_num,s,sc,real_traj,norm,Z = generate_traj(size,sub=True,
                                                                       clean=clean,diff_sigma=2.0,
                                                                       delta_sigma_directed=1.,ndim=ndim,
                                                                      anisentropy=0.1,deltav=0.2,rho_fixed=False)
                """
                clean = 10
                ModelN, Model_num, s, sc, real_traj, norm, Z = generate_traj(size, sub=sub,
                                                                             clean=clean, diff_sigma=2.0,
                                                                             delta_sigma_directed=6., ndim=ndim,
                                                                             anisentropy=0.1, deltav=.4, rho_fixed=False,
                                                                             random_rotation=False)
                mu = 2
                Ra0 = [0, 1.]

                alpharot = 2 * 3.14 * np.random.random()

                dt = real_traj[1:] - real_traj[:-1]
                std = np.mean(np.sum(dt**2, axis=1) / 3)**0.5

                noise_l = noise_level * np.random.rand()
                real_traj += np.random.normal(0, noise_l * std, real_traj.shape)

                real_traj = random_rot(real_traj, alpharot, ndim=ndim)

                # print real_traj.shape
                alligned_traj, normed, alpha, _ = traj_to_dist(real_traj, ndim=ndim)
                simple = True
                if not simple:
                    real_traj1 = np.array([Propertie(real_traj[::, 0]).smooth(2),
                                           Propertie(real_traj[::, 1]).smooth(2)])
                    alligned_traj1, normed1, alpha1, _ = traj_to_dist(real_traj1.T, ndim=ndim)
                    real_traj2 = np.array([Propertie(real_traj[::, 0]).smooth(5),
                                           Propertie(real_traj[::, 1]).smooth(5)])
                    alligned_traj2, normed2, alpha2, _ = traj_to_dist(real_traj2.T, ndim=ndim)

                    normed = np.concatenate((normed[::, :4], normed1[::, :4], normed2), axis=1)

                for zero in Z:
                    normed[zero, ::] = 0

                tryagain = False

            except:
                tryagain = True

        Y_train_traj.append(real_traj)
        X_train[i] = normed

        Y_trains[i][range(size), np.array(sc, dtype=np.int)] = 1

        Y_train_cat[i, Model_num] = 1
    return X_train, Y_trains, Y_train_cat, Y_train_traj

# print np.sum(np.isnan(X_train))
