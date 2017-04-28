import numpy as np
from .generator_traj_flexible import MotionGenerator, GenerateTraj
from .motion_type import random_rot
from ..features.prePostTools import traj_to_dist
from .motion_type import diffusive, subdiffusive, directed, accelerated, slowed, still
from .motion_type import diffusive_confined, subdiffusive_confined, continuous_time_random_walk
from .motion_type import continuous_time_random_walk_confined


def add_miss_tracking(traj, N, f=10):

    step = traj[1:] - traj[:-1]

    std = np.average(np.sum(step**2, axis=1)**0.5)

    for i in range(N):
        w = np.random.randint(0, len(traj))
        traj[w] = np.random.normal(traj[w], f * std)

    return traj


def generate_n_steps(N, nstep, ndim):
    add = 0
    index_zero = 8
    if ndim == 3:
        add = 1
    size = nstep

    X_train = np.zeros((N, size, 5 + add))
    Y_trains_b = np.zeros((N, size, 9))
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

                clean = False

                time = size
                ndim = 2
                scales = [[1, 3, 10] for i in range(8)]
                # print(scales)
                list_generator = [MotionGenerator(time, ndim,
                                                  parameters=np.random.rand(3),
                                                  generate_motion=still, scales=scales[0]),
                                  MotionGenerator(time, ndim,
                                                  parameters=np.random.rand(3),
                                                  generate_motion=subdiffusive_confined, scales=scales[1]),
                                  MotionGenerator(time, ndim,
                                                  parameters=np.random.rand(3),
                                                  generate_motion=subdiffusive, scales=scales[2]),
                                  MotionGenerator(time, ndim,
                                                  parameters=np.random.rand(3),
                                                  generate_motion=diffusive_confined, scales=scales[3]),
                                  MotionGenerator(time, ndim,
                                                  parameters=np.random.rand(3),
                                                  generate_motion=diffusive, scales=scales[4]),
                                  MotionGenerator(time, ndim,
                                                  parameters=np.random.rand(3),
                                                  generate_motion=continuous_time_random_walk, scales=scales[5]),
                                  MotionGenerator(time, ndim,
                                                  parameters=np.random.rand(3),
                                                  generate_motion=continuous_time_random_walk_confined, scales=scales[6]),

                                  MotionGenerator(time, ndim,
                                                  parameters=np.random.rand(3),
                                                  generate_motion=directed, scales=scales[7])]

                A = GenerateTraj(time, n_max=4, list_max_possible=[
                                 3, 3, 3, 3, 3, 3, 3, 3], list_generator=list_generator)

                # Clean small seq
                all_states = set(A.sequence)
                n_states = [A.sequence.count(istate) for istate in all_states]

                for s, ns in zip(all_states, n_states):
                    A.sequence = np.array(A.sequence)
                    if size > 25 and ns < 10:
                        A.sequence[A.sequence == s] = "%i_0" % (index_zero)

                def map_sequence(sequence):
                    ns = []
                    for iseque in sequence:
                        i0, j0 = map(int, iseque.split("_"))
                        ns.append(i0)
                    return ns

                real_traj = A.traj
                sc = map_sequence(A.sequence)

                alpharot = 2 * 3.14 * np.random.random()

                real_traj = random_rot(real_traj, alpharot, ndim=ndim)

                # Noise
                dt = real_traj[1:] - real_traj[:-1]
                std = np.mean(np.sum(dt**2, axis=1) / 3)**0.5

                if std == 0:
                    std = 1
                noise_level = 0.25 * np.random.rand()
                real_traj += np.random.normal(0, noise_level * std, real_traj.shape)

                alligned_traj, normed, alpha, _ = traj_to_dist(real_traj, ndim=ndim)
                nzeros = np.random.randint(0, 10)

                Z = []
                for _ in range(nzeros):
                    Z.append(np.random.randint(len(sc) - 1))
                    sc[Z[-1]] = index_zero

                for i_isc, isc in enumerate(sc):
                    if isc == index_zero:
                        normed[i_isc, ::] = 0

                # print  alligned_traj.shape ,len(sc)

                tryagain = False

            except IndexError:
                tryagain = True

        Y_train_traj.append(real_traj)
        # print X_train.shape
        X_train[i] = normed

        Y_trains_b[i][range(time), np.array(sc, dtype=np.int)] = 1

        # print sc

    return X_train, Y_trains_b, Y_train_traj
