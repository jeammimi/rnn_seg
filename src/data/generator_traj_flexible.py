import random
import numpy as np
#from matplotlib.axis import axis


def random_distr(l):
    r = np.random.uniform(0, 1)
    s = 0
    for item, prob in enumerate(l):
        s += prob
        if s >= r:
            return item
    return item


class HMC:
    """
    Generate markov chain
    """

    def __init__(self):
        self.list_state = ["start", "end"]
        self.transition = None

    def add_state(self, sname):
        self.list_state.append(sname)

    def add_transition(self, sn1, sn2, v):

        if self.transition is None:
            self.transition = np.zeros((len(self.list_state), len(self.list_state)))

        self.transition[self.list_state.index(sn1), self.list_state.index(sn2)] = v

    def get_transition(self, sn1, sn2):
        return self.transition[self.list_state.index(sn1), self.list_state.index(sn2)]

    def sample(self, time, noEnd=True):

        end = True
        while end:
            seq = ["start"]
            for i in range(time + 1):
                nextt = random_distr(self.transition[self.list_state.index(seq[-1]), ::])
                seq.append(self.list_state[nextt])

            if "end" in seq and noEnd:
                end = True
            else:
                end = False

        return seq[1:]


class MotionGenerator:

    def __init__(self, time, ndim, parameters, generate_motion, scales, constraint=[], continuous=False,):
        self.time = time
        self.ndim = ndim
        self.parameter_list = parameters
        self.generate_motion = generate_motion
        self.scales = scales
        self.continuous = continuous
        self.traj_generated = 0

    def generate_trajectories(self, n):
        scale = np.random.rand() * self.scales[self.traj_generated]
        self.trajectories = [self.generate_motion(
            p * scale, self.ndim, self.time + 1) for p in self.parameter_list[:n]]
        self.trajectories = [it[1:] - it[:-1] for it in self.trajectories]
        self.time_by_traj = [0 for p in self.parameter_list]
        self.traj_generated += 1

    def n_step(self, istate, n, last_step=""):

        c = self.trajectories[istate][self.time_by_traj[istate]:self.time_by_traj[istate] + n]
        self.time_by_traj[istate] += n

        if self.continuous and last_step != "":
            c = np.sum(last_step**2)**0.5 * (c / np.sum(c[0]**2)**0.5)
        return c


class GenerateTraj:

    def __init__(self, time, list_max_possible=[3, 3], n_max=5, list_generator=[],
                 lower_selfprob=0.4, fixed_self_proba=False,):

        self.time = time
        self.n_states = np.random.randint(1, n_max + 1)

        # generate list of possible state
        self.flat_list = ["%i" % (ni) for ni, i in enumerate(list_max_possible) for j in range(i)]
        self.list_state = []
        for i in range(self.n_states):
            self.list_state.append(self.flat_list.pop(np.random.randint(0, len(self.flat_list))))
        self.list_state.sort()
        self.n_list_state = []
        for n in range(len(list_max_possible)):
            if str(n) in self.list_state:
                self.n_list_state.extend(["%i_%i" % (n, j)
                                          for j in range(self.list_state.count(str(n)))])

        # Initiate model with list of possible state
        selfprob = lower_selfprob + (1 - lower_selfprob) * random.random()
        if fixed_self_proba:
            selfprob = lower_selfprob

        self.initiate_model(selfprob)
        self.sequence = self.model.sample(time - 1)

        all_states = list(set(self.sequence))
        all_states.sort()

        # Relabel for missing states
        table = [["%i_%i" % (ni, j) for j in range(i) if "%i_%i" % (ni, j) in all_states]
                 for ni, i in enumerate(list_max_possible)]

        translate_table = {}
        for it, t in enumerate(table):
            for n in range(len(t)):
                translate_table[t[n]] = "%i_%i" % (it, n)

        self.sequence = [translate_table[seq] for seq in self.sequence]

        # Generate the trajectories for the states needed
        nn = [istate.split("_")[0] for istate in all_states]
        for n in range(len(list_generator)):
            list_generator[n].generate_trajectories(nn.count(str(n)))

        self.traj = []
        t0, istate0 = map(int, self.sequence[0].split("_"))
        n = 1
        for seq in self.sequence[1:]:
            t, istate = map(int, seq.split("_"))
            if t == t0 and istate == istate0:
                n += 1
            else:
                ll = list_generator[t0].n_step(
                    istate0, n, last_step="" if self.traj == [] else self.traj[-1])
                self.traj.extend(ll)
                t0 = t
                istate0 = istate
                n = 1
        if n != 0:
            ll = list_generator[t0].n_step(
                istate0, n, last_step="" if self.traj == [] else self.traj[-1])
            self.traj.extend(ll)

        self.traj = np.array(self.traj)

        startc = [np.zeros_like(self.traj[0])]

        self.traj = np.cumsum(np.concatenate((startc, self.traj)), axis=0)

    def initiate_model(self, selfprob):

        model = HMC()
        for state in self.n_list_state:
            model.add_state(state)

        endp = 0.0000001
        for state0 in self.n_list_state:
            for state1 in self.n_list_state:
                if state1 == state0:
                    s0 = 0.166
                    s0 = selfprob
                    model.add_transition(state0, state1, s0)
                else:
                    model.add_transition(state0, state1, (1 - s0 - endp) /
                                         (len(self.n_list_state) - 1))

        for state in self.n_list_state:

            model.add_transition("start", state, 1.0 / len(self.n_list_state))
            model.add_transition(state, "end", endp)

        self.model = model


if __name__ == "__main__":
    from motion_type import diffusive, accelerated
    time = 100
    ndim = 2
    list_generator = [MotionGenerator(time, ndim,
                                      parameters=np.random.rand(3),
                                      generate_motion=diffusive,
                                      scales=[1, 10]),
                      MotionGenerator(time, ndim,
                                      parameters=np.random.rand(3),
                                      generate_motion=accelerated, continuous=True, scales=[1, 10])]

    A = GenerateTraj(100, list_max_possible=[2, 1], n_max=2, list_generator=list_generator)
    """
    print A.n_states
    print A.flat_list
    print A.list_state
    print A.n_list_state
    print A.sequence
    print A.traj,A.sequence"""
    print(len(A.traj), len(A.sequence))
    from Tools import plot_label

    def map_sequence(sequence):
        ns = []
        for iseque in sequence:
            i0, j0 = map(int, iseque.split("_"))
            ns.append(i0 * 3 + j0)
        print(ns)
        return ns
    print(np.sum((A.traj[1:] - A.traj[:-1])**2, axis=1)**0.5)
    plot_label(A.traj, map_sequence(A.sequence))
