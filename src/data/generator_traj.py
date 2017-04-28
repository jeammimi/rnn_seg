import random
import copy
import types
import numpy as np
import math
from .motion_type import random_rot


class EmptyError(Exception):
    pass
#from matplotlib.axis import axis


def create_random_alpha(N, alpha=0.5, drift=[0, 0, 0], ndim=2):
    coord = []
    hurstExponent = alpha / 2.

    L = int(N / 128.) + 1

    scaleFactor = 2 ** (2.0 * hurstExponent)

    def curve2D(t0, x0, y0, t1, x1, y1, variance, scaleFactor):
        if (t1 - t0) < .01:
            # print
            #stddraw.line(x0, y0, x1, y1)
            coord.append([t1, x1, y1])
            return  # [x0, y0] ,[x1, y1]
        tm = (t0 + t1) / 2.0
        ym = (y0 + y1) / 2.0
        xm = (x0 + x1) / 2.0
        deltax = np.random.normal(0, math.sqrt(variance))
        deltay = np.random.normal(0, math.sqrt(variance))

        curve2D(t0, x0, y0, tm, xm + deltax, ym + deltay, variance / scaleFactor, scaleFactor)
        curve2D(tm, xm + deltax, ym + deltay, t1, x1, y1, variance / scaleFactor, scaleFactor)

    def curve3D(t0, x0, y0, z0, t1, x1, y1, z1, variance, scaleFactor):
        if (t1 - t0) < .01:
            # print
            #stddraw.line(x0, y0, x1, y1)
            coord.append([t1, x1, y1, z1])
            return  # [x0, y0] ,[x1, y1]
        tm = (t0 + t1) / 2.0
        ym = (y0 + y1) / 2.0
        xm = (x0 + x1) / 2.0
        zm = (z0 + z1) / 2.0

        deltax = np.random.normal(0, math.sqrt(variance))
        deltay = np.random.normal(0, math.sqrt(variance))
        deltaz = np.random.normal(0, math.sqrt(variance))

        curve3D(t0, x0, y0, z0, tm, xm + deltax, ym + deltay,
                zm + deltaz, variance / scaleFactor, scaleFactor)
        curve3D(tm, xm + deltax, ym + deltay, zm + deltaz, t1,
                x1, y1, z1, variance / scaleFactor, scaleFactor)

    scale_step = 8.5
    if ndim == 2:
        curve2D(0., 0., 0., L, 0. + drift[0], 0.0 + drift[1], scale_step, scaleFactor)
    if ndim == 3:
        curve3D(0., 0., 0., 0, L, 0. + drift[0], 0.0 +
                drift[1], 0.0 + drift[2], scale_step, scaleFactor)

    # print L
    return np.array(coord)[::, 1:]


def random_distr(l):
    r = random.uniform(0, 1)
    s = 0
    for item, prob in enumerate(l):
        s += prob
        if s >= r:
            return item
    return item


class Model3:

    def __init__(self, name):
        self.name = name
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


def one_particle_n_states_s(ListState0, transition_mat=[], StateN={}, selfprob=0.033):

    model = Model3("Unknown")

    pre = 0.0001
    Ra0 = "Ra0"
    Ra1 = "Ra1"

    Ra2 = "Ra2"

    sRa0 = "sRa0"
    sRa1 = "sRa1"

    sRa2 = "sRa2"

    Le0 = "Le0"

    Ri0 = "Ri0"

    Ri1 = "Ri1"

    # ,Ra3,Ra4,Ra5,Le0]#,Ri0]#,Ri1,Le1]
    ListStatet = [Ra0, Ra1, Ra2, Le0, Ri0, Ri1, sRa0, sRa1, sRa2]
    ListState = []
    for i in ListState0:
        ListState.append(ListStatet[i])

    # print [l.name for l in ListState]
    for state in ListState:
        model.add_state(state)

    endp = 0.0000001
    for state0 in ListState:
        for state1 in ListState:
            if state1 == state0:
                s0 = 0.166
                s0 = selfprob
                model.add_transition(state0, state1, s0)
            else:
                model.add_transition(state0, state1, (1 - s0 - endp) / (len(ListState) - 1))

    for state in ListState:

        model.add_transition("start", state, 1.0 / len(ListState))
        model.add_transition(state, "end", endp)

    return model


def generate_traj(time, fight=False, diff_sigma=2, deltav=0.4,
                  delta_sigma_directed=6, force_model=None,
                  lower_selfprob=0.4, zeros=True, Ra0=[], Ra1=[], Mu0=[], Mu1=[], sRa0=[],
                  sub=False, clean=None, check_delta=False,
                  alpha=0.5, ndim=2, anisentropy=0.5, rho_fixed=False,
                  random_rotation=False, fixed_self_proba=False):

    global X, Y
    #nstate = np.random.randint(1,6)
    # nstate=6
    #ListState = range(6)
    # np.random.shuffle(ListState)
    #ListState = ListState[:nstate]

    # 0 1 = random
    # 2 3 = Left
    # 4 5 = Right

    # if 1 in ListState and not 0 in ListState:
    #    ListState[ListState.index(1)] = 0
    # if 3 in ListState and not 2 in ListState:
    #    ListState[ListState.index(3)] = 2
    # if 5 in ListState and not 4 in ListState:
    #    ListState[ListState.index(5)] = 4
    if sub:
        clean_number = 9
    else:
        clean_number = 6

    StateN = {"Ra0": 0, "Ra1": 1, "Ra2": 2, "Le0": 3, "Ri0": 4, "Ri1": 5,
              "sRa0": 6, "sRa1": 7, "sRa2": 8}  # ,"Ra3": 3,"Ra4":4,"Ra5":5,"Le0":6}
    iStateN = {v: k for k, v in StateN.items()}

    Model_type = {"D": [0, ["Ra0"]],
                  "Dv": [1, ["Le0"]],
                  "D-D": [2, ["Ra0", "Ra1"]],
                  "D-DvL": [3, ["Ra0", "Le0"]],
                  "DvR-DvL": [4, ["Le0", "Ri0"]],
                  "D-D-D": [5, ["Ra0", "Ra1", "Ra2"]],
                  "D-D-DvL": [6, ["Ra0", "Ra1", "Le0"]],
                  "D-DvL-DvR": [7, ["Ra0", "Le0", "Ri0"]],
                  "D-D-DvL-DvR": [8, ["Ra0", "Ra1", "Le0", "Ri0"]],
                  "DvL-DvR-DvR1": [9, ["Le0", "Ri0", "Ri1"]],
                  "D-DvL-DvR-DvR1": [10, ["Ra0", "Le0", "Ri0", "Ri1"]],
                  "D-D-DvL-DvR-DvR1": [11, ["Ra0", "Ra1", "Le0", "Ri0", "Ri1"]]}

    if sub:
        Model_type1 = {"sD": [12, ["sRa0"]],
                       "D-sD": [13, ["Ra0", "sRa0"]],
                       "sD-sD": [14, ["sRa0", "sRa1"]],
                       "sD-DvL": [15, ["sRa0", "Le0"]],
                       "sD-D-D": [16, ["sRa0", "Ra0", "Ra1"]],
                       "sD-sD-D": [17, ["sRa0", "sRa1", "Ra0"]],
                       "sD-sD-sD": [18, ["sRa0", "sRa1", "sRa2"]],
                       "sD-D-DvL": [19, ["sRa0", "Ra0", "Le0"]],
                       "sD-sD-DvL": [20, ["sRa0", "sRa1", "Le0"]],
                       "sD-DvL-DvR": [21, ["sRa0", "Le0", "Ri0"]],
                       "sD-D-DvL-DvR": [22, ["sRa0", "Ra0", "Le0", "Ri0"]],
                       "sD-sD-DvL-DvR": [23, ["sRa0", "sRa1", "Le0", "Ri0"]],
                       "sD-DvL-DvR-DvR1": [24, ["sRa0", "Le0", "Ri0", "Ri1"]],
                       "sD-D-DvL-DvR-DvR1": [25, ["sRa0", "Ra0", "Le0", "Ri0", "Ri1"]],
                       "sD-sD-DvL-DvR-DvR1": [26, ["sRa0", "sRa1", "Le0", "Ri0", "Ri1"]]
                       }
        Model_type = dict(Model_type.items() + Model_type1.items())

        #}
    if fight:
        Model_type.pop("D-D-DvL-DvR")
        Model_type.pop("D-DvL-DvR-DvR1")
        Model_type.pop("D-D-DvL-DvR-DvR1")
        Model_type["DvL-DvR-DvR1"][0] = 8

    """

    Model_type = {"D":[0,["Ra0"]],"D-D":[1,["Ra0","Ra1"]],"D-D-D":[2,["Ra0","Ra1","Ra2"]]}
    #print ListState
    Model_num =  np.random.randint(0,3)
    ModelN = Model_type.keys()
    ModelN.sort()
    ModelN = ModelN[Model_num]
    Model_num=Model_type[ModelN][0]"""

    iModel = {v[0]: k for k, v in Model_type.items()}
    Model_num = np.random.randint(0, len(Model_type.keys()))

    if force_model is not None:
        if type(force_model) == types.IntType:
            Model_num = force_model
        else:
            Model_num = force_model(np.random.randint(0, len(force_model)))

    #Model_num = 9

    ListState = [StateN[iname] for iname in Model_type[iModel[Model_num]][1]]

    # ListState=[0]
    selfprob = lower_selfprob + (1 - lower_selfprob) * random.random()
    if fixed_self_proba:
        selfprob = lower_selfprob
    # print "Sampling",ListState
    model = one_particle_n_states_s(ListState0=ListState, StateN=StateN, selfprob=selfprob)
    # print "ENdS"
    seq = np.zeros(time)
    sequence = model.sample(time - 1)

    scale = 1 + 9 * random.random()

    cats = scale * np.random.random()

    # diff_sigma=2
    # 1.5

    if Ra0 == []:

        Ra0 = [0, cats]
    else:
        scale = 1

    if Ra1 == []:

        Ra1 = [0, max(diff_sigma * Ra0[1] + scale * np.random.random(), scale)]
    else:
        scale = 1

    Ra2 = [0, max(diff_sigma * Ra1[1] + scale * np.random.random(), scale)]

    R_anisentropy = {"Ra0": 1 - anisentropy + anisentropy * (1 - 2 * np.random.random(3)),
                     "Ra1": 1 - anisentropy + anisentropy * (1 - 2 * np.random.random(3)),
                     "Ra2": 1 - anisentropy + anisentropy * (1 - 2 * np.random.random(3))}

    D = {"Ra0": Ra0, "Ra1": Ra1, "Ra2": Ra2}

    ##############################################
    # Sub
    cats = scale * np.random.random()

    if sRa0 == []:

        sRa0 = [0, cats]
    else:
        scale = 1
    sRa1 = [0, max(diff_sigma * sRa0[1] + scale * np.random.random(), scale)]

    sRa2 = [0, max(diff_sigma * sRa1[1] + scale * np.random.random(), scale)]

    sD = {"sRa0": sRa0, "sRa1": sRa1, "sRa2": sRa2}
    sR_anisentropy = {"sRa0": 1 - anisentropy + anisentropy * (1 - 2 * np.random.random(3)),
                      "sRa1": 1 - anisentropy + anisentropy * (1 - 2 * np.random.random(3)),
                      "sRa2": 1 - anisentropy + anisentropy * (1 - 2 * np.random.random(3))}

    start = {"sRa0": 0, "sRa1": 0, "sRa2": 0}

    #dsub = 0.15*(1-2*np.random.rand(3))
    dsub = [0, 0, 0]
    sds = {iname: create_random_alpha(time + 1, alpha=alpha, ndim=ndim)
           for iname in Model_type[iModel[Model_num]][1] if iname in ["sRa0", "sRa1", "sRa2"]}

    sds = {k: v[1:] - v[:-1] for k, v in sds.items()}

    ################################################################
    # Directed

    namesl = []
    alpha2 = 0.15 * 3.14 + 1.85 * 3.14 * random.random()
    #alpha2 = -3.14
    dalpha2 = max(0.1, 0.8 * np.random.random())
    dalpha1 = max(0.1, 0.8 * np.random.random())

    traj = np.zeros((time, ndim))
    tot0 = 0
    tot1 = 0

    mus1 = scale * (1 - 2 * np.random.random(ndim))

    mus2 = mus1.copy()

    mus3 = mus1.copy()
    #deltav = 0.1
    while np.sqrt(np.sum((mus1 - mus2)**2)) < deltav * scale and \
            np.sqrt(np.sum((mus1 - mus3)**2)) < deltav * scale and \
            np.sqrt(np.sum((mus2 - mus3)**2)) < deltav * scale:

        mus1 = scale * (1 - 2 * np.random.random(ndim))
        mus2 = scale * (1 - 2 * np.random.random(ndim))
        mus3 = scale * (1 - 2 * np.random.random(ndim))

    if random_rotation:

        while np.sqrt(np.sum(mus1**2) - np.sum(mus2)**2) < deltav * scale and \
                np.sqrt(np.sum(mus1**2) - np.sum(mus3)**2) < deltav * scale and \
                np.sqrt(np.sum(mus2**2) - np.sum(mus3)**2) < deltav * scale:

            mus1 = scale * (1 - 2 * np.random.random(ndim))
            mus2 = scale * (1 - 2 * np.random.random(ndim))
            mus3 = scale * (1 - 2 * np.random.random(ndim))

        """

        if mus1[0] < mus2[0]:
            mus1,mus2=mus2,mus1

        if mus1[0] < mus3[0]:
            mus1,mus3=mus3,mus1

        if mus2[0] < mus3[0]:
            mus2,mus3=mus3,mus2
        mus1[1] = 0
        """

    epsilon = 1e-7

    rho1 = np.random.random(1 + (ndim - 2) * 2) + epsilon
    sigmas1 = scale * np.random.random(ndim) + epsilon

    rho2 = np.random.random(1 + (ndim - 2) * 2) + epsilon
    sigmas2 = scale * np.random.random(ndim) + epsilon

    rho3 = np.random.random(1 + (ndim - 2) * 2) + epsilon
    sigmas3 = scale * np.random.random(ndim) + epsilon
    # if np.sum(mus1**2) > np.sum(mus2**2):
    #    mus1,mus2=mus2,mus1

    if rho_fixed:
        rho1 = [0, 0]
        rho2 = [0, 0]
        rho3 = [0, 0]

    # Before 4

    # Borned:

    d1 = np.sqrt(np.sum(mus1**2)) + epsilon
    """
    if d1 < 0.01:
        mus1 = 0.01*mus1/d1
    """
    for indim in range(ndim):
        sigmas1[indim] = min(d1 / delta_sigma_directed, sigmas1[indim])

    d2 = np.sqrt(np.sum(mus2**2)) + epsilon
    """
    if d2 < 0.01:
        mus2 = 0.01*mus2/d2
    """
    for indim in range(ndim):
        sigmas2[indim] = min(d2 / delta_sigma_directed, sigmas2[indim])

    d3 = np.sqrt(np.sum(mus3**2)) + epsilon
    """
    if d3 < 0.01:
        mus3 = 0.01*mus3/d3
    """
    for indim in range(ndim):
        sigmas3[indim] = min(d3 / delta_sigma_directed, sigmas3[indim])

    if Mu0 != []:
        mus1 = Mu0[0]
        sigmas1 = Mu0[1]
        rho1 = Mu0[2]

    if Mu1 != []:
        mus2 = Mu1[0]
        sigmas2 = Mu1[1]
        rho2 = Mu1[2]
    # mus2 = scale*

    """
    if Model_num == 3:
        print "Scale",scale
        print mus1,sigmas1
        print Ra0

    """
    if ndim == 2:
        scf = 2**0.5

        def get_covariance(sigmasm, rhom):
            return [[sigmasm[0] * scf, sigmasm[0] * sigmasm[1] * rhom[0] * scf],
                    [sigmasm[0] * sigmasm[1] * rhom[0] * scf, sigmasm[1] * scf]]

    if ndim == 3:
        scf = 2**0.5

        def get_covariance(sigmasm, rhom):
            return [[sigmasm[0] * scf, sigmasm[0] * sigmasm[1] * rhom[0] * scf, sigmasm[0] * sigmasm[2] * rhom[1] * scf],
                    [sigmasm[0] * sigmasm[1] * rhom[0] * scf, sigmasm[1] * scf,
                     sigmasm[1] * sigmasm[2] * rhom[2] * scf],
                    [sigmasm[0] * sigmasm[2] * rhom[1] * scf, sigmasm[1] * sigmasm[2] * rhom[2] * scf, sigmasm[2] * scf]]

    # print mus1
    SigmasMu = {'Le0': [mus1, get_covariance(sigmas1, rho1)],
                "Ri0": [mus2, get_covariance(sigmas2, rho2)],
                "Ri1": [mus3, get_covariance(sigmas3, rho3)]}
    StartMu = {"Le0": 0, "Ri0": 0, "Ri1": 0}

    Mus = {iname: np.random.multivariate_normal(SigmasMu[iname][0], SigmasMu[iname][1], time)
           for iname in Model_type[iModel[Model_num]][1] if iname in ["Le0", "Ri0", "Ri1"]}
    # MODIF

    transition = False
    for tt, v in enumerate(sequence):

        if tt > 0:
            if sequence[tt - 1] != sequence[tt]:
                transition = True
            else:
                transition = False

        #seq[tt] = int(round(v,0))
        seq[tt] = StateN[v]

        name = iStateN[seq[tt]]
        if name not in Model_type[iModel[Model_num]][1]:
            print(name, v)
            print("CHosen", Model_num)
            print("Allowed", Model_type[iModel[Model_num]][1])

            raise
        namesl.append(name)

        if name in ["Ra0", "Ra1", "Ra2", "Ra3", "Ra4", "Ra5"]:
            traj[tt][0] = np.random.normal(D[name][0], D[name][1] * scf * R_anisentropy[name][0])
            traj[tt][1] = np.random.normal(D[name][0], D[name][1] * scf * R_anisentropy[name][1])
            if ndim == 3:
                traj[tt][2] = np.random.normal(
                    D[name][0], D[name][1] * scf * R_anisentropy[name][2])

        if name in ["sRa0", "sRa1", "sRa2"]:
            # print  sD[name][1]
            # print start[name]
            traj[tt][0] = sD[name][1] * sds[name][start[name]][0] * scf * sR_anisentropy[name][0]
            traj[tt][1] = sD[name][1] * sds[name][start[name]][1] * scf * sR_anisentropy[name][1]
            if ndim == 3:
                traj[tt][2] = sD[name][1] * sds[name][start[name]][2] * scf * sR_anisentropy[name][2]

            start[name] += 1

        if name in ["Le0", "Ri0", "Ri1"]:

            if transition and random_rotation:
                Mus[name] = random_rot(Mus[name], ndim=ndim, centered=False)

            #theta = dalpha1*(1-2*random.random())
            #Dist = max(0.001,np.random.normal(D[name][0],D[name][1]))
            if ndim == 2:
                x, y = Mus[name][StartMu[name]]
            elif ndim == 3:
                x, y, z = Mus[name][StartMu[name]]
            traj[tt][0] = x
            traj[tt][1] = y
            if ndim == 3:
                traj[tt][2] = y

            StartMu[name] += 1

            #tot0 += Dist
    if check_delta:
        print(sD)
        # print sD

    def down_grade(seq, Model_num, clean=None):

        seq = np.array(seq)

        if clean:
            filteseq = np.ones_like(seq, dtype=np.bool)

            for icat in StateN.keys():
                if np.sum(seq == StateN[icat]) <= clean:
                    filteseq[seq == StateN[icat]] = False
            seq0 = copy.deepcopy(seq)

            seq = seq0[filteseq]

        realname = list(set(seq.tolist()))

        Nrealcat = [iStateN[ireal] for ireal in realname]

        realname.sort()
        realname = [iStateN[ir] for ir in realname]
        translate = {StateN[iname]: i for i, iname in enumerate(realname)}
        Nrealcat = realname
        bNrealcat = copy.deepcopy(Nrealcat)

        if "Ra1" in Nrealcat and not "Ra0" in Nrealcat:
            seq[seq == StateN["Ra1"]] = StateN["Ra0"]

        realname = list(set(seq))

        Nrealcat = [iStateN[ireal] for ireal in realname]

        if "Ra2" in Nrealcat and not "Ra1" in Nrealcat:
            if "Ra0" in Nrealcat:
                seq[seq == StateN["Ra2"]] = StateN["Ra1"]

            else:
                seq[seq == StateN["Ra2"]] = StateN["Ra0"]

        if "sRa1" in Nrealcat and not "sRa0" in Nrealcat:
            seq[seq == StateN["sRa1"]] = StateN["sRa0"]

        realname = list(set(seq))

        Nrealcat = [iStateN[ireal] for ireal in realname]

        if "sRa2" in Nrealcat and not "sRa1" in Nrealcat:
            if "sRa0" in Nrealcat:
                seq[seq == StateN["sRa2"]] = StateN["sRa1"]

            else:
                seq[seq == StateN["sRa2"]] = StateN["sRa0"]

        if "Ri0" in Nrealcat and not "Le0" in Nrealcat:
            seq[seq == StateN["Ri0"]] = StateN["Le0"]

        realname = list(set(seq))

        Nrealcat = [iStateN[ireal] for ireal in realname]
        if "Ri1" in Nrealcat and not "Ri0" in Nrealcat:
            if "Le0" in Nrealcat:
                seq[seq == StateN["Ri1"]] = StateN["Ri0"]

            else:
                seq[seq == StateN["Ri1"]] = StateN["Le0"]

        realname = list(set(seq))

        Nrealcat = [iStateN[ireal] for ireal in realname]
        Nrealcat.sort()
        # Classing by frequencies

        if "Ri0" in Nrealcat and "Le0" in Nrealcat:
            if np.sum(seq == StateN["Ri0"]) > np.sum(seq == StateN["Le0"]):
                seq[seq == StateN["Ri0"]] = 1000
                seq[seq == StateN["Le0"]] = StateN["Ri0"]
                seq[seq == 1000] = StateN["Le0"]

        if "Ri0" in Nrealcat and "Le0" in Nrealcat and "Ri1" in Nrealcat:
            freq = [[np.sum(seq == StateN["Le0"]), "Le0"],
                    [np.sum(seq == StateN["Ri0"]), "Ri0"],
                    [np.sum(seq == StateN["Ri1"]), "Ri1"]]

            freq.sort()
            freq = freq[::-1]
            seq1 = copy.deepcopy(seq)

            seq1[seq == StateN[freq[0][1]]] = StateN["Le0"]
            seq1[seq == StateN[freq[1][1]]] = StateN["Ri0"]
            seq1[seq == StateN[freq[2][1]]] = StateN["Ri1"]

            seq = seq1

        # Dowgrading models:

        found = False
        for k, v in Model_type.items():
            cats = v[1]
            cats.sort()
            if cats == Nrealcat:
                Model_num = v[0]
                found = True
                break
        if not found:
            if len(seq) == 0:
                raise EmptyError()
            print(seq, time)
            print(Model_type)
            print("CHosen", Model_num)
            print(bNrealcat)
            print(Nrealcat)
            raise "nimportquei"

        if clean:
            seq0[filteseq] = seq
            invfilteseq = np.array([not ifilte for ifilte in filteseq], dtype=np.bool)
            seq0[invfilteseq] = clean_number

            seq = seq0

        return seq, Model_num

    #print("Before", seq)
    seq, Model_num = down_grade(seq, Model_num, clean)
    #print("After", seq)

    # print translate
    startc = [[0, 0]]
    if ndim == 3:
        startc = [[0, 0, 0]]
    traj = np.cumsum(np.concatenate((startc, traj)), axis=0)

    # Random nan in seq
    nzeros = np.random.randint(0, 10)
    Z = [i for i, v in enumerate(seq) if v == clean_number]
    if zeros:

        for i in range(nzeros):
            Z.append(np.random.randint(len(seq) - 1))
            seq[Z[-1]] = clean_number
            #traj[Z[-1]:Z[-1]+1,0] = 0
            #traj[Z[-1]:Z[-1]+1,1] = 0

        Z = list(set(Z))

    """
    normed= [copy.deepcopy( np.sqrt(np.sum((traj[1:]-traj[:-1])**2,axis=1)))]
    #print normed[0].shape
    normed.append((traj[1:,0]-traj[:-1,0])/normed[0])
    normed.append((traj[1:,1]-traj[:-1,1])/normed[0])

    normed = np.array(normed).T

    #print normed.shape
    normed[::,0] = normed[::,0]-np.mean(normed[::,0])
    normed[::,0] /= np.std(normed[::,0])
    """
    ModelN = len(set(namesl))
    return ModelN, Model_num, seq, seq, traj, [], Z

if __name__ == "__main__":
    from prePostTools import get_parameters
    ModelN, Model_num, s, sc, traj, normed, alpha2 = generate_traj(200, fight=False, sub=True,
                                                                   zeros=False, clean=2, ndim=2, anisentropy=0,
                                                                   force_model=4, Mu0=[[5, 0], [1, 1], [0]],
                                                                   random_rotation=True)
    print(ModelN, Model_num)
    f = figure(figsize=(15, 10))

    ax = f.add_subplot(141)

    plot(s)
    ax = f.add_subplot(142)
    plot(sc)
    ax = f.add_subplot(143)
    # print traj.shape
    plot_label(traj[::, :2], sc, remove6=9)

    print(get_parameters(traj, s, 1, 1, 2))

    axis("equal")

    print(traj.shape)

    print(np.sum(s == 3), np.sum(s == 4), np.sum(s == 5))

    print(alpha2)
    print(sc)


# print normed.shape
