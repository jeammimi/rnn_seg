import scipy
import numpy as np
from numpy import mean, dot
from numpy import cross, eye
from scipy.linalg import expm3, norm
import copy
from math import ceil, log
import math

###########################
# Tools


def in_sphere(X, R):
    # print X <= R
    if np.all(X <= R) and np.sum(X**2) <= R**2:
        return True
    return False


def reflect(X, dX):
    norm = np.sum(X**2)**0.5
    norm = X / norm
    parallel = np.sum(norm * dX) * norm
    # print parallel
    return dX - 2 * parallel


def M(axis, theta):
    return expm3(cross(eye(3), axis / norm(axis) * theta))


def random_rot(traj, alpha=None, ndim=2, axis=[], centered=True):

    if ndim == 2:
        if alpha is None:
            alpha = 2 * 3.14 * np.random.random()
        if axis == []:
            axis = [[np.cos(-alpha), np.cos(-alpha + 3.14 / 2)],
                    [np.sin(-alpha), np.sin(-alpha + 3.14 / 2)]]

        axis = np.array(axis)

    if ndim == 3:
        if alpha is None:
            alpha = 3.14 * np.random.random()
        if axis == []:
            axis = np.random.random(3)

        axis = M(axis, alpha)

        # print axis

    # print axis.shape
    if centered:
        newtraj = (traj - mean(traj.T, axis=1)).T
    else:
        newtraj = traj.T

    return dot(axis.T, newtraj).T


def create_random_alpha(N, alpha=0.5, drift=[0, 0, 0], ndim=2):
    coord = []
    hurstExponent = alpha / 2.

    L = int(ceil(log(N) / log(2)))

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
    return np.array(coord)[:N, 1:]


def fractional_1D(size, H):
    # numpy.random.seed(0)
    N = size
    HH = 2 * H
    covariance = np.zeros((N, N))
    I = np.indices((N, N))

    covariance = abs(I[0] - I[1])
    covariance = (abs(covariance - 1)**HH + (covariance + 1)**HH - 2 * covariance**HH) / 2.

    w, v = np.linalg.eig(covariance)
    ws = w ** 0.25
    v = ws[np.newaxis, ::] * v
    A = np.inner(v, v)
    x = np.random.randn((N))
    eta = np.dot(A, x)
    xfBm = np.cumsum(eta)
    return np.concatenate([[0], xfBm])
#############################
# Motions


def diffusive(scale, ndim, time, epsilon=1e-7):

    mus1 = np.zeros((ndim))
    # rho1 = np.random.random(1 + (ndim - 2) * 2) + epsilon
    sigmas1 = scale * np.random.random(1) + epsilon

    cov = sigmas1 * np.eye(ndim) + epsilon
    traj = np.cumsum(np.random.multivariate_normal(mus1, cov, time), axis=0)
    alpha = 2 * 3.14 * np.random.rand()

    return random_rot(traj, alpha, ndim)[:time]


def diffusive_confined(scale, ndim, time, epsilon=1e-7):

    mus1 = np.zeros((ndim))
    # rho1 = np.random.random(1 + (ndim - 2) * 2) + epsilon
    sigmas1 = scale * np.random.random(1) + epsilon

    cov = sigmas1 * np.eye(ndim) + epsilon
    anisentropy = 0.2 * np.random.rand()
    cov[1, 1] *= anisentropy
    traj = np.cumsum(np.random.multivariate_normal(mus1, cov, time), axis=0)
    alpha = 2 * 3.14 * np.random.rand()

    return random_rot(traj, alpha, ndim)[:time]


def directed(scale, ndim, time, epsilon=1e-7, delta_sigma_directed=6):

    mus1 = scale * (1 - 2 * np.random.random(ndim))
    rho1 = np.random.random(1 + (ndim - 2) * 2) + epsilon
    sigmas1 = scale * np.random.random(ndim) + epsilon

    d1 = np.sqrt(np.sum(mus1**2)) + epsilon
    if d1 < 0.01:
        mus1 = 0.01 * mus1 / d1

    for indim in range(ndim):
        sigmas1[indim] = min(d1 / delta_sigma_directed, sigmas1[indim])

    if ndim == 2:
        scf = 2**0.5

        def get_covariance(sigmasm, rhom):
            return [[sigmasm[0] * scf, sigmasm[0] * sigmasm[1] * rhom[0] * scf],
                    [sigmasm[0] * sigmasm[1] * rhom[0] * scf, sigmasm[1] * scf]]

    if ndim == 3:
        scf = 2**0.5

        def get_covariance(sigmasm, rhom):
            return [[sigmasm[0] * scf, sigmasm[0] * sigmasm[1] * rhom[0] * scf, sigmasm[0] * sigmasm[2] * rhom[1] * scf],
                    [sigmasm[0] * sigmasm[1] * rhom[0] * scf, sigmasm[1]
                        * scf,  sigmasm[1] * sigmasm[2] * rhom[2] * scf],
                    [sigmasm[0] * sigmasm[2] * rhom[1] * scf, sigmasm[1] * sigmasm[2] * rhom[2] * scf, sigmasm[2] * scf]]

    cov = get_covariance(sigmas1, rho1)
    traj = np.cumsum(np.random.multivariate_normal(mus1, cov, time), axis=0)
    alpha = 2 * 3.14 * np.random.rand()

    return random_rot(traj, alpha, ndim)[:time]


def accelerated(scale, ndim, time, epsilon=1e-7, delta_sigma_directed=6):

    mus1 = scale * (1 - 2 * np.random.random(ndim))
    rho1 = np.random.random(1 + (ndim - 2) * 2) + epsilon
    sigmas1 = scale * np.random.random(ndim) + epsilon

    accelerated = 2 + 2 * np.random.rand()

    d1 = np.sqrt(np.sum(mus1**2)) + epsilon
    if d1 < 0.01:
        mus1 = 0.01 * mus1 / d1

    for indim in range(ndim):
        sigmas1[indim] = min(d1 / delta_sigma_directed, sigmas1[indim])

    if ndim == 2:
        scf = 2**0.5

        def get_covariance(sigmasm, rhom):
            return [[sigmasm[0] * scf, sigmasm[0] * sigmasm[1] * rhom[0] * scf],
                    [sigmasm[0] * sigmasm[1] * rhom[0] * scf, sigmasm[1] * scf]]

    if ndim == 3:
        scf = 2**0.5

        def get_covariance(sigmasm, rhom):
            return [[sigmasm[0] * scf, sigmasm[0] * sigmasm[1] * rhom[0] * scf, sigmasm[0] * sigmasm[2] * rhom[1] * scf],
                    [sigmasm[0] * sigmasm[1] * rhom[0] * scf, sigmasm[1]
                        * scf,  sigmasm[1] * sigmasm[2] * rhom[2] * scf],
                    [sigmasm[0] * sigmasm[2] * rhom[1] * scf, sigmasm[1] * sigmasm[2] * rhom[2] * scf, sigmasm[2] * scf]]

    cov = get_covariance(sigmas1, rho1)

    acc = np.random.rand() * np.random.multivariate_normal(mus1, cov, time) + \
        mus1 / accelerated * np.array([np.arange(time), np.arange(time),
                                       np.arange(time)]).T[::, :ndim]

    maxi = np.max(np.sum(acc**2, axis=1)**0.5)
    acc = acc * scale / maxi

    return np.cumsum(acc, axis=0)


def slowed(scale, ndim, time, epsilon=1e-7, delta_sigma_directed=6):

    return accelerated(scale, ndim, time, epsilon, delta_sigma_directed)[::-1]


def still(scale, ndim, time, epsilon=1e-7, delta_sigma_directed=6):

    return np.zeros((time, ndim))


def sinusoidal(scale, ndim, time, epsilon=1e-7):

    period = np.random.rand()
    sample_rate = np.random.randint(5, 10)

    t = np.arange(time)
    traj = np.array([scale * t, scale * np.sin(2 * 3.14 *
                                               period * t / sample_rate), np.zeros_like(t)]).T

    alpha = 2 * 3.14 * np.random.rand()

    if ndim == 2:
        traj = traj[::, :2]

    noise = diffusive(scale / 4., ndim, time + 1, epsilon=1e-7)
    return random_rot(traj, alpha, ndim) + noise[:-1] - noise[1:]


def heart(scale, ndim, time):

    percent = 1.05 + 0.5 * np.random.rand()

    real_heart = int(time * percent)

    ratio = 0.5 * np.random.rand()  # xy ratio
    x = scipy.linspace(-2, 2, real_heart / 2)
    y1 = scipy.sqrt(1 - (abs(x) - 1)**2)
    y2 = -3 * scipy.sqrt(1 - (abs(x[::-1]) / 2)**0.5)

    Y = np.concatenate([y1, y2])
    X = ratio * np.concatenate([x, x[::-1]])

    shift = np.random.randint(0, real_heart)
    Y = np.roll(Y, shift)[:time]
    X = np.roll(X, shift)[:time]
    traj = np.array([X, Y, np.zeros_like(Y)]).T

    alpha = 2 * 3.14 * np.random.rand()

    if ndim == 2:
        traj = traj[::, :2]

    noise = diffusive(scale / 800., ndim, time + 1, epsilon=1e-7)
    return scale * random_rot(traj, alpha, ndim) + noise[:-1] - noise[1:]


def subdiffusive_confined(scale, ndim, time):
    alpha = 0.5
    traj = create_random_alpha(time + 1, alpha=alpha, ndim=ndim)
    anisentropy = 0.2 * np.random.rand()
    traj[::, 1] *= anisentropy
    alpha = 2 * 3.14 * np.random.rand()

    return scale * random_rot(traj, alpha, ndim)[:time]


def continuous_time_random_walk(scale, ndim, time):
    alpha = 2 + np.random.rand()
    s = time
    x = np.zeros((s))
    y = np.zeros((s))
    t = 0
    while t < time:
        alpha = 1
        dt = 0.01 * np.power(1 - np.random.rand(), -1 / alpha)

        t += dt
        theta = np.random.rand() * 2 * np.pi

        if int(t) >= time:
            break
        x[int(t)] += np.random.normal(0, 1)
        y[int(t)] += np.random.normal(0, 1)
        if int(t) < time - 1:
            x[int(t + 1):] = x[int(t)]
            y[int(t + 1):] = y[int(t)]

    coord = np.array([x, y]).T
    alpha = 2 * 3.14 * np.random.rand()

    return scale * coord


def continuous_time_random_walk_confined(scale, ndim, time):
    alpha = 2 + np.random.rand()
    s = time
    x = np.zeros((s))
    y = np.zeros((s))
    t = 0
    while t < time:
        alpha = 1
        dt = 0.01 * np.power(1 - np.random.rand(), -1 / alpha)

        t += dt
        theta = np.random.rand() * 2 * np.pi

        if int(t) >= time:
            break
        x[int(t)] += np.random.normal(0, 1)
        y[int(t)] += np.random.normal(0, 1)
        if int(t) < time - 1:
            x[int(t + 1):] = x[int(t)]
            y[int(t + 1):] = y[int(t)]

    anisentropy = 0.2 * np.random.rand()
    coord = np.array([x, y * anisentropy]).T
    alpha = 2 * 3.14 * np.random.rand()

    return scale * random_rot(coord, alpha, ndim)[:time]


def subdiffusive(scale, ndim, time):
    alpha = 0.5
    traj = create_random_alpha(time + 1, alpha=alpha, ndim=ndim)

    alpha = 2 * 3.14 * np.random.rand()
    return scale * random_rot(traj, alpha, ndim)[:time]


def levi_flight(scale, ndim, time):
    alpha = 2 + np.random.rand()
    s = time
    x = np.zeros((s))
    y = np.zeros((s))
    for n in range(1, s):
        theta = np.random.rand() * 2 * np.pi
        f = np.random.rand()**(-1 / alpha)
        x[n] = x[n - 1] + f * np.cos(theta)
        y[n] = y[n - 1] + f * np.sin(theta)

    coord = np.array([x, y]).T
    return scale * coord


def fractionnal_brownian(scale, ndim, time):
    alpha = 0.25
    traj = np.array([fractional_1D(time, alpha) for d in range(ndim)]).T

    alpha = 2 * 3.14 * np.random.rand()
    return scale * random_rot(traj, alpha, ndim)[:time]


def brownian_confined_on_plane(scale, ndim, time):

    traj = diffusive(scale, 2, time)
    traj = np.concatenate((traj, np.zeros((traj.shape[0], 1))), axis=1)

    return random_rot(traj, ndim=3)[:time, :ndim]


def sub_confined_on_plane(scale, ndim, time):

    traj = subdiffusive(scale, 2, time)
    traj = np.concatenate((traj, np.zeros((traj.shape[0], 1))), axis=1)

    return random_rot(traj, ndim=3)[:time, :ndim]


def sub_confined_on_plane_0p7(scale, ndim, time, p=0.7):

    #p = 0.5 + 0.2 * np.random.rand()

    traj = subdiffusive(scale, 3, time + 1)

    to_cut = range(len(traj) - 1)
    N = int(len(to_cut) * p)
    np.random.shuffle(to_cut)

    dtraj = traj[1:] - traj[:-1]
    dtraj[to_cut[:N], 2] = 0

    traj = np.cumsum(dtraj, axis=0)
    return random_rot(traj, ndim=3)[:time, :ndim]


def brownian_confined_on_plane_0p7(scale, ndim, time, p=0.7):

    #p = 0.5 + 0.2 * np.random.rand()

    traj = diffusive(scale, 3, time + 1)

    to_cut = range(len(traj) - 1)
    N = int(len(to_cut) * p)
    np.random.shuffle(to_cut)

    dtraj = traj[1:] - traj[:-1]
    dtraj[to_cut[:N], 2] = 0

    traj = np.cumsum(dtraj, axis=0)
    return random_rot(traj, ndim=3)[:time, :ndim]


def min_contact(ncontact):
    def dec1(func):
        # print "Min contact",ncontact
        def wrapper(*args, **kwargs):
            kwargs["contact"] = True
            response = [0, 0]
            while response[1] < max(1, ncontact):
                response = func(*args, **kwargs)

            return response[0]
        return wrapper
    return dec1


@min_contact(10)
def brownian_confined_in_sphere(scale, ndim, time, show=False, contact=True):

    traj = diffusive(scale, ndim, time + 1)
    delta = traj[1:] - traj[:-1]

    R = (0.2 + 0.6 * np.random.rand()) * scale * np.sqrt(time)
    # New 0.3 + 0.1
    # before 0.2 + 0.6
    # print R
    ftraj = [np.zeros((traj.shape[1]))]
    n_contact = 0
    for v in delta:
        # print v
        if in_sphere(ftraj[-1] + v, R):
            ftraj.append(ftraj[-1] + v)
        else:
            n_contact += 1
            r = reflect(ftraj[-1], v)
            # print r,v
            ftraj.append(ftraj[-1] + r)
    if show:
        print(n_contact)
    traj = np.array(ftraj)
    if contact:
        return random_rot(np.array(traj), ndim=ndim)[:time, :ndim], n_contact
    else:
        return random_rot(np.array(traj), ndim=ndim)[:time, :ndim]


@min_contact(10)
def sub_confined_in_sphere(scale, ndim, time, show=False, contact=True):

    traj = subdiffusive(scale, ndim, time + 1)
    delta = traj[1:] - traj[:-1]

    R = (0.9 + 0.4 * np.random.rand()) * scale * (1.0 * time)**0.33
    R = (1.4 + 1.0 * np.random.rand()) * scale * (1.0 * time)**(0.15 + 0.5 / (time - 20))

    # print R
    ftraj = [np.zeros((traj.shape[1]))]
    n_contact = 0
    for v in delta:
        # print v
        if in_sphere(ftraj[-1] + v, R):
            ftraj.append(ftraj[-1] + v)
        else:
            n_contact += 1
            r = reflect(ftraj[-1], v)
            # print r,v
            ftraj.append(ftraj[-1] + r)

    if show:
        print("N_contact", n_contact)
    traj = np.array(ftraj)
    if contact:
        return random_rot(np.array(traj), ndim=ndim)[:time, :ndim], n_contact
    else:
        return random_rot(np.array(traj), ndim=ndim)[:time, :ndim]


def generate_traj_general():
    pass

if __name__ == "__main__":
    """
    print sinusoidal(1,2,10).shape

    #X = sinusoidal(1,3,200)
    #plot(X[::,0],X[::,1])

    X = sub_confined_in_sphere(1,3,400,show=True)

    #X = brownian_confined_on_plane_0p7(1,3,40)
    #for i in range(10):
    #     brownian_confined_in_sphere(1,3,40,show=True)

    #X = fb_confined_in_sphere(1,2,400)

    print X.shape

    plot(X[::,0],X[::,2],"-o")"""
