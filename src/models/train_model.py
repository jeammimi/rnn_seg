print(__package__)

import theano
#theano.config.mode = "FAST_COMPILE"

from .build_model import build_model
from ..data.generate_n_steps_flexible import generate_n_steps as Flexible
from ..data.generate_n_steps import generate_n_steps as BDSD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# sys.path.append("../features")
# print(__name__)

# print(sys.path)


import numpy as np
ndim = 2
np.random.seed(6)


def generator(**kwargs):
    n_steps_before_change = kwargs.get("n_steps_before_change", 50)
    step = 0
    Step = kwargs.get("Step", {0: 26, 1: 50, 2: 100, 3: 200, 4: 400})
    #Step = kwargs.get("Step", {0: 26, 1: 26, 2: 26, 3: 26, 4: 26})

    while True:
        n_steps = step % len(Step)
        if kwargs["type"] == "flexible":
            X, Y, Trajs = Flexible(kwargs["size_sample"], Step[n_steps], kwargs["ndim"])
            yield X, Y
        elif kwargs["type"] == "BDSD":
            X, Y, Y_cat = BDSD(kwargs["size_sample"], Step[n_steps], kwargs["ndim"], kwargs["sub"])

            yield X, {"category": Y_cat, "output": Y}
        step += 1


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--NLayers', default=3, type=int)
    parser.add_argument('--Ndim', default=3, type=int)
    parser.add_argument('--dir', type=str)

    parser.add_argument('--hidden', default=50, type=int)
    parser.add_argument('--simple', dest='simple', action='store_true')
    parser.add_argument('--no-segmentation', dest='segmentation', action='store_false')
    parser.add_argument('--sub', dest='sub', action='store_true')
    parser.add_argument('--Nepochs', default=200, type=int)

    args = parser.parse_args()

    print(args.simple)
    if args.sub:
        n_cat = 27
        n_states = 10
    else:
        n_cat = 12
        n_states = 7

    if args.Ndim == 3:
        inputsize = 6
    elif args.Ndim == 2:
        inputsize = 5

    model = build_model(n_states=n_states, n_cat=n_cat, n_layers=args.NLayers,
                        inputsize=inputsize, hidden=args.hidden, simple=args.simple)

    Generator = generator(size_sample=10, n_steps_before_change=50,
                          sub=args.sub, type="BDSD", ndim=args.Ndim)
    # for epochs in range(args.Nepochs):
    Check = ModelCheckpoint(filepath="./data/" + args.dir + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                            save_best_only=False, save_weights_only=True, mode='auto', period=5)
    Reduce = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.01)

    model.fit_generator(generator=Generator, steps_per_epoch=45,
                        validation_steps=5, epochs=args.Nepochs, workers=2,
                        callbacks=[Reduce, Check], validation_data=Generator,
                        max_q_size=20)
