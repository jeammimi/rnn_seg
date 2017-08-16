print(__package__)

import theano
# theano.config.mode = "FAST_COMPILE"

from ..data.generate_n_steps_flexible import generate_n_steps as Flexible
from ..data.generate_n_steps import generate_n_steps as BDSD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import threading
import os

# sys.path.append("../features")
# print(__name__)

# print(sys.path)


import numpy as np
ndim = 2
np.random.seed(6)


def generator(**kwargs):
    # n_steps_before_change = kwargs.get("n_steps_before_change", 50)
    step = 0
    Step = kwargs.get("Step", {0: 26, 1: 50, 2: 100, 3: 200, 4: 400})
    # Step = kwargs.get("Step", {0: 26, 1: 26, 2: 26, 3: 26, 4: 26})

    while True:
        n_steps = int(step // 50) % len(Step)

        if kwargs.get("model", None):
            n_steps = len(kwargs.get("model").history.epoch) % len(Step)

        if kwargs.get("validation", None):
            n_steps = step % len(Step)

        if kwargs["type"] == "flexible":
            X, Y, Trajs = Flexible(kwargs["size_sample"], Step[n_steps], kwargs["ndim"])

            if kwargs.get("traj", False):
                yield X, Y, Trajs
            else:
                yield X, Y
        elif kwargs["type"] == "BDSD":
            X, Y, Y_cat, Trajs = BDSD(kwargs["size_sample"], Step[
                                      n_steps], kwargs["ndim"], kwargs["sub"])
            if kwargs.get("traj", False):
                yield X, {"category": Y_cat, "output": Y}, Trajs
            else:

                # print(X.shape, step)
                # if kwargs.get("model", None):
                #    print(kwargs.get("model").history.epoch)
                if kwargs.get("old", False):

                    yield {"input1": X, "category": Y_cat, "output": Y}
                else:
                    yield X, {"category": Y_cat, "output": Y}

        step += 1


class createBatchGenerator:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        kwargs = self.kwargs
        with self.lock:
            step = 0
            Step = kwargs.get("Step", {0: 26, 1: 50, 2: 100, 3: 200, 4: 400})
            # Step = kwargs.get("Step", {0: 26, 1: 26, 2: 26, 3: 26, 4: 26})

            while True:
                n_steps = int(step // 50) % len(Step)

                if kwargs.get("model", None):
                    n_steps = len(kwargs.get("model").history.epoch) % len(Step)

                if kwargs.get("validation", None):
                    n_steps = step % len(Step)

                if kwargs["type"] == "flexible":
                    X, Y, Trajs = Flexible(kwargs["size_sample"], Step[n_steps], kwargs["ndim"])

                    if kwargs.get("traj", False):
                        yield X, Y, Trajs
                    else:
                        yield X, Y
                elif kwargs["type"] == "BDSD":
                    X, Y, Y_cat, Trajs = BDSD(kwargs["size_sample"], Step[
                                              n_steps], kwargs["ndim"], kwargs["sub"])
                    if kwargs.get("traj", False):
                        yield X, {"category": Y_cat, "output": Y}, Trajs
                    else:

                        # print(X.shape, step)
                        # if kwargs.get("model", None):
                        #    print(kwargs.get("model").history.epoch)
                        if kwargs.get("old", False):

                            yield {"input1": X, "category": Y_cat, "output": Y}
                        else:
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
    parser.add_argument('--old', dest='old', action='store_true')

    parser.add_argument('--no-segmentation', dest='segmentation', action='store_false')
    parser.add_argument('--sub', dest='sub', action='store_true')
    parser.add_argument('--Nepochs', default=200, type=int)
    parser.add_argument('--average', dest='merge_mode', action='store_true')

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

    type_traj = "BDSD"
    if args.segmentation is False:
        type_traj = "flexible"

    merge_mode = "concat"
    if args.merge_mode:
        merge_mode = "ave"

    if not args.old:
        from .build_model import build_model

        model = build_model(n_states=n_states, n_cat=n_cat, n_layers=args.NLayers,
                            inputsize=inputsize, hidden=args.hidden, simple=args.simple,
                            segmentation=args.segmentation, merge_mode=merge_mode)
        Generator = lambda model, validation: generator(size_sample=50, n_steps_before_change=50,
                                                        sub=args.sub, type=type_traj, ndim=args.Ndim, model=model, validation=validation, old=args.old)

    else:
        from .build_model_old import return_layer_paper

        model = return_layer_paper(ndim=2, inside=args.hidden, permutation=True, inputsize=inputsize, simple=False,
                                   n_layers=3, category=True, output=True, n_cat=n_cat, sub=args.sub)

        Generator = lambda model, validation: createBatchGenerator(size_sample=50, n_steps_before_change=50,
                                                                   sub=args.sub, type=type_traj, ndim=args.Ndim, model=model, validation=validation, old=args.old)

    # for epochs in range(args.Nepochs):
    if not args.old:
        Check = ModelCheckpoint(filepath="./data/" + args.dir + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                                save_best_only=False, save_weights_only=True, mode='auto', period=5)
    else:
        Check = ModelCheckpoint(filepath="./data/" + args.dir + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                                save_best_only=False, save_weights_only=True, mode='auto')
    Reduce = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.01)

    if not args.old:
        Log = CSVLogger(filename="./data/" + args.dir + "/training.log")
        model.fit_generator(generator=Generator(model, False), steps_per_epoch=45,
                            validation_steps=5, epochs=args.Nepochs, workers=1,
                            callbacks=[Reduce, Check, Log], validation_data=Generator(model, True),
                            max_q_size=10)
    else:
        gen = generator(size_sample=20 * 50, sub=args.sub, type=type_traj,
                        ndim=args.Ndim, validation=True, old=args.old)
        if os.path.exists("./data/" + args.dir + "/training.log"):
            os.remove("./data/" + args.dir + "/training.log")
        Reduce.on_train_begin()
        for i in range(args.Nepochs):

            for data in gen:
                data = data
                break

            print(data["input1"].shape)
            Log = CSVLogger(filename="./data/" + args.dir + "/training.log", append=True)

            r = model.fit(data, batch_size=20, nb_epoch=1,
                          callbacks=[Check, Log], validation_split=0.1)
            from IPython import embed
            embed()
            print(r["val_loss"])

            if i % 5 == 0:
                Reduce.model = model
                Reduce.on_epoch_end(i)
