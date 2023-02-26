
import asyncio
import time
import sys


import numpy as np
import random 
import tensorflow as tf

from tensorflow import keras
from keras import layers

from matplotlib import pyplot as plt

from CustomCallbacks import CustomLogger
from utils import utils
#from Slopes import Slopes

utils.reset_seeds(0)


async def do_work(s: str, delay_s: float = 1.0):
    print(f"{s} started")
    await asyncio.sleep(delay_s)
    print(f"{s} done")

async def treeni_stuff(seed):
    utils.reset_seeds(seed)

    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(units=5, activation='relu', name="layer_1"),
        layers.Dense(units=5, activation='relu', name="layer_2"),
        layers.Dense(units=2, activation='linear', name="layer_3")
    ])

    fn = lambda x: 0.5*x**3 + 0.2*x**2
    X, y = utils.train_data_maker(
        [(-0.4, 0.2, 5, 0.2),
        (-0.1, 0.1, 5, 0.1),
        (0.2, 0.1, 5, 0.1),
        (0.8, 0.2, 5, 0.2)],
        fn=fn
    )

    X_train, y_train = X, y
    X_valid, y_valid = np.linspace(-1, 1, 10000), fn(np.linspace(-1, 1, 10000))

    model.compile(
        optimizer=keras.optimizers.Adam(
        learning_rate=0.001, amsgrad=True, epsilon=0.01),
        loss=utils.neg_log_likelihood,
    )

    h = model.fit(X, y, batch_size=8, epochs=500, verbose=0,
                    callbacks=[CustomLogger(100)], shuffle=True)

    print(f"done seed{seed}")


async def main():
    start = time.perf_counter()

    todo = ['get package', 'laundry', 'bake cake']

    # tasks = [asyncio.create_task(do_work(item)) for item in todo]
    # done, pending = await asyncio.wait(tasks)
    # for task in done:
    #     result = task.result()

    tasks = [asyncio.create_task(treeni_stuff(item)) for item in todo]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # coros = [do_work(item) for item in todo]
    # results = await asyncio.gather(*coros, return_exceptions=True)

    # async with asyncio.TaskGroup() as tg:  # Python 3.11+
    #     tasks = [tg.create_task(do_work(item)) for item in todo]

    end = time.perf_counter()
    print(f"it took: {end - start:.2f}s")


if __name__ == '__main__':
    asyncio.run(main())