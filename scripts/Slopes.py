import tensorflow as tf
import numpy as np


def slope_checker(m: tf.keras.Sequential, X: list, /, max_delta=0.00001) -> list[tuple]:
    breakpoints = []
    predictions = m.predict(X, verbose="0")
    prev_point = (X[0], predictions[0][0])
    current_point = (X[1], predictions[1][0])

    for nxt in zip(X[2:], predictions[2:]):
        nxt_point = (nxt[0], nxt[1][0])
        #nxt_point = (nxt, m.predict([nxt], verbose=0)[0][0])
        if abs(slope(prev_point, current_point) - slope(current_point, nxt_point)) > max_delta:
            # print("slope1 = ", slope(prev_point, current_point))
            # print("slope2 = ", slope(current_point, nxt_point))
            # print("change at ", current_point)
            # print()
            breakpoints.append(current_point)
        prev_point = current_point
        current_point = nxt_point
    return breakpoints


def breakpoint_finder(m: tf.keras.Sequential, X: list[float]) -> list[tuple[float, float]]:
    """Finds the places where the model's activation pattern changes, arvestab sellega et on 2 outputi modelil, mean ja variance
    WORKS FOR 2 LAYER MODEL ONLY

    :param m: model (keras sequential), all layers have to have same width
    :param X: input range to look for breakpoints
    :return: array of (x_bpoint, y_bpoint), array(pattern)
    """

    #X = np.linspace(-1,1, 100)

    layer_outputs = [layer.output for layer in m.layers]
    ll_outputing_model = tf.keras.Model(inputs=m.input, outputs=layer_outputs)

    results = ll_outputing_model.predict([X])
    different_paterns = set()

    prev_patern = np.concatenate(
        [np.nonzero(results[0][0])[0], np.nonzero(results[1][0])[0]])

    b_points = []

    n_layer = m.layers[0].weights[0].shape[1]

    # hakkame loopima alates teine element, esimene juba on prev_paternis
    # results[2][1:][:,:1] = results[viimase kihi][alates teine tul][ainult esimene väärtus tuplest, sest see on mean, teine on variance]
    for l1, l2, point in zip(results[0][1:], results[1][1:], zip(X[1:], results[2][1:][:,:1].flatten())):
        #print(point)
        # print("------")
        patern = np.concatenate([np.nonzero(l1)[0], np.nonzero(l2)[0] + n_layer])

        if len(prev_patern) != len(patern) or not (prev_patern == patern).all():
            prev_patern = patern
            if tuple(patern) in different_paterns:
                print("EKSISTEERIVAD MITTEPIDEVAD PIIRKONNAD!!!!")
            different_paterns.add(tuple(patern))
            b_points.append((point, patern))
    return b_points


def slope(p1: tuple, p2: tuple) -> float:
    return (p2[1]-p1[1]) / (p2[0]-p1[0])


# from itertools import tee
# def pairwise(it):
#     a, b = tee(it)
#     next(b, None)
#     return zip(a,b)
