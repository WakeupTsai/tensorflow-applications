#!/usr/bin/env python

__all__ = (
    'detect',
    'post_process',
)


import collections
import itertools
import math
import sys

import cv2
import numpy
import tensorflow as tf

import common
import model


def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5
    shape = (im.shape[0] / ratio, im.shape[1] / ratio)

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield cv2.resize(im, (shape[1], shape[0]))


def detect(im, param_vals):
    scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))

    x, y, params = model.get_detect_model()

    with tf.Session() as sess:
        y_vals = []
        for scaled_im in scaled_ims:
            feed_dict = {x: numpy.stack([scaled_im])}
            feed_dict.update(dict(zip(params, param_vals)))
            y_vals.append(sess.run(y, feed_dict=feed_dict))

    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] >
                                                       -math.log(1./0.99 - 1)):
            letter_probs = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    7, len(common.CHARS)))
            letter_probs = common.softmax(letter_probs)

            img_scale = float(im.shape[0]) / scaled_im.shape[0]

            bbox_tl = window_coords * (8, 4) * img_scale
            bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

            present_prob = common.sigmoid(
                               y_val[0, window_coords[0], window_coords[1], 0])

            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs


def _overlaps(match1, match2):
    bbox_tl1, bbox_br1, _, _ = match1
    bbox_tl2, bbox_br2, _, _ = match2
    return (bbox_br1[0] > bbox_tl2[0] and
            bbox_br2[0] > bbox_tl1[0] and
            bbox_br1[1] > bbox_tl2[1] and
            bbox_br2[1] > bbox_tl1[1])


def _group_overlapping_rectangles(matches):
    matches = list(matches)
    num_groups = 0
    match_to_group = {}
    for idx1 in range(len(matches)):
        for idx2 in range(idx1):
            if _overlaps(matches[idx1], matches[idx2]):
                match_to_group[idx1] = match_to_group[idx2]
                break
        else:
            match_to_group[idx1] = num_groups
            num_groups += 1

    groups = collections.defaultdict(list)
    for idx, group in match_to_group.items():
        groups[group].append(matches[idx])

    return groups


def post_process(matches):
    groups = _group_overlapping_rectangles(matches)

    for group_matches in groups.values():
        mins = numpy.stack(numpy.array(m[0]) for m in group_matches)
        maxs = numpy.stack(numpy.array(m[1]) for m in group_matches)
        present_probs = numpy.array([m[2] for m in group_matches])
        letter_probs = numpy.stack(m[3] for m in group_matches)

        yield (numpy.max(mins, axis=0).flatten(),
               numpy.min(maxs, axis=0).flatten(),
               numpy.max(present_probs),
               letter_probs[numpy.argmax(present_probs)])


def letter_probs_to_code(letter_probs):
    return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))


if __name__ == "__main__":
    im = cv2.imread(sys.argv[1])
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

    f = numpy.load(sys.argv[2])
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
    print ("RUN")

    for pt1, pt2, present_prob, letter_probs in post_process(
                                                  detect(im_gray, param_vals)):
        pt1 = tuple(reversed(map(int, pt1)))
        pt2 = tuple(reversed(map(int, pt2)))


        code = letter_probs_to_code(letter_probs)

        color = (0.0, 255.0, 0.0)
        cv2.rectangle(im, pt1, pt2, color)

        cv2.putText(im,
                    code,
                    pt1,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (0, 0, 0),
                    thickness=5)

        cv2.putText(im,
                    code,
                    pt1,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 255, 255),
                    thickness=2)

    cv2.imwrite(sys.argv[3], im)
