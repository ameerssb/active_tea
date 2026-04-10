import os
import json
import numpy as np
import argparse
from collections import defaultdict
import hashlib


# -------------------------
# Utility: Normalize
# -------------------------
def norm_dict(_dict):
    if len(_dict) == 0:
        return _dict
    v_max = max(_dict.values())
    for k, v in _dict.items():
        _dict[k] = v / (v_max + 1e-6)
    _dict['max'] = v_max
    return _dict


# -------------------------
# Load inference results
# -------------------------
def PreprocessData(file_name):
    print('Loading File {}...'.format(file_name))
    with open(file_name, 'r') as f:
        data = json.load(f)

    difficult_indicators = defaultdict(float)
    information_indicators = defaultdict(float)
    diversity_indicators = defaultdict(float)

    for image_path, boxes_info in data.items():
        _difficult = 0
        _information = 0
        _cls_set = set()

        for box in boxes_info:
            _information += box['confidence score']
            _difficult = box['entropy']
            _cls_set.add(box['pred class'])

        _diversity = len(_cls_set)

        difficult_indicators[image_path] = _difficult
        information_indicators[image_path] = _information
        diversity_indicators[image_path] = _diversity

    return data, difficult_indicators, information_indicators, diversity_indicators


# -------------------------
# Count-Min Sketch
# -------------------------
class CountMinSketch:
    def __init__(self, width=2000, depth=5):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width))

    def _hash(self, key, i):
        return int(hashlib.md5((str(key)+str(i)).encode()).hexdigest(), 16) % self.width

    def update(self, key, value=1):
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.table[i][idx] += value

    def query(self, key):
        return min(self.table[i][self._hash(key, i)] for i in range(self.depth))


# -------------------------
# 🔥 YOUR FULL ALGORITHM
# -------------------------
def LearnedSketchSelection(data,
                           difficult_indicators,
                           information_indicators,
                           diversity_indicators,
                           file_name):

    print("Running LearnedSketchSelection...")

    # -------- PARAMETERS --------
    B = 1000       # total memory
    Br = 200       # heavy bucket capacity
    threshold = 0.6  # entropy threshold

    unique_buckets = {}   # exact storage (heavy hitters)
    sketch = CountMinSketch(width=B - Br, depth=5)

    final_scores = {}

    # Normalize indicators (optional but useful)
    difficult_indicators = norm_dict(difficult_indicators)
    information_indicators = norm_dict(information_indicators)
    diversity_indicators = norm_dict(diversity_indicators)

    # -------- STREAM PROCESS --------
    for image_path, boxes_info in data.items():

        # "frequency" = number of uncertain detections
        freq = 0
        for box in boxes_info:
            if box['entropy'] > threshold:
                freq += 1

        # -------- Learned Oracle --------
        if freq > 0:   # HH(i) = 1

            if image_path in unique_buckets:
                unique_buckets[image_path] += freq

            else:
                if len(unique_buckets) < Br:
                    unique_buckets[image_path] = freq
                else:
                    # Replace smallest bucket (optional)
                    min_key = min(unique_buckets, key=unique_buckets.get)
                    if unique_buckets[min_key] < freq:
                        del unique_buckets[min_key]
                        unique_buckets[image_path] = freq

        else:
            # -------- SketchAlg --------
            sketch.update(image_path, 1)

    # -------- MERGE RESULTS --------
    for image_path in data.keys():
        if image_path in unique_buckets:
            final_scores[image_path] = 2.0 + unique_buckets[image_path]
        else:
            final_scores[image_path] = sketch.query(image_path)

    # -------- SAVE OUTPUT --------
    with open(file_name + '.txt', 'w') as f:
        for image_path, score in final_scores.items():
            f.write(str(score) + '\n')

    with open(file_name + '.json', 'w') as f:
        json.dump(final_scores, f)

    print('Finish {}'.format(file_name))


# -------------------------
# Main wrapper
# -------------------------
def special_(args):

    data, difficult_indicators, information_indicators, diversity_indicators = PreprocessData(file_name=args.static_file)

    LearnedSketchSelection(
        data,
        difficult_indicators,
        information_indicators,
        diversity_indicators,
        file_name=args.indicator_file
    )


# -------------------------
# Entry point
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score function')

    parser.add_argument(
        "--static-file",
        type=str,
        default='temp/coco/static_by_random10.json'
    )

    parser.add_argument(
        "--indicator-file",
        type=str,
        default='results/coco/custom_learned_sketch'
    )

    args = parser.parse_args()

    special_(args)