import os
import json
import numpy as np
import argparse
from collections import defaultdict
import hashlib


# -------------------------
# Count-Min Sketch
# -------------------------
class CountMinSketch:
    def __init__(self, width=2000, depth=5, decay=0.99):
        self.width = width
        self.depth = depth
        self.decay = decay
        self.table = np.zeros((depth, width))

    def _hash(self, key, i):
        return int(hashlib.md5((str(key) + str(i)).encode()).hexdigest(), 16) % self.width

    def apply_decay(self):
        self.table *= self.decay

    def update(self, key, value):
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.table[i][idx] += value

    def query(self, key):
        return min(self.table[i][self._hash(key, i)] for i in range(self.depth))


# -------------------------
# Load inference results
# -------------------------
def PreprocessData(file_name):
    print(f'Loading File {file_name}...')
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data


# -------------------------
# 🔥 LBFE WITH TEMPORAL DECAY
# -------------------------
def LBFE_Selection(data, file_name):

    print("Running LBFE + Temporal Decay...")

    # Parameters
    sketch = CountMinSketch(width=2000, depth=5, decay=0.99)

    final_scores = {}

    # -------- PASS 1: LEARN FREQUENCY WITH DECAY --------
    for image_path, boxes_info in data.items():

        # 🔑 Apply temporal decay before each new image
        sketch.apply_decay()

        for box in boxes_info:
            cls = box['pred class']
            conf = box['confidence score']
            entropy = box['entropy']

            # 🔑 Confidence-aware + uncertainty-aware update
            value = conf * entropy

            sketch.update(cls, value)

    # -------- PASS 2: SCORE IMAGES --------
    for image_path, boxes_info in data.items():

        score = 0

        for box in boxes_info:
            cls = box['pred class']
            entropy = box['entropy']

            # Query frequency
            freq = sketch.query(cls)

            # 🔑 Rare-class boosting
            weight = 1.0 / (np.log(freq + 1.0) + 1e-6)

            score += entropy * weight

        final_scores[image_path] = score

    # -------- SAVE OUTPUT --------
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name + '.txt', 'w') as f:
        for image_path in final_scores:
            f.write(str(final_scores[image_path]) + '\n')

    with open(file_name + '.json', 'w') as f:
        json.dump(final_scores, f)

    print(f'Finish {file_name}')


# -------------------------
# Wrapper
# -------------------------
def special_(args):
    data = PreprocessData(file_name=args.static_file)

    LBFE_Selection(
        data,
        file_name=args.indicator_file
    )


# -------------------------
# Entry point
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LBFE score function')

    parser.add_argument(
        "--static-file",
        type=str,
        default='temp/coco/static_by_random10.json'
    )

    parser.add_argument(
        "--indicator-file",
        type=str,
        default='results/coco/lbfe_scores'
    )

    args = parser.parse_args()

    special_(args)