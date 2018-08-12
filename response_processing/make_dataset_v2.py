#!/usr/bin/env python
import json
import os
import numpy as np
import argparse
from urllib.parse import urlparse

from madmom.custom_processors import LabelOutputProcessor
from madmom.features.tf_beats import TfRhythmicGroupingPreProcessor
from madmom.processors import _process, IOProcessor

from response_processing import util


def main():
    parser = argparse.ArgumentParser("merges a csv of survey responses, and a sqlite3 database of responses.")
    parser.add_argument("dumpfile", help="The output of \"flask dumpdb --outfile=dump.json\"")
    parser.add_argument("samples", help="folder with the actual mp3 samples")
    parser.add_argument('outfile', help='output file (EX: train_dataset.npz)')
    parser.add_argument('--fps', action='store', type=float, default=100, help='frames per second [default=100]')

    args = parser.parse_args()

    experiments = util.load_by_url(args.dumpfile)
    responses_by_url = util.get_final_responses_list(experiments)

    # trials = json.load(open(args.dumpfile, "r"))['dataset']

    n_features = 314
    n_frames = 800
    data = []
    labels = []
    sample_names = []
    for sample_url, final_responses in responses_by_url.items():

        o = urlparse(sample_url)
        sample_name = os.path.split(o.path)[-1]
        preprocessor = TfRhythmicGroupingPreProcessor()

        infile = os.path.join(args.samples, sample_name)

        print(infile)

        label_processor = LabelOutputProcessor(final_responses, args.fps)

        # create an IOProcessor
        processor = IOProcessor(preprocessor, label_processor)
        sample_data, sample_labels = _process((processor, infile, None, vars(args)))

        if sample_data.shape[0] < n_frames or sample_data.shape[1] != n_features:
            print("SKIPPING: Shapes is {} but it must be (>={}, {}), ".format(sample_data.shape, n_frames, n_features))
            continue

        sample_data = np.expand_dims(sample_data, axis=2)
        data.append(sample_data[:n_frames])
        labels.append(sample_labels[:n_frames])
        sample_names.append(sample_name)

    print("DATASET:")
    print('{} clips'.format(len(data)))
    print('{} frames of audio at {} fps'.format(n_frames, args.fps))
    print('{} features per frame'.format(n_features))
    print("saving {} ...".format(args.outfile))

    data = np.array(data)
    labels = np.array(labels)
    np.savez(args.outfile, x=data, labels=labels, sample_names=sample_names)

    print("done.")


if __name__ == '__main__':
    main()
