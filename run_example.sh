#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python $CDIR/diaper/infer_single_file.py -c $CDIR/examples/infer_16k_10attractors.yaml --wav-dir $CDIR/examples --wav-name IS1009a --models-path $CDIR/models/10attractors/SC_LibriSpeech_2spk_adapted1-10/models/ --rttms-dir $CDIR/examples
