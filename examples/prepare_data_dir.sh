#!/bin/bash

# Generate 
#    
#    wav.scp 
#    rttm
#    segments
#    spk2utt
#    utt2spk
#    reco2dur

SCRIPTS_DIR=<directory with data scripts, see below>

cd $SCRIPTS_DIR

PATH_WAV=$1
PATH_RTTM=$2
LIST=$3
OUT_DATA_DIR=$4



if [ -d $OUT_DATA_DIR ]
then
    echo "$OUT_DATA_DIR already exists"
    exit -1
fi

mkdir $OUT_DATA_DIR

# wav.scp
while IFS= read -r line; do
	echo "$PATH_WAV/$line.wav" >> $OUT_DATA_DIR/tmp_wav.list
done < $LIST
awk -F'/' '{print $NF}' $OUT_DATA_DIR/tmp_wav.list | sed 's|\.wav||g' > $OUT_DATA_DIR/tmp_wav_ids
paste -d' ' $OUT_DATA_DIR/tmp_wav_ids $OUT_DATA_DIR/tmp_wav.list > $OUT_DATA_DIR/wav.scp

# rttm
while IFS= read -r line; do
	cat $PATH_RTTM/$line.rttm >> $OUT_DATA_DIR/rttm
done < $LIST

# segments: create from rttm
awk '{printf "%s_%s_%07d_%07d %s %.2f %.2f\n", \
      $8, $2, $4*100, ($4+$5)*100, $2, $4, $4+$5}' \
      $OUT_DATA_DIR/rttm | sort > $OUT_DATA_DIR/segments

# utt2spk
# The line below needs to be updated depending on the dataset
awk '{split($1,A,"_"); printf "%s %s_%s_%s\n", $1, A[1], A[2], A[3]}' \
    $OUT_DATA_DIR/segments > $OUT_DATA_DIR/utt2spk

# spk2utt
$SCRIPTS_DIR/utils/utt2spk_to_spk2utt.pl $OUT_DATA_DIR/utt2spk > $OUT_DATA_DIR/spk2utt

# reco2dur
$SCRIPTS_DIR/utils/data/get_reco2dur.sh $OUT_DATA_DIR

rm $OUT_DATA_DIR/tmp_wav.list $OUT_DATA_DIR/tmp_wav_ids

$SCRIPTS_DIR/utils/fix_data_dir.sh $OUT_DATA_DIR
