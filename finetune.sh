#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/fairseq/scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

ORIG=$ROOT/dataset/flores101_orig
DATA=$ROOT/dataset/flores101.jv_id_ms_tl_ta_en.bpe
mkdir -p "$ORIG" "$DATA"

TRAIN_FOLDER=$ORIG/small_task2_filt
VALID_FOLDER=$ORIG/flores101_dataset

TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

URLS=(
    "data.statmt.org/wmt21/multilingual-task/small_task2_filt_v2.tar.gz"
    "dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz"
)
ARCHIVES=(
    "small_task2_filt_v2.tar.gz"
    "flores101_dataset.tar.gz"
)

# download and extract data
for ((i=0;i<${#URLS[@]};++i)); do
    ARCHIVE=$ORIG/${ARCHIVES[i]}
    if [ -f "$ARCHIVE" ]; then
        echo "$ARCHIVE already exists, skipping download"
    else
        URL=${URLS[i]}
        wget -P "$ORIG" "$URL"
        if [ -f "$ARCHIVE" ]; then
            echo "$URL successfully downloaded."
        else
            echo "$URL not successfully downloaded."
            exit 1
        fi
    fi
    FILE=${ARCHIVE: -4}
    if [ -e "$FILE" ]; then
        echo "$FILE already exists, skipping extraction"
    else
        tar -C "$ORIG" -xzvf "$ARCHIVE"
    fi
done

echo "pre-processing train data..."
set -- en id jv ms ta tl
for SRC; do
    shift
    for TGT; do
        for LANG in "${SRC}" "${TGT}"; do
            FILES="$(find $TRAIN_FOLDER/ -name "*${SRC}-${TGT}.${LANG}*" -type f)"
            echo "$FILES"

            for FILE in $FILES; do
                cat "$FILE" \
                    > "$DATA/train.${SRC}-${TGT}.${LANG}"
            done
        done
    done
done

echo "pre-processing valid data..."
set -- eng ind jav msa tam tgl
for SRC; do
    shift
    for TGT; do
        echo $SRC
        echo $TGT
        SRC_FILES="$(find $VALID_FOLDER/ -name "*${SRC}*" -type f)"
        TGT_FILES="$(find $VALID_FOLDER/ -name "*${TGT}*" -type f)"
        echo $SRC_FILES
        echo $TGT_FILES

        if [[ "$SRC" == "eng" ]]; then
            SRC_NAME=en
        elif [[ "$SRC" == "ind" ]]; then
            SRC_NAME=id
        elif [[ "$SRC" == "jav" ]]; then
            SRC_NAME=jv
        elif [[ "$SRC" == "msa" ]]; then
            SRC_NAME=ms
        elif [[ "$SRC" == "tam" ]]; then
            SRC_NAME=ta
        elif [[ "$SRC" == "tgl" ]]; then
            SRC_NAME=tl
        fi

        if [[ "$TGT" == "eng" ]]; then
            TGT_NAME=en
        elif [[ "$TGT" == "ind" ]]; then
            TGT_NAME=id
        elif [[ "$TGT" == "jav" ]]; then
            TGT_NAME=jv
        elif [[ "$TGT" == "msa" ]]; then
            TGT_NAME=ms
        elif [[ "$TGT" == "tam" ]]; then
            TGT_NAME=ta
        elif [[ "$TGT" == "tgl" ]]; then
            TGT_NAME=tl
        fi

        i=0
        for SRC_FILE in $SRC_FILES; do
            cat "${SRC_FILE}" > "$DATA/valid${i}.${SRC_NAME}-${TGT_NAME}.${SRC_NAME}"
            ((i=i+1))
        done

        i=0
        for TGT_FILE in $TGT_FILES; do
            cat "${TGT_FILE}" > "$DATA/valid${i}.${SRC_NAME}-${TGT_NAME}.${TGT_NAME}"
            ((i=i+1))
        done
    done
done

# Download 175M param model.
wget https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz

#Extract
tar -xvzf flores101_mm100_175M.tar.gz

# encode train/valid
echo "encoding train with learned BPE..."
set -- en id jv ms ta tl
for SRC; do
    shift
    for TGT; do
        python "$SPM_ENCODE" \
            --model flores101_mm100_175M/sentencepiece.bpe.model \
            --output_format=piece \
            --inputs $DATA/train.${SRC}-${TGT}.${SRC} $DATA/train.${SRC}-${TGT}.${TGT} \
            --outputs $DATA/train.bpe.${SRC}-${TGT}.${SRC} $DATA/train.bpe.${SRC}-${TGT}.${TGT}
    done
done

echo "encoding valid with learned BPE..."
set -- en id jv ms ta tl
for SRC; do
    shift
    for TGT; do
        echo $SRC
        echo $TGT
        for i in 0 1
        do
             python "$SPM_ENCODE" \
                --model flores101_mm100_175M/sentencepiece.bpe.model \
                --output_format=piece \
                --inputs $DATA/valid${i}.${SRC}-${TGT}.${SRC} $DATA/valid${i}.${SRC}-${TGT}.${TGT} \
                --outputs $DATA/valid${i}.bpe.${SRC}-${TGT}.${SRC} $DATA/valid${i}.bpe.${SRC}-${TGT}.${TGT}
        done
    done
done

echo "binarize the dataset..."

TEXT=$ROOT/dataset/flores101.jv_id_ms_tl_ta_en.bpe
DATA="$ROOT/dataset/data-bin/flores101.jv_id_ms_tl_ta_en.bpe"
mkdir -p "$DATA"

set -- en id jv ms ta tl
for SRC; do
    shift
    for TGT; do
        fairseq-preprocess --source-lang $SRC --target-lang $TGT --trainpref $TEXT/train.bpe.${SRC}-${TGT} --validpref $TEXT/valid0.bpe.${SRC}-${TGT},$TEXT/valid1.bpe.${SRC}-${TGT} --srcdict $ROOT/flores101_mm100_175M/dict.txt --tgtdict $ROOT/flores101_mm100_175M/dict.txt --destdir $DATA --workers 16

    done
done
