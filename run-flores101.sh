# Binarize the dataset
TEXT=examples/flores101/dataset/flores101.jv_id_ms_tl_ta_en.bpe16k
DATA=examples/flores101/dataset/data-bin/flores101.jv_id_ms_tl_ta_en.bpe16k
mkdir -p "$DATA"

set -- en id jv ms ta tl
for SRC; do
    shift
    for TGT; do
        SRC_DICT="$(find $DATA/ -name "dict.${SRC}.txt" -type f)"
        TGT_DICT="$(find $DATA/ -name "dict.${TGT}.txt" -type f)"

        #fairseq-preprocess --source-lang $SRC --target-lang $TGT --trainpref $TEXT/train.bpe.${SRC}-${TGT} --validpref $TEXT/valid0.bpe.${SRC}-${TGT},$TEXT/valid1.bpe.${SRC}-${TGT} $( [[ -f "$SRC_DICT" ]] && printf %s "--srcdict $SRC_DICT" )  $( [[ -f "$TGT_DICT" ]] && printf %s "--tgtdict $TGT_DICT" ) --destdir $DATA --workers 16

        fairseq-preprocess --source-lang $SRC --target-lang $TGT --trainpref $TEXT/train.bpe.${SRC}-${TGT} --validpref $TEXT/valid0.bpe.${SRC}-${TGT},$TEXT/valid1.bpe.${SRC}-${TGT} --srcdict examples/flores101/fairseq.vocab --tgtdict examples/flores101/fairseq.vocab --destdir $DATA --workers 16

    done
done
