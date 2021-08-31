flores101_dataset=../dataset/flores101_orig/flores101_dataset
fairseq=fairseq

generation="dataset/generation"
mkdir -p "$generation"

total=0
cd $fairseq

set -- eng ind jav msa tam tgl
for SRC; do
    shift
    for TGT; do
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

        python scripts/spm_encode.py \
            --model ../flores101_mm100_175M/sentencepiece.bpe.model \
            --output_format=piece \
            --inputs=$flores101_dataset/devtest/${SRC}.devtest \
            --outputs=../$generation/spm.${SRC_NAME}-${TGT_NAME}.${SRC_NAME}

        python scripts/spm_encode.py \
            --model ../flores101_mm100_175M/sentencepiece.bpe.model \
            --output_format=piece \
            --inputs=$flores101_dataset/devtest/${TGT}.devtest \
            --outputs=../$generation/spm.${SRC_NAME}-${TGT_NAME}.${TGT_NAME}

        fairseq-preprocess \
            --source-lang ${SRC_NAME} --target-lang ${TGT_NAME} \
            --testpref ../$generation/spm.${SRC_NAME}-${TGT_NAME} \
            --thresholdsrc 0 --thresholdtgt 0 \
            --destdir ../$generation/data_bin_${SRC_NAME}_${TGT_NAME} \
            --srcdict ../flores101_mm100_175M/dict.txt --tgtdict ../flores101_mm100_175M/dict.txt
    done
done

path=$1
set -- eng ind jav msa tam tgl
for SRC; do
    shift
    for TGT; do
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

        generation_i="../$path/generation_${SRC_NAME}_${TGT_NAME}"
        mkdir -p "$generation_i"


        FILE=$generation_i/generate-test.txt
        if [[ -f "$FILE" ]]; then
            echo "$FILE exists, skipping generation"
        else
            fairseq-generate \
            ../$generation/data_bin_${SRC_NAME}_${TGT_NAME} \
            --batch-size 1 \
            --path ../$path/checkpoint_best.pt \
            --fixed-dictionary ../flores101_mm100_175M/dict.txt \
            -s ${SRC_NAME} -t ${TGT_NAME} \
            --remove-bpe 'sentencepiece' \
            --beam 5 \
            --task translation_multi_simple_epoch \
            --lang-pairs 'en-id,id-en,en-jv,jv-en,en-ms,ms-en,en-ta,ta-en,en-tl,tl-en,id-jv,jv-id,id-ms,ms-id,id-ta,ta-id,id-tl,tl-id,jv-ms,ms-jv,jv-ta,ta-jv,jv-tl,tl-jv,ms-ta,ta-ms,ms-tl,tl-ms,ta-tl,tl-ta' \
            --decoder-langtok --encoder-langtok src \
            --gen-subset test \
            --fp16 \
            --dataset-impl mmap \
            --distributed-world-size 1 --distributed-no-spawn \
            --results-path $generation_i

            # clean fairseq generated file to only create hypotheses file.
            cat $generation_i/generate-test.txt  | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $generation_i/sys.txt

            result=$(sacrebleu ${flores101_dataset}/devtest/${TGT}.devtest < $generation_i/sys.txt --tokenize spm -b)
            printf "%s,%s,%s\n" ${SRC_NAME} ${TGT_NAME} ${result} >> ../"$path"/sacrebleu.csv
        fi

        generation_i="../$path/generation_${TGT_NAME}_${SRC_NAME}"
        mkdir -p "$generation_i"


        FILE=$generation_i/generate-test.txt
        if [[ -f "$FILE" ]]; then
            echo "$FILE exists, skipping generation"
        else
            fairseq-generate \
            ../$generation/data_bin_${SRC_NAME}_${TGT_NAME} \
            --batch-size 1 \
            --path ../$path/checkpoint_best.pt \
            --fixed-dictionary ../flores101_mm100_175M/dict.txt \
            -s ${TGT_NAME} -t ${SRC_NAME} \
            --remove-bpe 'sentencepiece' \
            --beam 5 \
            --task translation_multi_simple_epoch \
            --lang-pairs 'en-id,id-en,en-jv,jv-en,en-ms,ms-en,en-ta,ta-en,en-tl,tl-en,id-jv,jv-id,id-ms,ms-id,id-ta,ta-id,id-tl,tl-id,jv-ms,ms-jv,jv-ta,ta-jv,jv-tl,tl-jv,ms-ta,ta-ms,ms-tl,tl-ms,ta-tl,tl-ta' \
            --decoder-langtok --encoder-langtok src \
            --gen-subset test \
            --fp16 \
            --dataset-impl mmap \
            --distributed-world-size 1 --distributed-no-spawn \
            --results-path $generation_i

            # clean fairseq generated file to only create hypotheses file.
            cat $generation_i/generate-test.txt  | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $generation_i/sys.txt


            result=$(sacrebleu ${flores101_dataset}/devtest/${SRC}.devtest < $generation_i/sys.txt --tokenize spm -b)
            printf "%s,%s,%s\n" ${TGT_NAME} ${SRC_NAME} ${result} >> ../"$path"/sacrebleu.csv
        fi
    done
done
