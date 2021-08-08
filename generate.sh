flores101_dataset=../flores101_dataset
fairseq=fairseq

generation="fairseq/generation"
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
            --outputs=generation/spm.${SRC_NAME}-${TGT_NAME}.${SRC_NAME}

        python scripts/spm_encode.py \
            --model ../flores101_mm100_175M/sentencepiece.bpe.model \
            --output_format=piece \
            --inputs=$flores101_dataset/devtest/${TGT}.devtest \
            --outputs=generation/spm.${SRC_NAME}-${TGT_NAME}.${TGT_NAME}

        fairseq-preprocess \
            --source-lang ${SRC_NAME} --target-lang ${TGT_NAME} \
            --testpref generation/spm.${SRC_NAME}-${TGT_NAME} \
            --thresholdsrc 0 --thresholdtgt 0 \
            --destdir generation/data_bin_${SRC_NAME}_${TGT_NAME} \
            --srcdict ../flores101_mm100_175M/dict.txt --tgtdict ../flores101_mm100_175M/dict.txt
    done
done

n=20
for i in $(seq 1 $n)
do
    counter=0
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

            generation="generation/$i/generation_${SRC_NAME}_${TGT_NAME}"
            mkdir -p "$generation"

            fairseq-generate \
                generation/data_bin_${SRC_NAME}_${TGT_NAME} \
                --batch-size 1 \
                --path ../checkpoints/flores101_mm100_175M/"$i"/checkpoint_best.pt \
                --fixed-dictionary ../flores101_mm100_175M/dict.txt \
                -s ${SRC_NAME} -t ${TGT_NAME} \
                --remove-bpe 'sentencepiece' \
                --beam 5 \
                --task translation_multi_simple_epoch \
                --lang-pairs ../flores101_mm100_175M/language_pairs.txt \
                --decoder-langtok --encoder-langtok src \
                --gen-subset test \
                --fp16 \
                --dataset-impl mmap \
                --distributed-world-size 1 --distributed-no-spawn \
                --results-path $generation

            # clean fairseq generated file to only create hypotheses file.
            cat $generation/generate-test.txt  | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $generation/sys.txt

            result=$(sacrebleu ${flores101_dataset}/devtest/${TGT}.devtest < $generation/sys.txt --tokenize spm -b)
            total=$(echo "$total + $result" | bc -l)
            printf "%s,%s,%s\n" ${SRC_NAME} ${TGT_NAME} ${result} >> ../checkpoints/flores101_mm100_175M/"$i"/sacrebleu.txt
            ((counter++))
        done
    done

    echo Counter: $counter
    average=$(echo "$total / $counter" | bc -l)
    printf "%s,%0.2f,%0.2f\n" ${i} ${total} ${average}>> ../sacrebleu.txt
done
