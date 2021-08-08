for i in {1..20}
do
    batch_size=$((8 + $RANDOM % 128))
    learning_rate="$(awk -v seed=$RANDOM -v min=3e-5 -v max=3e-4 'BEGIN{srand(seed); print min+rand()*(max-min)}')"
    beta1="$(awk -v seed=$RANDOM -v min=0.7 -v max=0.9999 'BEGIN{srand(seed); print min+rand()*(max-min)}')"
    beta2="$(awk -v seed=$RANDOM -v min=0.7 -v max=0.9999 'BEGIN{srand(seed); print min+rand()*(max-min)}')"
    epsilon="$(awk -v seed=$RANDOM -v min=9.98e-09 -v max=9.99e-06 'BEGIN{srand(seed); print min+rand()*(max-min)}')"
    weight_decay="$(awk -v seed=$RANDOM -v min=8.3e-7 -v max=0.018 'BEGIN{srand(seed); print min+rand()*(max-min)}')"
    attention_dropout="$(awk -v seed=$RANDOM -v min=0 -v max=0.5 'BEGIN{srand(seed); print min+rand()*(max-min)}')"
    dropout="$(awk -v seed=$RANDOM -v min=0 -v max=0.5 'BEGIN{srand(seed); print min+rand()*(max-min)}')"
    seed=$((0 + $RANDOM % 100))

    # Train a multilingual transformer model
    fairseq-train 'dataset_10%/data-bin/flores101.jv_id_ms_tl_ta_en.bpe' --finetune-from-model flores101_mm100_175M/model.pt --save-dir checkpoints/flores101_mm100_175M/"$i" --task translation_multi_simple_epoch --encoder-normalize-before --langs 'en,id,jv,ms,ta,tl' --lang-pairs 'en-id,id-en,en-jv,jv-en,en-ms,ms-en,en-ta,ta-en,en-tl,tl-en,id-jv,jv-id,id-ms,ms-id,id-ta,ta-id,id-tl,tl-id,jv-ms,ms-jv,jv-ta,ta-jv,jv-tl,tl-jv,ms-ta,ta-ms,ms-tl,tl-ms,ta-tl,tl-ta' --max-tokens 512 --decoder-normalize-before --sampling-method temperature --sampling-temperature 1.5 --encoder-langtok src --decoder-langtok --criterion label_smoothed_cross_entropy --label-smoothing 0.2 --optimizer adam --adam-eps ${epsilon} --adam-betas "(${beta1}, ${beta2})" --lr-scheduler inverse_sqrt --lr ${learning_rate} --warmup-updates 2500 --max-epoch 2 --dropout ${dropout} --attention-dropout ${attention_dropout} --weight-decay ${weight_decay} --update-freq 4 --seed ${seed} --log-format simple --log-interval 2 --arch transformer_wmt_en_de_big  --share-decoder-input-output-embed --share-all-embeddings --ddp-backend c10d  --no-epoch-checkpoints --num-workers 4 --batch-size ${batch_size} --empty-cache-freq 1 --combine-val --pipeline-encoder-balance 6 --pipeline-encoder-balance 6 --scoring sacrebleu --save-interval 1 --save-interval-updates 1000 --keep-interval-updates 1

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" ${i} ${epsilon} ${beta1} ${beta2} ${learning_rate} ${dropout} ${attention_dropout} ${weight_decay} ${seed} ${batch_size} >> random_search.txt

done
