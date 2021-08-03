git clone https://github.com/fatyanosa/WMT21

cd WMT21

pip install -r requirements.txt

Follow Installation in here: https://github.com/pytorch/fairseq

cd ../..

## Train from scratch:

bash prepare-flores101.sh

mkdir -p checkpoints/multilingual_transformer

CUDA_VISIBLE_DEVICES=0 fairseq-train dataset/data-bin/flores101.jv_id_ms_tl_ta_en.bpe16k/ \
--max-epoch 1 \
--ddp-backend=legacy_ddp \
--task multilingual_translation \
--lang-pairs 'en-id,id-en,en-jv,jv-en,en-ms,ms-en,en-ta,ta-en,en-tl,tl-en,id-jv,jv-id,id-ms,ms-id,id-ta,ta-id,id-tl,tl-id,jv-ms,ms-jv,jv-ta,ta-jv,jv-tl,tl-jv,ms-ta,ta-ms,ms-tl,tl-ms,ta-tl,tl-ta' \
--arch multilingual_transformer \
--share-decoders --share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--lr 0.0005 --lr-scheduler inverse_sqrt \
--warmup-updates 4000 --warmup-init-lr '1e-07' \
--label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
--dropout 0.3 --weight-decay 0.0001 \
--save-dir checkpoints/multilingual_transformer \
--max-tokens 4000 \
--update-freq 2 --combine-val


## Fine tuning from pretrained model:

bash finetune.sh

mkdir -p checkpoints/flores101_mm100_615M

CUDA_VISIBLE_DEVICES=1 fairseq-train dataset/data-bin/flores101.jv_id_ms_tl_ta_en.bpe16k/ \
--finetune-from-model flores101_mm100_615M/model.pt \
--save-dir checkpoints/flores101_mm100_615M \
--task translation_multi_simple_epoch \
--encoder-normalize-before --langs 'en,id,jv,ms,ta,tl' \
--lang-pairs 'en-id,id-en,en-jv,jv-en,en-ms,ms-en,en-ta,ta-en,en-tl,tl-en,id-jv,jv-id,id-ms,ms-id,id-ta,ta-id,id-tl,tl-id,jv-ms,ms-jv,jv-ta,ta-jv,jv-tl,tl-jv,ms-ta,ta-ms,ms-tl,tl-ms,ta-tl,tl-ta' \
--max-tokens 1200 --decoder-normalize-before --sampling-method temperature \
--sampling-temperature 1.5 --encoder-langtok src --decoder-langtok \
--criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 \
--max-update 40000 --dropout 0.3 --attention-dropout 0.1 \
--weight-decay 0.0 --update-freq 2 --save-interval 1 \
--save-interval-updates 5000 --keep-interval-updates 10 \
--seed 222 --log-format simple --log-interval 2 --patience 10 \
--arch transformer_wmt_en_de_big --encoder-layers 12 \
--decoder-layers 12 --encoder-layerdrop 0.05 \
--decoder-layerdrop 0.05 --share-decoder-input-output-embed \
--share-all-embeddings --ddp-backend no_c10d \
--no-last-checkpoints --keep-best-checkpoints 1 --combine-val




NOTE: If error "$'\r': command not found" run the following:

sed -i 's/\r$//' filename
