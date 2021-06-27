git clone https://github.com/fatyanosa/WMT21

cd WMT21

pip install -r requirements.txt

Follow Installation in here: https://github.com/pytorch/fairseq

cd ../..

bash prepare-flores101.sh

mkdir -p checkpoints/multilingual_transformer

Train from scratch:

CUDA_VISIBLE_DEVICES=0 fairseq-train dataset/data-bin/flores101.jv_id_ms_tl_ta_en.bpe16k/ \
--max-epoch 1 \
--ddp-backend=legacy_ddp \
--task multilingual_translation --lang-pairs en-id,en-jv,en-ms,en-ta,en-tl,id-jv,id-ms,id-ta,id-tl,jv-ms,jv-ta,jv-tl,ms-ta,ms-tl,ta-tl \
--arch multilingual_transformer \
--share-decoders --share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--lr 0.0005 --lr-scheduler inverse_sqrt \
--warmup-updates 4000 --warmup-init-lr '1e-07' \
--label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
--dropout 0.3 --weight-decay 0.0001 \
--save-dir checkpoints/multilingual_transformer \
--max-tokens 4000 \
--update-freq 2 \
--combine-val