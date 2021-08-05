* Clone all codes:

``` bash
git clone https://github.com/fatyanosa/WMT21
cd WMT21
```

* Create new environment:

``` bash
conda create --name Flores101 python=3.6
conda activate Flores101
```

* Requirements and installation:

``` bash
# pip install -r requirements.txt or using the following:
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c conda-forge pyarrow fastbpe sacremoses sacrebleu sentencepiece fairseq wandb
pip install subword-nmt
```

* Clone and install fairseq:

``` bash
git clone https://github.com/pytorch/fairseq
pip install git+https://github.com/pytorch/fairseq.git
```

* For faster training install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

## Train from scratch:

``` bash
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
```

## Fine tuning from pretrained model:
``` bash
bash finetune.sh
mkdir -p checkpoints/flores101_mm100_175M
CUDA_VISIBLE_DEVICES=1 fairseq-train dataset/data-bin/flores101.jv_id_ms_tl_ta_en.bpe --finetune-from-model flores101_mm100_175M/model.pt --save-dir checkpoints/flores101_mm100_175M --task translation_multi_simple_epoch --encoder-normalize-before --langs 'en,id,jv,ms,ta,tl' --lang-pairs 'en-id,id-en,en-jv,jv-en,en-ms,ms-en,en-ta,ta-en,en-tl,tl-en,id-jv,jv-id,id-ms,ms-id,id-ta,ta-id,id-tl,tl-id,jv-ms,ms-jv,jv-ta,ta-jv,jv-tl,tl-jv,ms-ta,ta-ms,ms-tl,tl-ms,ta-tl,tl-ta' --max-tokens 512 --decoder-normalize-before --sampling-method temperature --sampling-temperature 1.5 --encoder-langtok src --decoder-langtok --criterion label_smoothed_cross_entropy --label-smoothing 0.2 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 100000 --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 --update-freq 1 --seed 0 --log-format simple --log-interval 2 --patience 3 --arch transformer_wmt_en_de_big  --share-decoder-input-output-embed --share-all-embeddings --ddp-backend c10d  --no-epoch-checkpoints --num-workers 4 --batch-size 128 --empty-cache-freq 1 --combine-val --pipeline-encoder-balance 6 --pipeline-encoder-balance 6 --scoring sacrebleu --save-interval 1 --save-interval-updates 10000 --keep-interval-updates 1
```

## Fine tuning from pretrained model (10% dataset using random search):
``` bash
bash finetune-10%.sh
bash random_search.sh
```

* NOTE: If error "$'\r': command not found" run the following:
``` bash
sed -i 's/\r$//' filename
```
