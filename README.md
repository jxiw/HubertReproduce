# HubertReproduce

## Prepare pretrain data

### Step 1. Download libri-Ligh data

Download Libri-Light in https://github.com/facebookresearch/libri-light/tree/main/data_preparation

Libri-Light 600 hr: https://dl.fbaipublicfiles.com/librilight/data/small.tar

Libri-Light 6k hr: https://dl.fbaipublicfiles.com/librilight/data/small.tar + https://dl.fbaipublicfiles.com/librilight/data/medium.tar

Libri-Light 60k hr: https://dl.fbaipublicfiles.com/librilight/data/small.tar + https://dl.fbaipublicfiles.com/librilight/data/medium.tar + https://dl.fbaipublicfiles.com/librilight/data/large.tar

### Step 2. Generate train/valid split file using wave2vec script

```
cd ~/fairseq/examples/wav2vec
python wav2vec_manifest.py ~/fairseq/examples/hubert/libri6k/ --dest ~/fairseq/examples/hubert/libri6k/ --ext flac --valid-percent 0.01
```

### Step 3. Dump MFCC feature

Prepare training data

```
export tsv_dir=libri6k/
export feat_dir=libri6k_feat/
export split=train
export nshard=2000

for rank in $(seq 0 $((nshard-1))); do
	echo "simple_kmeans/dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}"
	python simple_kmeans/dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}
done
```

Prepare validation data 

```
export tsv_dir=libri6k/
export feat_dir=libri6k_feat/
export split=valid
export nshard=50

for rank in $(seq 0 $((nshard-1))); do
	echo "simple_kmeans/dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}"
	python simple_kmeans/dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}
done
```

Check data shape, accroding to HuBERT paper, MFCC features are 39-dimensional vectors.

```
import numpy as np
a=np.load("libri6k_feat/train_178_2000.npy")
assert a.shape[1]==39
```

### Step 4. Run K-means clustering

```
export feat_dir=libri6k_feat/
export split=train
export nshard=2000
export km_path=libri6k_km
export n_cluster=100

python learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 0.1
```

### Step 5. Apply a trained k-means model to obtain labels 

```
export feat_dir=libri6k_feat/
export split=train
export km_path=libri6k_km
export nshard=2000
export lab_dir=libri6k_lab/

for rank in $(seq 0 $((nshard-1))); do
	python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
done
```

### Step 6. Create a dummy dict

```
export lab_dir=libri6k_lab/
export n_cluster=100

for x in $(seq 0 $((n_clusters - 1))); do
  echo "$x 1"
done >> $lab_dir/dict.km.txt
```

## Pretrain

```
export data_path=libri6k_feat/
export label_path=libri6k_lab/
export config_dir=config/pretrain/

python fairseq_cli/hydra_train.py \
  --config-dir ${config_dir} \
  --config-name hubert_base_librispeech \
  task.data=${data_path} task.label_dir=${label_path} task.labels='["km"]' model.label_rate=100
```