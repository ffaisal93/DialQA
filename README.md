# DialQa

Requirements ([`requirements.txt`]) and installation: 
```
./install.sh
```

### Data File Structure
```
data/
	dialqa-train.json
	dialqa-dev-og.json
	dialqa-dev-aug.json
	audio/
		dev/
			lang/
				dialect-region/
					{lang}-{id}-{dialect-region}.wav
```
- [`dialqa-dev-og.json`]: Original Development dataset gold questions.
- [`dialqa-dev-aug.json`]: Development dataset with dialectal questions (speech recognition i.e. ASR outputs). This is our **task development dataset**.
- [`lang`]: eng, ara, swa
- [`audio`]: folder containing question audio files. The audio file names [`{lang}-{id}-{dialect-region}`] have one-to-one mappings with the example ids from the json files.

## Baseline (ASR QA)

The task is to perform Extractive-QA using dialectal questions (Speech to text Outputs). We use google translation regional units (eg. en-US, sw-TZ) to perform speech to text conversion. The training file is based on huggingface's [`run_squad.py`] file.


#### Training baseline:

``` 

source vdial/bin/activate


python src/run_squad.py \
	--model_type bert \
	--model_name_or_path=bert-base-multilingual-uncased \
	--do_train \
	--do_lower_case \
	--train_file 'data/dialqa-train.json' \
	--per_gpu_train_batch_size 16 \
	--per_gpu_eval_batch_size 24 \
	--learning_rate 3e-5 \
	--num_train_epochs 3 \
	--max_seq_length 384 \
	--doc_stride 128 \
	--output_dir 'train_cache_output/' \
	--overwrite_cache \
	--overwrite_output_dir
```

#### Prediction on augemnted dev data

```
python src/run_squad.py \
	--model_type bert \
	--model_name_or_path='train_cache_output' \
	--do_eval \
	--do_lower_case \
	--predict_file 'data/dialqa-dev-aug.json' \
	--per_gpu_train_batch_size 16 \
	--per_gpu_eval_batch_size 16 \
	--learning_rate 3e-5 \
	--num_train_epochs 3 \
	--max_seq_length 384 \
	--doc_stride 128 \
	--output_dir 'outputs/aug-mbert' \
	--overwrite_output_dir
``` 

#### Baseline result	

| Language-Dialect | F1    | Exact Match | Example Count |
|------------------|-------|-------------|---------------|
| english-nga      | 73.36 | 58.70       | 494           |
| english-usa      | 74.35 | 59.31       | 494           |
| english-ind_s    | 72.22 | 58.10       | 494           |
| english-aus      | 73.67 | 59.52       | 494           |
| english-phl      | 73.76 | 59.11       | 494           |
|                  |       |             |               |
| arabic-dza       | 71.72 | 56.17       | 324           |
| arabic-egy       | 72.39 | 56.79       | 324           |
| arabic-jor       | 73.27 | 57.41       | 324           |
| arabic-tun       | 73.55 | 57.71       | 324           |
|                  |       |             |               |
| swahili-kenya    | 72.12 | 63.1        | 1000          |
| swahili-tanzania | 70.74 | 61.7        | 1000          |
|                  |       |             |               |
| Total            | 72.60 | 59.71       | 5766          |

## Citation
If you use SD-QA, please cite the "[SD-QA: Spoken Dialectal Question Answering for the Real World](https://arxiv.org/abs/2109.12072)". You can use the following BibTeX entry
~~~
@inproceedings{faisal-etal-21-sdqa,
 title = {{SD-QA}: {S}poken {D}ialectal {Q}uestion {A}nswering for the {R}eal {W}orld},
  author = {Faisal, Fahim and Keshava, Sharlina and ibn Alam, Md Mahfuz and Anastasopoulos, Antonios},
  url={https://arxiv.org/abs/2109.12072},
  year = {2021},
  booktitle = {Findings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP Findings)},
  publisher = {Association for Computational Linguistics},
  month = {November},
}
~~~

We built our augmented dataset and baselines on top of TydiQA. Kindly also make sure to cite the original TyDi QA paper,
~~~
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
journal = {TACL},
year    = {2020}
}
~~~

## License
Both the code and data for SD-QA are availalbe under the [Apache License 2.0](LICENSE).