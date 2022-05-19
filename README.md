# DialQa

Requirements (`requirements.txt`) and installation: 
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
			{lang}/
				{dialect-region}/
					{lang}-{id}-{dialect-region}.wav
```
- `dialqa-dev-og.json`: Original Development dataset gold questions.
- `dialqa-dev-aug.json`: Development dataset with dialectal questions (speech recognition i.e. ASR outputs). This is our **task development dataset**.
- `lang`: eng, ara, swa
- `audio`: folder containing question audio files. The audio file names `{lang}-{id}-{dialect-region}` have one-to-one mappings with the example ids from the json files.

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
| english-phl      | 73.76 | 59.11       | 494           |
| **english(avg)** | 73.47 | 58.95       | 2470          |
|                  |       |             |               |
| arabic-dza       | 71.72 | 56.17       | 324           |
| arabic-egy       | 72.39 | 56.79       | 324           |
| arabic-jor       | 73.27 | 57.41       | 324           |
| arabic-tun       | 73.55 | 57.71       | 324           |
| **arabic(avg)**  | 72.73 | 57.02       | 1296          |
|                  |       |             |               |
| swahili-kenya    | 72.12 | 63.1        | 1000          |
| swahili-tanzania | 70.74 | 61.7        | 1000          |
|                  |       |             |               |
| Total            | 72.60 | 59.71       | 5766          |

## Citation
Audio files and augmented dataset are from SD-QA which was built on top of TyDiQA.
~~~
@inproceedings{faisal-etal-2021-sd-qa,
    title = "{SD}-{QA}: Spoken Dialectal Question Answering for the Real World",
    author = "Faisal, Fahim  and
      Keshava, Sharlina  and
      Alam, Md Mahfuz Ibn  and
      Anastasopoulos, Antonios",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.281",
    doi = "10.18653/v1/2021.findings-emnlp.281",
    pages = "3296--3315",
}
~~~

~~~
@article{clark-etal-2020-tydi,
    title = "{T}y{D}i {QA}: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages",
    author = "Clark, Jonathan H.  and
      Choi, Eunsol  and
      Collins, Michael  and
      Garrette, Dan  and
      Kwiatkowski, Tom  and
      Nikolaev, Vitaly  and
      Palomaki, Jennimaria",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "8",
    year = "2020",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2020.tacl-1.30",
    doi = "10.1162/tacl_a_00317",
    pages = "454--470",
}
~~~

## License
The data is availalbe under the [Apache License 2.0](LICENSE).
