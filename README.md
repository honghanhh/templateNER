# L3i++ at Semeval 2023-Task 2: CoNER

## Introduction

This repository contains the source code for the L3i++ team at Semeval 2023-Task 2: CoNER.

## Datasets

We use the dataset from the SemEval 2023 Task 2: [MultiCoNER II
Multilingual Complex Named Entity Recognition](https://multiconer.github.io/), which is available at [here](https://codalab.lisn.upsaclay.fr/competitions/10025). This dataset contains 12 languages (English, Spanish, Swedish, Ukrainian, Portuguese, French, Farsi, German, Chinese, Hindi, Bangla, and Italian), divided into 3 parts: train, dev, and test. Each part contains a set of CoNLL files, which are the input data for the model. The CoNLL files are in the following format:

```conll
# id 0d88e010-c6e8-4409-9dec-a785e43eac16	domain=de
sie _ _ O
war _ _ O
die _ _ O
erste _ _ O
frau _ _ O
die _ _ O
beim _ _ O
großes _ _ B-Facility
auge _ _ I-Facility
beobachtet _ _ O
durfte _ _ O
. _ _ O
```

See the sample files in the [`public_data/DE-German/`](./public_data/DE-German/) folder.

## Requirements

Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To preprocess the data, run the following command:

```bash
python ./models/preprocess.py --input_dir './public_data/DE-German/' --output_dir './preprocessed_data/' --lang 'de'
```

To train the model, run the following command:

```bash
python  ./models/train.py --train './preprocessed_data/de-train.csv' --test './preprocessed_data/de-dev.csv' --output_dir './bart_de' --model 'bart'
```

To inference the model and export the results, run the following command:

```bash
python  ./models/inference.py --data_path './public_data/DE-German/de_test.conll' --word_max_length 4 --model 'mbart' --model_path './best_model/' --output_path './de.pred.conll'
```

If you are lazy to run theses 3 above commands, you can run the following command to end-to-end reproduce the results:

```bash
chmod +x run.sh
./run.sh
```

You can also access the monolingual English trained model at [here](https://drive.google.com/drive/folders/18FUAM1oUZp9-jXqGzBBeNzhrXyWdFJTF?usp=share_link).
## Results


We will update the results after the leaderboard is released.

## Contributors

- 🐮 [TRAN Thi Hong Hanh](https://github.com/honghanhh) 🐮
