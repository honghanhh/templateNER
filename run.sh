# Preprocessing
python preprocess.py --input_dir './public_data/DE-German/' --output_dir './preprocessed_data/' --lang 'de'

# Training
python train.py --train './preprocessed_data/de-train.csv' --test './preprocessed_data/de-dev.csv' --output_dir './bart_de' --model 'bart'

# Inference
python inference.py --data_path './public_data/DE-German/de_test.conll' --word_max_length 4 --model 'mbart' --model_path './best_model/' --output_path './de.pred.conll'
