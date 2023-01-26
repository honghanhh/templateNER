import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
import torch  
torch.manual_seed(3407)
import random
random.seed(3407)
import numpy as np
np.random.seed(3407)

import pandas as pd
from seq2seq_model import Seq2SeqModel

import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")
import timeit

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-train", "--train", type=str, help='./processed_data/template1/en-train.csv')
    parser.add_argument("-test", "--test", type=str, help='./processed_data/template1/en-dev.csv')
    parser.add_argument("-lang", "--lang", type=str, help='en')
    parser.add_argument("-output_dir", "--output_dir",type=str, help="./exp/template1")
    
    args = parser.parse_args()
        
    start = timeit.default_timer()

    print("Loading dataset...")
    # language_dict = {'en':177955,
    #                  'bn': 93792, 
    #                  'de':100905,
    #                  'es':189814, 
    #                  'fa':215856,
    #                  'fr':174251,
    #                  'hi':117342,
    #                  'it':167045,
    #                  'pt':186538,
    #                  'sv':156386,
    #                  'uk':163088,
    #                  'zh':145752
    #             }
    # if args.lang == 'en':
    #     df = pd.read_csv(args.data)
    #     train_df = df.iloc[:language_dict[args.lang]]
    #     eval_df = df.iloc[language_dict[args.lang]:].reset_index(drop=True)
    
    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.test)

    print("Loading model...")
    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 70,
        "train_batch_size": 32,
        "num_train_epochs": 5,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "evaluate_during_training": True,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "use_multiprocessing": False,
        "max_length": 35,
        "manual_seed": 4,
        "save_steps": 11898,
        "gradient_accumulation_steps": 1,
        "output_dir": args.output_dir
    }

    # Initialize model
    model = Seq2SeqModel(
        encoder_decoder_type= "bart", #"blender-large"
        # encoder_decoder_name= "facebook/blenderbot-400M-distill",
        # encoder_decoder_name= "t5-small",
        # encoder_decoder_name="facebook/mbart-large-50",
        encoder_decoder_name="facebook/bart-large",
        args=model_args,
        # use_cuda=False,
    )
    print("Training model...")
    # Train the model
    model.train_model(train_df, eval_data=eval_df)

    results = model.eval_model(eval_df, 
                               verbose = False, 
                               silent = True)
    print(results)

    print("Finishing model...")

    stop = timeit.default_timer()

    print('Time: ', stop - start) 
