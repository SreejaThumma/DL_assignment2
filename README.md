# DL_assignment2
# Character-Level Sequence-to-Sequence Transliteration
This project implements a character-level sequence-to-sequence model for transliterating text from Latin script to Devanagari script using the Dakshina dataset. The model is built using TensorFlow/Keras and follows the encoder-decoder architecture with LSTM layers.

# Table of Contents
Overview

Installation

Data Preprocessing

Model Architecture

Training

Evaluation

Prediction

# Overview
The task is to train a sequence-to-sequence (seq2seq) model for transliteration, which converts Latin characters into their corresponding Devanagari characters. The model leverages the Dakshina dataset, which contains parallel text pairs in Latin and Devanagari scripts. The model uses an encoder-decoder architecture with LSTM layers, and the input and target sequences are tokenized at the character level.

# Installation
To run this project, you need to have Python and the necessary dependencies installed.
# Install the required libraries:

bash
Copy
Edit
pip install -r requirements.txt
# Data Preprocessing
The dataset consists of parallel text pairs (Devanagari, Latin transliteration, frequency). The script performs the following steps:

Loads the dataset from the .tsv files.

Prepares the input and target sequences.

Adds special tokens (\t for start and \n for end) to the target sequences.

Creates token-to-index mappings for both input and target sequences.

Pads the input and output sequences to a fixed length for feeding into the model.

# Model Architecture
The model follows a standard sequence-to-sequence architecture with the following components:

Encoder:

Takes the input sequence and passes it through an embedding layer followed by an LSTM layer.

Outputs the encoder states (hidden and cell states).

Decoder:

Takes the previous token as input and generates the next token in the sequence using an embedding layer and an LSTM layer.

Outputs the predicted sequence via a Dense layer with a softmax activation.

Training
The model is trained using categorical cross-entropy loss and the Adam optimizer. The training process includes the following steps:

Encode the input and target sequences.

Train the model on the training data.

Validate the model using a separate validation set.

The model is evaluated on both the training and test sets after training.

# Example:
python
Copy
Edit
history = model.fit(
    [train_enc, train_dec_in], train_dec_out,
    validation_data=([dev_enc, dev_dec_in], dev_dec_out),
    batch_size=64,
    epochs=30
)
# Evaluation
After training, the model is evaluated using the evaluate() method to measure the accuracy on the training and test sets.

# Example:
python
Copy
Edit
train_loss, train_acc = model.evaluate([train_enc, train_dec_in], train_dec_out, verbose=0)
test_loss, test_acc = model.evaluate([test_enc, test_dec_in], test_dec_out, verbose=0)
Prediction
The trained model can be used to predict transliterations for new input sequences. The decoder model is used for inference, where the input sequence is passed through the encoder, and the decoder generates the transliterated sequence one token at a time.

# Example:
python
Copy
Edit
decoded_sentence = decode_sequence(input_seq)
