import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model

model = load_model('hamet_model.h5')

# Load Tokenizer
with open('tokenizer.pkl','rb') as obj:
    tokenizer = pickle.load(obj)

# Function to predict the next word

def predict_next_word(model, tokenizer, text, max_seq_len):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_seq_len:
    token_list = token_list[-(max_seq_len-1):]  #Ensure the sequence length matches
  token_list = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')
  predicted = model.predict(token_list, verbose=0)
  predicted_word_idx = np.argmax(predicted, axis=1)
  for word,index in tokenizer.word_index.items():
    if index == predicted_word_idx:
      return word
  return None

#Streamlit Web APP
st.write("Next word prediction using LSTM and Early Stopping")
input_text = st.text_input("Enter the sequence of words", "To be or not to")
if st.button("Predict Next Word"):
  max_seq_len = model.input_shape[1]+1
  next_word = predict_next_word(model,tokenizer,input_text,max_seq_len)
  st.write(f"Next Word is: {next_word}")