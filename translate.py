from datasets import load_dataset
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tải dataset
datasets = load_dataset("ILT37/translate_vi_en")

# Lấy 5000 mẫu từ train và 500 mẫu từ test
data_train = datasets['train'][:5000]
data_test = datasets['test'][:500]

# Tách câu nguồn (en) và đích (vi)
train_sentences = data_train['en']
train_labels = data_train['vi']

test_sentences = data_test['en']
test_labels = data_test['vi']

# Xác định siêu tham số
vocab_size = 10000
embedding_dim = 64
max_length = 140

# === Tokenizer cho tiếng Anh (nguồn) ===
src_tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
src_tokenizer.fit_on_texts(train_sentences)
src_sequences = src_tokenizer.texts_to_sequences(train_sentences)
encoder_input_data = pad_sequences(src_sequences, maxlen=max_length, padding='post', truncating='post')

# === Tokenizer cho tiếng Việt (đích) ===
# Thêm <start> và <end> token
tgt_texts_in = ["<start> " + txt for txt in train_labels]
tgt_texts_out = [txt + " <end>" for txt in train_labels]

tgt_tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tgt_tokenizer.fit_on_texts(tgt_texts_in + tgt_texts_out)

tgt_sequences_in = tgt_tokenizer.texts_to_sequences(tgt_texts_in)
tgt_sequences_out = tgt_tokenizer.texts_to_sequences(tgt_texts_out)

decoder_input_data = pad_sequences(tgt_sequences_in, maxlen=max_length, padding='post', truncating='post')
decoder_target_data = pad_sequences(tgt_sequences_out, maxlen=max_length, padding='post', truncating='post')

# Reshape để phù hợp với sparse_categorical_crossentropy
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# === Lặp lại các bước xử lý cho test data ===
test_sentences = data_test['en']
test_labels = data_test['vi']

test_sequences = src_tokenizer.texts_to_sequences(test_sentences)
encoder_input_data_test = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

tgt_texts_in_test = ["<start> " + txt for txt in test_labels]
tgt_texts_out_test = [txt + " <end>" for txt in test_labels]

tgt_sequences_in_test = tgt_tokenizer.texts_to_sequences(tgt_texts_in_test)
tgt_sequences_out_test = tgt_tokenizer.texts_to_sequences(tgt_texts_out_test)

decoder_input_data_test = pad_sequences(tgt_sequences_in_test, maxlen=max_length, padding='post', truncating='post')
decoder_target_data_test = pad_sequences(tgt_sequences_out_test, maxlen=max_length, padding='post', truncating='post')

decoder_target_data_test = np.expand_dims(decoder_target_data_test, -1)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# === Encoder ===
encoder_inputs = Input(shape=(None,), name="encoder_inputs")
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="encoder_embedding")(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(embedding_dim, return_state=True, name="encoder_lstm")(x)
encoder_states = [state_h, state_c]

# === Decoder ===
decoder_inputs = Input(shape=(None,), name="decoder_inputs")
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="decoder_embedding")(decoder_inputs)
decoder_lstm = LSTM(embedding_dim, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax', name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

# === Model ===
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Huấn luyện mô hình
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=64,
    epochs=20,
    validation_data=([encoder_input_data_test, decoder_input_data_test], decoder_target_data_test)
)
