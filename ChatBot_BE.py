from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pandas as pd
import numpy as np
import re, string
from collections import Counter
from underthesea import word_tokenize
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "http://127.0.0.1:8080"}})  # Cập nhật origin thành 8080

# Khởi tạo các biến global
model = None
encoder_model = None
decoder_model = None
tokenizer = None
word2idx = None
idx2word = None
maxlen_questions = None
maxlen_answers = None
embeddings_dim = 300
EMOTICONS = None
RAREWORDS = None

def initialize_chatbot():
    """Khởi tạo chatbot và load model"""
    global model, encoder_model, decoder_model, tokenizer, word2idx, idx2word
    global maxlen_questions, maxlen_answers, EMOTICONS, RAREWORDS
    
    try:
        # Đọc dữ liệu để tạo tokenizer
        df = pd.read_csv('./dataset.csv')
        if df.empty:
            raise ValueError("Tệp dataset.csv trống")
        
        # Xử lý giá trị NaN
        idx = df[df['user_b'].isnull()].index.tolist()
        df['user_b'] = df['user_b'].fillna('Luật sư')
        
        # Xử lý trước dữ liệu
        global EMOTICONS, RAREWORDS
        EMOTICONS = {
            u":-3": "Happy face smiley",
            u":3": "Happy face smiley",
            u":->": "Happy face smiley",
            u":>": "Happy face smiley",
            u":))": "Happy face smiley",
            u":)))": "Happy face smiley",
            u":))))": "Happy face smiley",
            u":'<": "Happy face smiley",
            u":)": "Happy face smiley",
            u":(": "Happy face smiley",
            u":((": "Happy face smiley",
            u":‑D": "Laughing, big grin or laugh with glasses",
            u":D": "Laughing, big grin or laugh with glasses",
            u"XD": "Laughing, big grin or laugh with glasses",
            u"=D": "Laughing, big grin or laugh with glasses",
            u":‑c": "Frown, sad, andry or pouting",
            u":c": "Frown, sad, andry or pouting",
            u":‑<": "Frown, sad, andry or pouting",
            u":<": "Frown, sad, andry or pouting",
            u":@": "Frown, sad, andry or pouting",
            u"D:": "Sadness",
            u":O": "Surprise",
            u":o": "Surprise",
        }

        cnt = Counter()
        for text in df["user_b"].values:
            for word in text.split():
                cnt[word] += 1

        RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-10-1:-1]])
        
        df = preprocessing(df)
        
        # Tách câu hỏi và câu trả lời
        data = df.values
        questions = data[:, 1]
        answers = data[:, 2]

        # Tokenization
        questions = [word_tokenize(ques) for ques in questions]
        answers = [word_tokenize(ans) for ans in answers]

        # Tạo tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(questions + answers)
        VOCAB_SIZE = len(tokenizer.word_index) + 1
        word2idx = tokenizer.word_index
        idx2word = {v: k for k, v in word2idx.items()}

        # Chuẩn bị maxlen
        tokenized_questions = tokenizer.texts_to_sequences(questions)
        maxlen_questions = max([len(x) for x in tokenized_questions])
        tokenized_answers = tokenizer.texts_to_sequences(answers)
        maxlen_answers = max([len(x) for x in tokenized_answers])

        # Load mô hình đã lưu
        model = tf.keras.models.load_model('./seq2seq_with_attention.keras')
        print("Mô hình đã được tải thành công")
        
        # Tạo encoder và decoder models
        setup_inference_models()
        
        return True
        
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy tệp - {e}")
        return False
    except Exception as e:
        print(f"Lỗi khởi tạo chatbot: {e}")
        return False

def remove_emoticons(text):
    arr = [word for word in text.split() if word not in EMOTICONS.keys()]
    return " ".join(arr)

def remove_rarewords(text):
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

def preprocessing(df):
    df["user_a"] = df["user_a"].apply(lambda ele: str(ele).translate(str.maketrans('', '', string.punctuation)))
    df["user_b"] = df["user_b"].apply(lambda ele: str(ele).translate(str.maketrans('', '', string.punctuation)))
    df["user_a"] = df["user_a"].apply(lambda ele: remove_emoticons(ele))
    df["user_b"] = df["user_b"].apply(lambda ele: remove_emoticons(ele))
    df["user_a"] = df["user_a"].apply(lambda ele: remove_rarewords(ele))
    df["user_b"] = df["user_b"].apply(lambda ele: remove_rarewords(ele))
    df['user_b'] = df['user_b'].apply(lambda ele: 'START ' + ele + ' END')
    df["user_a"] = df["user_a"].apply(lambda ele: ele.lower())
    df["user_b"] = df["user_b"].apply(lambda ele: ele.lower())
    return df

def setup_inference_models():
    """Thiết lập các model cho inference"""
    global encoder_model, decoder_model
    
    # Trích xuất các layer từ mô hình đã load
    encoder_inputs = model.input[0]
    decoder_inputs = model.input[1]
    encoder_embedding_layer = model.get_layer('encoder_embedding')
    encoder_lstm = model.get_layer('encoder_lstm')
    decoder_embedding_layer = model.get_layer('decoder_embedding')
    decoder_lstm = model.get_layer('decoder_lstm')
    attention = model.get_layer('attention')
    decoder_dense = model.get_layer('dense')

    # Encoder model
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding_layer(encoder_inputs))
    encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

    # Decoder model
    decoder_state_input_h = Input(shape=(embeddings_dim,), name='decoder_state_h')
    decoder_state_input_c = Input(shape=(embeddings_dim,), name='decoder_state_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_encoder_outputs = Input(shape=(maxlen_questions, embeddings_dim), name='decoder_encoder_outputs')

    decoder_emb = decoder_embedding_layer(decoder_inputs)
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_emb, initial_state=decoder_states_inputs)
    attention_output = attention([decoder_outputs, decoder_encoder_outputs])
    decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, attention_output])
    decoder_outputs = decoder_dense(decoder_combined_context)
    decoder_states = [state_h, state_c]

    decoder_model = Model([decoder_inputs, decoder_encoder_outputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)

def preprocess_new_question(question):
    """Tiền xử lý câu hỏi mới"""
    question = question.translate(str.maketrans('', '', string.punctuation)).lower()
    question = remove_emoticons(question)
    question = remove_rarewords(question)
    tokens = word_tokenize(question)
    sequence = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(sequence, maxlen=maxlen_questions, padding='post')
    return np.array(padded)

def decode_sequence(input_seq):
    """Dự đoán câu trả lời"""
    encoder_out, h, c = encoder_model.predict(input_seq, verbose=0)
    states_value = [h, c]
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx['start']
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, encoder_out] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = idx2word.get(sampled_token_index, '<UNK>')
        if sampled_word == 'end' or len(decoded_sentence) > maxlen_answers:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return ' '.join(decoded_sentence)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Tin nhắn không được để trống'})
        
        # Xử lý câu hỏi
        test_input = preprocess_new_question(user_message)
        bot_response = decode_sequence(test_input)
        
        return jsonify({
            'response': bot_response,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Có lỗi xảy ra: {str(e)}',
            'status': 'error'
        })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Đang khởi tạo chatbot...")
    if initialize_chatbot():
        print("Chatbot đã được khởi tạo thành công!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Lỗi khởi tạo chatbot. Vui lòng kiểm tra lại.")