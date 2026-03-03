import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Betting Categories Mapping ---
def map_bet_category(s):
    is_small = s <= 13
    is_big = s >= 14
    is_even = (s % 2 == 0)
    is_odd = not is_even
    categories = []
    if is_small:
        categories.append('S')
    if is_big:
        categories.append('B')
    if is_even:
        categories.append('E')
    if is_odd:
        categories.append('O')
    if is_small and is_even:
        categories.append('SE')
    if is_big and is_even:
        categories.append('BE')
    if is_small and is_odd:
        categories.append('SO')
    if is_big and is_odd:
        categories.append('BO')
    return categories

# --- Prepare DataFrame with Categories ---
def prepare_data(df):
    df['categories'] = df['sum'].apply(map_bet_category)
    return df

# --- Probability Distribution of sums ---
def calc_probability_distribution(df):
    total = len(df)
    counts = df['sum'].value_counts().sort_index()
    probs = counts / total
    return probs

# --- Prepare LSTM Data ---
def create_lstm_data(series, time_steps=10):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i:i+time_steps])
        y.append(series[i+time_steps])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(28, activation='softmax')  # sums 0-27
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Prepare classification data ---
def create_features(df):
    df['sum_lag1'] = df['sum'].shift(1)
    df['sum_lag2'] = df['sum'].shift(2)
    return df.dropna()

def prepare_classification_data(df):
    X = df[['sum_lag1', 'sum_lag2']]
    category_map = {'S':0, 'B':1, 'E':2, 'O':3, 'SE':4, 'BE':5, 'SO':6, 'BO':7}
    df['label'] = df['categories'].apply(lambda cats: category_map[cats[0]])
    y = df['label']
    return X, y

# --- Simple Reinforcement Learning Bettor (placeholder) ---
class SimpleRLBettor:
    def __init__(self, categories):
        self.categories = categories
        self.q_values = {cat: 0 for cat in categories}  # q-values initial zero
    def suggest_bet(self):
        return max(self.q_values, key=self.q_values.get)
    def update(self, chosen_cat, reward):
        alpha = 0.1
        self.q_values[chosen_cat] += alpha * (reward - self.q_values[chosen_cat])

# --- Prediction function ---
@st.cache(allow_output_mutation=True)
def train_models(historical_sums):
    df = pd.DataFrame({'sum': historical_sums})
    df = prepare_data(df)

    # Probability distribution
    prob_dist = calc_probability_distribution(df)

    # LSTM preparation and training
    series = df['sum'].values
    X_lstm, y_lstm = create_lstm_data(series)
    X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
    lstm_model = build_lstm_model((X_lstm.shape[1], 1))
    lstm_model.fit(X_lstm, y_lstm, epochs=5, batch_size=64, verbose=0)

    # Classification data preparation
    df_cls = create_features(df)
    X_cls, y_cls = prepare_classification_data(df_cls)
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    # Return all trained elements and data needed for prediction
    return lstm_model, clf, acc, series, prob_dist

def predict_next_number(lstm_model, clf, acc, series, prob_dist):
    categories = ['S', 'B', 'E', 'O', 'SE', 'BE', 'SO', 'BO']
    bettor = SimpleRLBettor(categories)

    # LSTM prediction
    last_sequence = series[-10:]
    input_seq = last_sequence.reshape((1, 10, 1))
    pred_probs = lstm_model.predict(input_seq, verbose=0)[0]
    pred_sum = np.argmax(pred_probs)

    # Classification prediction of category
    last_two = np.array(series[-2:]).reshape(1, -1)
    pred_cat_index = clf.predict(last_two)[0]
    pred_category = categories[pred_cat_index]

    suggested_bet = bettor.suggest_bet()

    return {
        "predicted_sum": pred_sum,
        "probability_of_sum": pred_probs[pred_sum],
        "predicted_category": pred_category,
        "classification_accuracy": acc,
        "rl_suggested_bet": suggested_bet,
        "probability_distribution": prob_dist.to_dict()
    }


# --- Streamlit app UI ---
st.title("Lucky 28 AI Prediction App")

st.markdown("""
Enter historical Lucky 28 sums (comma separated).
Example: `12, 17, 9, 20, 15, 13, 14, 18, 7, 10, 16, 25, 11, 19, 21`
""")

user_input = st.text_area("Enter historical sums", height=150)

if st.button("Predict Next Number"):
    try:
        sums_str = [x.strip() for x in user_input.split(",")]
        sums = [int(x) for x in sums_str if x.isdigit()]
        if len(sums) < 15:
            st.error("Please enter at least 15 historical sums for meaningful prediction.")
        else:
            with st.spinner("Training models and predicting..."):
                lstm_model, clf, acc, series, prob_dist = train_models(sums)
                prediction = predict_next_number(lstm_model, clf, acc, series, prob_dist)

            st.success(f"Predicted Next Sum: {prediction['predicted_sum']} (Probability: {prediction['probability_of_sum']:.3f})")
            st.info(f"Predicted Betting Category: {prediction['predicted_category']} (Accuracy of category prediction: {prediction['classification_accuracy']:.2f})")
            st.info(f"RL Suggested Bet Category: {prediction['rl_suggested_bet']}")

            st.subheader("Probability Distribution of Past Sums")
            prob_df = pd.DataFrame(sorted(prediction['probability_distribution'].items()), columns=["Sum", "Probability"])
            st.bar_chart(prob_df.set_index("Sum"))

    except Exception as e:
        st.error(f"Error processing input: {e}")


st.markdown("""
---
### Deployment on Render.com Free Tier
1. Save this script as `app.py`.
2. Create a `requirements.txt` file with:
