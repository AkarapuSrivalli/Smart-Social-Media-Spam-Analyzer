import pandas as pd
import nltk
import streamlit as st
import string
import matplotlib.pyplot as plt
import io

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from wordcloud import WordCloud

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Spam Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ---------------- HIDE STREAMLIT MENU ----------------
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stDeployButton"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DOWNLOAD STOPWORDS ----------------
nltk.download('stopwords')

# ---------------- LOAD DATASET ----------------
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['CONTENT', 'CLASS']]
df.columns = ['Comment', 'Class']

# ---------------- TEXT CLEANING ----------------
stop_words = stopwords.words('english')

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['Cleaned_Comment'] = df['Comment'].apply(clean_text)

# ---------------- SENTIMENT FUNCTION ----------------
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive 😊"
    elif polarity < 0:
        return "Negative 😞"
    else:
        return "Neutral 😐"

# ---------------- SPAM EXPLANATION FUNCTION ----------------
def explain_comment(comment):
    reasons = []
    spam_keywords = ["free","win","offer","click","subscribe","buy","discount","money","cash"]
    comment_lower = comment.lower()

    if "http" in comment_lower or "www" in comment_lower:
        reasons.append("Contains suspicious link")

    for word in spam_keywords:
        if word in comment_lower:
            reasons.append(f"Contains promotional keyword: {word}")

    if comment.isupper() and len(comment) > 5:
        reasons.append("Uses excessive capital letters")

    if len(comment.split()) < 2:
        reasons.append("Very short suspicious comment")

    if not reasons:
        reasons.append("No suspicious patterns detected")

    return reasons

# ---------------- FEATURES ----------------
X = df['Cleaned_Comment']
y = df['Class']

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# ---------------- MODEL COMPARISON ----------------
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

model = lr_model
accuracy = lr_acc

# ---------------- SESSION STATE ----------------
if "spam_total" not in st.session_state:
    st.session_state.spam_total = int(df['Class'].value_counts()[1])

if "genuine_total" not in st.session_state:
    st.session_state.genuine_total = int(df['Class'].value_counts()[0])

# ---------------- SIDEBAR ----------------
st.sidebar.title("🤖 AI Spam Analyzer")
st.sidebar.write("### Technologies Used")
st.sidebar.write("Python, NLP, TF-IDF, Logistic Regression, Streamlit")

st.sidebar.progress(int(accuracy * 100))
st.sidebar.write(f"{accuracy * 100:.2f}% Accurate")

# ---------------- MAIN UI ----------------
st.title("📱 Smart Social Media Spam Analyzer")

st.subheader("📊 Live Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Comments", len(df))
col2.metric("Spam Comments", st.session_state.spam_total)
col3.metric("Genuine Comments", st.session_state.genuine_total)

# ---------------- DASHBOARD PERCENTAGES ----------------
spam_percentage = (st.session_state.spam_total / (st.session_state.spam_total + st.session_state.genuine_total)) * 100
genuine_percentage = 100 - spam_percentage

col4, col5 = st.columns(2)
col4.metric("Spam %", f"{spam_percentage:.1f}%")
col5.metric("Genuine %", f"{genuine_percentage:.1f}%")

# ---------------- INPUT ----------------
st.subheader("💬 Comment Analyzer")
user_input = st.text_area("Enter Social Media Comment")

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Comment"):

    cleaned_input = clean_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])

    prediction = model.predict(input_vector)
    probability = model.predict_proba(input_vector)

    spam_prob = probability[0][1] * 100
    genuine_prob = probability[0][0] * 100

    if prediction[0] == 1:
        st.error("🚨 Spam Comment Detected")
        st.session_state.spam_total += 1
        result_value = "Spam"
        result_score = f"{spam_prob:.2f}%"
    else:
        st.success("✅ Genuine Comment")
        st.session_state.genuine_total += 1
        result_value = "Genuine"
        result_score = f"{genuine_prob:.2f}%"

    st.subheader("📊 Analysis Result")
    col6, col7 = st.columns(2)
    col6.metric("Prediction", result_value)
    col7.metric("Confidence", result_score)

    # ---------------- CONFIDENCE BAR ----------------
    confidence = max(spam_prob, genuine_prob)
    st.progress(int(confidence))
    if confidence >= 90:
        st.success("Very High Confidence")
    elif confidence >= 70:
        st.info("High Confidence")
    else:
        st.warning("Moderate Confidence")

    # ---------------- SENTIMENT ----------------
    sentiment = get_sentiment(user_input)
    st.info(f"Sentiment Analysis: {sentiment}")

    # ---------------- EXPLANATION ----------------
    st.subheader("🧠 Explanation")
    reasons = explain_comment(user_input)
    for reason in reasons:
        st.write("•", reason)

    # ---------------- DOWNLOAD REPORT ----------------
    report = pd.DataFrame({
        "Comment":[user_input],
        "Prediction":[result_value],
        "Confidence":[result_score],
        "Sentiment":[sentiment]
    })
    csv = report.to_csv(index=False)
    st.download_button("📥 Download Report", csv, "analysis_report.csv", "text/csv")

    # ---------------- GRAPH ----------------
    st.subheader("📊 Prediction Graph")
    labels = ["Spam", "Genuine"]
    values = [spam_prob, genuine_prob]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Probability %")
    ax.set_title("Prediction Analysis")
    st.pyplot(fig)

    # ---------------- LIVE STATS ----------------
    st.subheader("📉 Updated Statistics")
    col8, col9 = st.columns(2)
    col8.metric("Spam Comments", st.session_state.spam_total)
    col9.metric("Genuine Comments", st.session_state.genuine_total)

# ---------------- PERFORMANCE ----------------
st.subheader("📈 Model Performance")
st.progress(int(accuracy * 100))
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# ---------------- UPLOAD CSV ----------------
st.subheader("📂 Upload CSV File")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.write(uploaded_df.head())

# ---------------- DATA VIEW ----------------
if st.checkbox("📄 Show Dataset"):
    st.write(df.head())

# ---------------- WORD CLOUD ----------------
st.subheader("☁️ Spam Word Cloud")
spam_text = " ".join(df[df["Class"] == 1]["Cleaned_Comment"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(spam_text)
fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# ---------------- MODEL COMPARISON ----------------
st.subheader("🤖 Model Comparison")
comparison_df = pd.DataFrame({
    "Model":["Logistic Regression","Naive Bayes","Random Forest"],
    "Accuracy":[round(lr_acc*100,2), round(nb_acc*100,2), round(rf_acc*100,2)]
})
st.dataframe(comparison_df)

st.markdown("---")
st.markdown("Built using Python + NLP + Machine Learning + Streamlit")
