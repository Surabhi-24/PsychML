import streamlit as st
import pickle
import re

# Page configuration
st.set_page_config(
    page_title="PsychML - Mood Detector",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .remedy-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Emotion remedies dictionary
REMEDIES = {
    "joy": [
        "🎉 Celebrate this moment! Share your happiness with loved ones.",
        "📝 Write down what made you happy - it'll help you recreate these moments.",
        "💪 Use this positive energy to tackle a challenging task.",
        "🎵 Listen to uplifting music and dance!",
        "🌟 Practice gratitude - acknowledge what's going well in your life."
    ],
    "sadness": [
        "🤗 Reach out to a trusted friend or family member for support.",
        "🚶‍♀️ Take a gentle walk outside - nature can be healing.",
        "📖 Journal your feelings to process your emotions.",
        "🎬 Watch something that makes you smile (comedy, cute animals).",
        "💆‍♀️ Practice self-care: take a warm bath, listen to soothing music.",
        "🧘‍♀️ Try meditation or deep breathing exercises."
    ],
    "anger": [
        "🧘 Take 10 deep breaths - breathe in for 4, hold for 4, exhale for 4.",
        "🏃‍♂️ Physical exercise can help release tension (run, boxing, yoga).",
        "📝 Write down your feelings in a private journal.",
        "🎵 Listen to calming music or sounds of nature.",
        "💬 Talk to someone you trust about what's bothering you.",
        "⏸️ Step away from the situation temporarily if possible."
    ],
    "fear": [
        "🧘‍♂️ Practice grounding techniques: name 5 things you see, 4 you touch, 3 you hear.",
        "💭 Challenge your fears: are they realistic? What's the worst that could happen?",
        "🤝 Talk to someone you trust about your concerns.",
        "📚 Research what you're afraid of - knowledge reduces fear.",
        "🎯 Break down the scary situation into smaller, manageable steps.",
        "😮‍💨 Use box breathing: inhale 4, hold 4, exhale 4, hold 4."
    ],
    "love": [
        "💌 Express your feelings to the person you care about.",
        "🎁 Do something thoughtful for someone you love.",
        "📸 Capture and cherish these positive emotions.",
        "🌸 Spread the love - perform random acts of kindness.",
        "💭 Reflect on what you appreciate about this relationship.",
        "✍️ Write a heartfelt letter or message."
    ],
    "surprise": [
        "⏸️ Take a moment to process what just happened.",
        "📝 Write down your thoughts and feelings about this surprise.",
        "🤔 Reflect on how this changes things (if it's a big surprise).",
        "💬 Share your experience with someone close to you.",
        "🎯 If it's a positive surprise, think about how to make the most of it.",
        "😌 Practice acceptance if it's unexpected news."
    ]
}

# Emotion colors for visual feedback
EMOTION_COLORS = {
    "joy": "#FFD700",
    "sadness": "#4682B4",
    "anger": "#DC143C",
    "fear": "#9370DB",
    "love": "#FF69B4",
    "surprise": "#FF8C00"
}

# Emotion emojis
EMOTION_EMOJIS = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😰",
    "love": "❤️",
    "surprise": "😲"
}

@st.cache_resource
def load_models():
    """Load the trained models and vectorizer"""
    try:
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('models/logistic_regression.pkl', 'rb') as f:
            model_lr = pickle.load(f)
        
        with open('models/svm.pkl', 'rb') as f:
            model_svm = pickle.load(f)
        
        return vectorizer, model_lr, model_svm
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.error("Please make sure you have:")
        st.error("- models/tfidf_vectorizer.pkl")
        st.error("- models/logistic_regression.pkl")
        st.error("- models/svm.pkl")
        st.info("Run 'python MLmodel.py' first to train and save the models!")
        return None, None, None

def clean_text(text):
    """Clean user input text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_emotion(text, vectorizer, model):
    """Predict emotion from text"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Transform using TF-IDF
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    
    # Get probability scores
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = max(probabilities) * 100
    elif hasattr(model, 'decision_function'):
        # For SVM
        decision_scores = model.decision_function(text_tfidf)[0]
        # Normalize scores to get confidence
        exp_scores = [2.71828 ** score for score in decision_scores]
        total = sum(exp_scores)
        confidence = (max(exp_scores) / total) * 100
    else:
        confidence = 85.0  # Default confidence
    
    return prediction, confidence

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 PsychML - Mood Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand your emotions and get personalized remedies</p>', unsafe_allow_html=True)
    
    # Load models
    vectorizer, model_lr, model_svm = load_models()
    
    if vectorizer is None or model_lr is None or model_svm is None:
        st.stop()
    
    # Sidebar for model selection
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        model_choice = st.radio(
            "Choose ML Model:",
            ("Logistic Regression", "Support Vector Machine (SVM)"),
            help="Select which machine learning model to use for emotion detection"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        if model_choice == "Logistic Regression":
            st.info("**Logistic Regression**: Fast and accurate for text classification. Great for probability estimates.")
        else:
            st.info("**SVM**: Powerful algorithm that finds optimal decision boundaries. Excellent for complex patterns.")
        
        st.markdown("---")
        st.markdown("### 🎯 Emotions Detected")
        st.markdown("- 😊 Joy\n- 😢 Sadness\n- 😠 Anger\n- 😰 Fear\n- ❤️ Love\n- 😲 Surprise")
    
    # Select the model based on user choice
    selected_model = model_lr if model_choice == "Logistic Regression" else model_svm
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 💬 How are you feeling today?")
        user_input = st.text_area(
            "",
            placeholder="Type your thoughts here... (e.g., 'I'm feeling really happy today because I got good news!')",
            height=150,
            key="mood_input"
        )
        
        analyze_button = st.button("🔍 Analyze My Mood", type="primary", use_container_width=True)
        
        # Show which model is being used
        st.caption(f"Using: **{model_choice}**")
    
    with col2:
        if analyze_button and user_input.strip():
            with st.spinner("Analyzing your mood..."):
                # Predict emotion
                emotion, confidence = predict_emotion(user_input, vectorizer, selected_model)
                
                # Display results
                st.markdown("### 🎯 Analysis Results")
                
                # Emotion with color and emoji
                emoji = EMOTION_EMOJIS.get(emotion, "😊")
                color = EMOTION_COLORS.get(emotion, "#1f77b4")
                
                st.markdown(f"""
                    <div class="emotion-box" style="background-color: {color}20; border: 2px solid {color};">
                        <h2 style="color: {color}; margin: 0;">
                            {emoji} {emotion.upper()}
                        </h2>
                        <p style="font-size: 1.5rem; margin: 10px 0;">
                            Confidence: <strong>{confidence:.1f}%</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Progress bar for confidence
                st.progress(confidence / 100)
                
                # Remedies section
                st.markdown("---")
                st.markdown("### 💡 Suggested Remedies")
                
                remedies = REMEDIES.get(emotion, ["Take a deep breath and relax."])
                
                for remedy in remedies:
                    st.markdown(f"""
                        <div class="remedy-box">
                            {remedy}
                        </div>
                    """, unsafe_allow_html=True)
                
        elif analyze_button and not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze your mood.")
    
    # Footer with information
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p><strong>About PsychML</strong></p>
            <p>This app uses Machine Learning (TF-IDF + Classification) to detect emotions from text.</p>
            <p>Compare results between Logistic Regression and SVM models using the sidebar!</p>
            <p><em>Note: This is for educational purposes. For serious mental health concerns, please consult a professional.</em></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()