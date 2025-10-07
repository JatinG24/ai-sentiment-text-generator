import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# SENTIMENT CLASSIFIER
# -----------------------------
class SentimentClassifier:
    def __init__(self, model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name_or_path,
            device=self.device
        )

    def predict(self, text: str) -> str:
        result = self.pipe(text)
        label = result[0]["label"].lower()
        if "pos" in label:
            return "positive"
        elif "neg" in label:
            return "negative"
        elif "neutral" in label or "mixed" in label:
            return "neutral"
        return "neutral"


# -----------------------------
# TEXT GENERATOR
# -----------------------------
class TextGenerator:
    def __init__(self, model_name="gpt2-medium"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.torch_device = torch.device("cuda") if self.device == 0 else torch.device("cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.torch_device)

        # Create generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def generate(self, prompt: str, sentiment: str = "positive", max_length: int = 150) -> str:
        # Adjust prompt by sentiment
        if sentiment.lower() == "positive":
            full_prompt = f"Write a positive, uplifting, and optimistic paragraph about {prompt}. Focus on happiness, growth, and gratitude."
        elif sentiment.lower() == "negative":
            full_prompt = f"Write a critical, sad, or emotional paragraph about {prompt}. Focus on frustration, pain, or challenges."
        else:
            full_prompt = f"Write a neutral and factual paragraph about {prompt}. Focus on balance and clarity."

        # Generate text
        result = self.generator(
            full_prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )

        text = result[0]["generated_text"]
        return text.replace("\n", " ").strip()


# -----------------------------
# STREAMLIT APP
# -----------------------------
st.set_page_config(page_title="AI Sentiment Text Generator", layout="centered")
st.title("🧠 AI Sentiment Text Generator")

# Load models
@st.cache_resource
def load_models():
    return SentimentClassifier(), TextGenerator()

sentiment_model, text_gen = load_models()

# -----------------------------
# USER INPUT
# -----------------------------
prompt = st.text_area("✍️ Enter your prompt:", height=150)

col1, col2 = st.columns([1, 1])

with col1:
    manual_override = st.checkbox("🧭 Manually select sentiment")
    sentiment_choice = None
    if manual_override:
        sentiment_choice = st.selectbox("Choose sentiment:", ["positive", "neutral", "negative"])

with col2:
    max_tokens = st.slider(
        "📏 Essay length (tokens)",
        min_value=50,
        max_value=400,
        value=150,
        step=10
    )

# -----------------------------
# GENERATE BUTTON
# -----------------------------
if st.button("🚀 Generate Text"):
    if not prompt.strip():
        st.warning("Please enter a prompt to generate text.")
    else:
        if manual_override:
            sentiment = sentiment_choice
            st.write(f"**Manual sentiment selected:** {sentiment.capitalize()}")
        else:
            with st.spinner("Detecting sentiment..."):
                sentiment = sentiment_model.predict(prompt)
            st.write(f"**Detected sentiment:** {sentiment.capitalize()}")

        with st.spinner("Generating text..."):
            generated_text = text_gen.generate(prompt, sentiment=sentiment, max_length=max_tokens)

        st.subheader("📝 Generated Text:")
        st.write(generated_text)

        # Optional: allow download
        st.download_button(
            label="💾 Download Text",
            data=generated_text,
            file_name="generated_text.txt",
            mime="text/plain"
        )
