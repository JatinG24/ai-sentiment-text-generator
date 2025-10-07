# 🧠 AI Sentiment Text Generator

This project is an **AI-powered text generator** that analyzes the **sentiment** of an input prompt (positive, negative, or neutral) and then produces a **paragraph aligned with that sentiment**.  
It combines **sentiment analysis** and **text generation** using pre-trained Transformer models from Hugging Face and includes an **interactive Streamlit frontend** for real-time input and output generation.

---

## 🎯 Objective

- Implement sentiment analysis on input prompts using Python and machine learning frameworks.
- Develop a text generation model that produces sentiment-aligned outputs.
- Build an interactive frontend for user input and text display.
- Document the methodology, dataset(s), and project challenges.

---

## 🧠 Methodology

### 1. Sentiment Detection
- The **sentiment analysis module** uses Hugging Face’s pre-trained model  
  **`distilbert-base-uncased-finetuned-sst-2-english`**.
- This model is fine-tuned on the **Stanford Sentiment Treebank (SST-2)** dataset, which classifies text as *positive* or *negative*.
- A post-processing function normalizes the result into **positive**, **negative**, or **neutral** categories.

### 2. Sentiment-Aligned Text Generation
- The **text generation module** uses **`gpt2-medium`**, a pre-trained language model.
- It adapts its output to match the detected or manually selected sentiment.
- The input prompt is modified with sentiment-based instructions:
  - **Positive:** “Write a positive, uplifting, and optimistic paragraph about…”
  - **Negative:** “Write a sad, emotional, or critical paragraph about…”
  - **Neutral:** “Write a neutral and factual paragraph about…”

### 3. User Interaction Layer (Frontend)
- The interface is built using **Streamlit**.
- Users can:
  - Enter a custom prompt.
  - Choose to **auto-detect sentiment** or **manually select one**.
  - Adjust **essay length** using a token slider (50–400 tokens).
  - View the generated result instantly.
  - **Download** the output text.

---

## 🧩 Project Structure

```
ai-sentiment-text-generator/
│
├── app.py                # Complete Streamlit application
├── requirements.txt      # Python dependencies
└── README.md             # Documentation (this file)
```

---

## ⚙️ Technical Approach

| Component | Framework / Model | Description |
|------------|------------------|--------------|
| **Frontend** | Streamlit | Provides user interface for input and output. |
| **Sentiment Analysis** | Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`) | Detects the sentiment of input text. |
| **Text Generation** | GPT-2 (`gpt2-medium`) | Generates sentiment-aligned paragraphs. |
| **Backend Framework** | PyTorch | Underlying engine for model execution. |

---

## 📚 Dataset(s) Used

| Dataset | Used By | Description |
|----------|----------|-------------|
| **SST-2 (Stanford Sentiment Treebank)** | DistilBERT | Provides labeled text samples for fine-tuning sentiment models. |
| **WebText (GPT-2 Pretraining Corpus)** | GPT-2 | Large-scale corpus used by OpenAI for natural text generation. |

✅ No additional training or dataset collection was required.  
All models are **pre-trained** and sourced from **Hugging Face Hub**.

---

## 🧰 Installation & Running Instructions

### 1. Clone or Download
```bash
git clone https://github.com/<your-username>/ai-sentiment-text-generator.git
cd ai-sentiment-text-generator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

### 4. Open in Browser
Navigate to `http://localhost:8501` (or as Streamlit indicates).

---

## 🧪 Example Run

**Input Prompt:**  
> “Technology and mental health”

**Detected Sentiment:**  
> positive

**Generated Output:**  
> “Technology has opened new ways for people to improve their mental well-being. From meditation apps to online therapy sessions, digital solutions are making emotional support more accessible and inclusive.”

---

## 💡 Reflections & Challenges

| Challenge | Solution / Reflection |
|------------|-----------------------|
| **Sentiment alignment drift:** GPT-2 tends to generate neutral text even when sentiment is strong. | Added **sentiment-steering prompts** (“Write a positive paragraph about...”). |
| **Model size & performance:** GPT-2-medium is memory-heavy. | Used dynamic device selection (CPU/GPU) and caching for efficiency. |
| **Latency:** Model loading caused slow startup. | Preloaded both models once on app initialization. |
| **Neutral sentiment handling:** Sometimes ambiguous in short prompts. | Normalized prediction logic and fallback to “neutral.” |
| **User control:** Needed flexibility in essay length and sentiment. | Added manual sentiment override and adjustable token slider. |

---

## 🌐 Deployment (Optional)

The app can be deployed freely using:
- **Streamlit Cloud:** [https://share.streamlit.io](https://share.streamlit.io)
- **Hugging Face Spaces (Streamlit App):** [https://huggingface.co/spaces](https://huggingface.co/spaces)

Steps:
1. Push project to GitHub.
2. Connect to Streamlit Cloud or Hugging Face Spaces.
3. Set `app.py` as the entry point and deploy.

---

## 🧭 Future Enhancements

- Multi-language sentiment and text generation support.
- User-selectable **tone** (formal, creative, persuasive).
- Option to generate **multi-paragraph essays**.
- Integration with **fine-tuned GPT models** for better sentiment alignment.

---

## 👨‍💻 Author

**Jatin Gangwani**  
📧 **jatin.gangwani2409@gmail.com**
💼 AI | NLP | Data Science Enthusiast  

---

## 🏁 License

This project is open-source under the **MIT License**.  
Feel free to modify, distribute, and enhance it for your use.

---
