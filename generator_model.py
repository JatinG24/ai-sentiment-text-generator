import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class TextGenerator:
    def __init__(self, model_name="gpt2-medium"):
        # Detect device
        self.device = 0 if torch.cuda.is_available() else -1
        self.torch_device = torch.device("cuda") if self.device == 0 else torch.device("cpu")

        # Load tokenizer and model fully on a real device (avoid meta tensors)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None)
        self.model.to(self.torch_device)  # safe: model now has real weights

        # Create a pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def generate(self, prompt: str, sentiment: str = "positive") -> str:
        # Adjust the prompt based on sentiment
        if sentiment.lower() == "positive":
            full_prompt = f"Write a short, optimistic, and positive paragraph about {prompt}. Focus on happiness, growth, and gratitude."
        elif sentiment.lower() == "negative":
            full_prompt = f"Write a long, negative, and emotional paragraph about {prompt}. Focus on sadness, anger, frustration, and loneliness."
        else:
            full_prompt = f"Write a neutral paragraph about {prompt}. Focus on facts and clarity."

        # Generate text
        result = self.generator(
            full_prompt,
            max_length=150,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )

        text = result[0]["generated_text"]
        return text.replace("\n", " ").strip()
