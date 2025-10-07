import warnings
warnings.filterwarnings("ignore")
from transformers import pipeline
import torch


def generate_essay(prompt, max_length=800):
    """
    Generates an essay based purely on the given prompt using GPT-Neo.

    Args:
        prompt (str): The starting text or essay topic.
        max_length (int): The maximum number of tokens to generate.

    Returns:
        str: The generated essay text.
    """
    try:
        # Detect device (GPU if available)
        device = 0 if torch.cuda.is_available() else -1

        # Load GPT-Neo model (high quality, medium size)
        generator = pipeline(
            "text-generation",
            model="EleutherAI/gpt-neo-1.3B",
            device=device
        )

        # Generate essay text
        result = generator(
            prompt,
            max_length=max_length,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=generator.tokenizer.eos_token_id
        )

        essay_text = result[0]["generated_text"]

        # Clean prompt repetition (GPT-Neo sometimes repeats input)
        if essay_text.lower().startswith(prompt.lower()):
            essay_text = essay_text[len(prompt):].strip()

        # Format output nicely into paragraphs
        essay_text = essay_text.replace("\n", " ").strip()
        essay_text = ". ".join([s.strip().capitalize() for s in essay_text.split(". ") if s.strip()])

        return essay_text

    except Exception as e:
        return f"An error occurred: {e}"


# Example usage:
if __name__ == "__main__":
    essay_topic = "The impact of climate change on global agriculture"
    print(f"\nGenerating essay on: {essay_topic}\n")
    essay = generate_essay(essay_topic)
    print(essay)
