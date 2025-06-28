from .utils import load_model, load_tokenizer
import logging
import torch

logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.max_input_length = 500  # Max words for prompt

    def _load_model(self):
        if self.model is None:
            self.model = load_model("gpt2", self.model_name)
            self.tokenizer = load_tokenizer(self.model_name)
            # Set pad token to avoid EOS token conflict
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Model {self.model_name} loaded with pad_token_id: {self.tokenizer.pad_token_id}")

    def generate(self, prompt, max_length=500, num_beams=5, temperature=0.7):
        self._load_model()
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Truncate prompt if too long
        prompt_words = prompt.split()
        if len(prompt_words) > self.max_input_length:
            prompt = ' '.join(prompt_words[:self.max_input_length])
            logger.warning(f"Prompt truncated to {self.max_input_length} words")
        
        try:
            # Get model device
            device = next(self.model.parameters()).device
            logger.info(f"Using device: {device} for generation")
            
            # Tokenize input with padding and attention mask
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
                return_attention_mask=True
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Generate text
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=temperature,
                do_sample=True,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Clean up output
            cleaned_text = ' '.join(generated_text.split())
            logger.info(f"Generated text for prompt: {prompt[:50]}...")
            return cleaned_text
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise ValueError(f"Text generation failed: {str(e)}")