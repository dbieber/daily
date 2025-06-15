import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from typing import List, Tuple
from steganographic_llm import SteganographicLLM


class HuggingFaceModelWrapper:
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize with a HuggingFace model.
        
        Args:
            model_name: Name of the HuggingFace model to use (default: distilgpt2)
        """
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully!")
    
    def get_top_tokens(self, prompt: str, prefix: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get top k tokens and their probabilities given a prompt and prefix.
        
        Args:
            prompt: The original prompt
            prefix: The text generated so far
            top_k: Number of top tokens to return
            
        Returns:
            List of (token_string, probability) tuples
        """
        # Combine prompt and prefix
        full_text = prompt + prefix
        
        # Tokenize
        inputs = self.tokenizer.encode(full_text, return_tensors="pt")
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits[0, -1, :]  # Get logits for last token
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get top k tokens
        top_k_probs, top_k_indices = torch.topk(probs, min(top_k, len(probs)))
        
        # Convert to strings and return
        result = []
        for prob, idx in zip(top_k_probs.tolist(), top_k_indices.tolist()):
            token_str = self.tokenizer.decode([idx])
            result.append((token_str, prob))
        
        return result


def create_huggingface_model_fn(model_name: str = "Qwen/Qwen2.5-3B"):
    """
    Create a model function compatible with SteganographicLLM.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Function that takes (prompt, prefix) and returns top 128 tokens with probabilities
    """
    wrapper = HuggingFaceModelWrapper(model_name)
    
    def model_fn(prompt: str, prefix: str) -> List[Tuple[str, float]]:
        return wrapper.get_top_tokens(prompt, prefix, top_k=20)
    
    return model_fn


if __name__ == "__main__":
    # Test with a real model
    print("Creating HuggingFace model function...")
    model_fn = create_huggingface_model_fn("Qwen/Qwen2.5-3B")
    
    print("Initializing steganographic system...")
    steg_llm = SteganographicLLM(model_fn)
    
    # Test parameters
    prompt = "The weather today is"
    hidden_text = "Hi!"
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Hidden text: '{hidden_text}'")
    print(f"Binary: {steg_llm.text_to_bits(hidden_text)}")
    
    # Test basic encoding
    print("\n--- Basic Encoding Test ---")
    try:
        encoded_basic = steg_llm.encode_basic(prompt, hidden_text)
        print(f"Generated tokens: {encoded_basic}")
        
        # Create the full generated text
        generated_text = ''.join(encoded_basic)
        print(f"Generated text: '{generated_text}'")
        print(f"Full text: '{prompt + generated_text}'")
        
        # Decode
        decoded_basic = steg_llm.decode(prompt, encoded_basic, 'basic')
        print(f"Decoded: '{decoded_basic}'")
        print(f"Match: {decoded_basic.strip() == hidden_text}")
        
    except Exception as e:
        print(f"Error in basic encoding: {e}")
    
    # Test Huffman encoding
    print("\n--- Huffman Encoding Test ---")
    try:
        encoded_huffman = steg_llm.encode_huffman(prompt, hidden_text)
        print(f"Generated tokens: {encoded_huffman}")
        
        # Create the full generated text
        generated_text = ''.join(encoded_huffman)
        print(f"Generated text: '{generated_text}'")
        print(f"Full text: '{prompt + generated_text}'")
        
        # Decode
        decoded_huffman = steg_llm.decode(prompt, encoded_huffman, 'huffman')
        print(f"Decoded: '{decoded_huffman}'")
        print(f"Match: {decoded_huffman.strip() == hidden_text}")
        
    except Exception as e:
        print(f"Error in Huffman encoding: {e}")