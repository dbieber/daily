import heapq
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Any
import math


class HuffmanNode:
    def __init__(self, char: Optional[int] = None, freq: float = 0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


class SteganographicLLM:
    def __init__(self, model_fn):
        """
        Initialize with a language model function.
        
        Args:
            model_fn: Function that takes (prompt, prefix) and returns list of (token, probability) 
                     tuples for the top 20 most likely next tokens
        """
        self.model_fn = model_fn
        self.top_k = 20
    
    def text_to_bits(self, text: str) -> str:
        """Convert text to binary string."""
        return ''.join(format(ord(char), '08b') for char in text)
    
    def bits_to_text(self, bits: str) -> str:
        """Convert binary string back to text."""
        chars = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte = bits[i:i+8]
                char_code = int(byte, 2)
                if char_code > 0:  # Skip null bytes from padding
                    chars.append(chr(char_code))
        return ''.join(chars)
    
    def build_huffman_tree(self, probabilities: List[float]) -> Tuple[HuffmanNode, Dict[int, str]]:
        """Build Huffman tree from token probabilities."""
        heap = []
        for i, prob in enumerate(probabilities):
            heapq.heappush(heap, HuffmanNode(char=i, freq=prob))
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        root = heap[0] if heap else None
        
        # Build encoding table
        codes = {}
        if root:
            if root.char is not None:  # Single node case
                codes[root.char] = '0'
            else:
                self._build_codes(root, '', codes)
        
        return root, codes
    
    def _build_codes(self, node: HuffmanNode, code: str, codes: Dict[int, str]):
        """Recursively build Huffman codes."""
        if node.char is not None:
            codes[node.char] = code
        else:
            if node.left:
                self._build_codes(node.left, code + '0', codes)
            if node.right:
                self._build_codes(node.right, code + '1', codes)
    
    def encode_basic(self, prompt: str, hidden_text: str) -> List[str]:
        """
        Encode hidden text using basic 5-bit selection from top 20 tokens.
        """
        bits = self.text_to_bits(hidden_text)
        generated_tokens = []
        current_text = prompt
        
        bit_idx = 0
        while bit_idx < len(bits):
            # Always use exactly 5 bits (pad the entire message if needed)
            if bit_idx + 5 > len(bits):
                # Pad the bits string to make it divisible by 5
                bits = bits.ljust(((len(bits) + 4) // 5) * 5, '0')
            
            # Extract exactly 5 bits
            token_bits = bits[bit_idx:bit_idx + 5]
            token_idx = int(token_bits, 2)
            
            # Get top 20 tokens from model
            token_probs = self.model_fn(prompt, current_text)
            if len(token_probs) < self.top_k:
                # If model returns fewer than 20 tokens, pad the index
                token_idx = token_idx % len(token_probs)
            else:
                token_idx = token_idx % self.top_k
            
            # Select the token
            selected_token = token_probs[token_idx][0]
            generated_tokens.append(selected_token)
            current_text += selected_token
            
            bit_idx += 5
        
        return generated_tokens
    
    def encode_huffman(self, prompt: str, hidden_text: str) -> List[str]:
        """
        Encode hidden text using Huffman encoding based on token probabilities.
        """
        bits = self.text_to_bits(hidden_text)
        generated_tokens = []
        current_text = prompt
        
        bit_idx = 0
        while bit_idx < len(bits):
            # Get top 128 tokens and their probabilities
            token_probs = self.model_fn(prompt, current_text)
            tokens = [tp[0] for tp in token_probs[:self.top_k]]
            probs = [tp[1] for tp in token_probs[:self.top_k]]
            
            # Build Huffman tree for current token probabilities
            root, codes = self.build_huffman_tree(probs)
            
            # Find the longest matching Huffman code in remaining bits
            best_match = None
            best_token_idx = None
            
            for token_idx, code in codes.items():
                if bit_idx + len(code) <= len(bits):
                    candidate_bits = bits[bit_idx:bit_idx + len(code)]
                    if candidate_bits == code:
                        if best_match is None or len(code) > len(best_match):
                            best_match = code
                            best_token_idx = token_idx
            
            # If no exact match found, use the shortest code
            if best_match is None:
                shortest_code = min(codes.values(), key=len)
                for token_idx, code in codes.items():
                    if code == shortest_code:
                        best_match = code
                        best_token_idx = token_idx
                        break
            
            # Select the token
            selected_token = tokens[best_token_idx]
            generated_tokens.append(selected_token)
            current_text += selected_token
            
            bit_idx += len(best_match)
        
        return generated_tokens
    
    def decode(self, prompt: str, generated_tokens: List[str], method: str = 'basic') -> str:
        """
        Decode hidden text from generated tokens.
        
        Args:
            prompt: Original prompt used for encoding
            generated_tokens: List of tokens that were generated
            method: 'basic' or 'huffman'
        """
        if method == 'basic':
            return self._decode_basic(prompt, generated_tokens)
        elif method == 'huffman':
            return self._decode_huffman(prompt, generated_tokens)
        else:
            raise ValueError("Method must be 'basic' or 'huffman'")
    
    def _decode_basic(self, prompt: str, generated_tokens: List[str]) -> str:
        """Decode using basic 7-bit method."""
        bits = []
        current_text = prompt
        
        for token in generated_tokens:
            # Get top 20 tokens from model at this point
            token_probs = self.model_fn(prompt, current_text)
            
            # Find the index of our token in the top 20
            token_idx = None
            for i, (t, _) in enumerate(token_probs[:self.top_k]):
                if t == token:
                    token_idx = i
                    break
            
            if token_idx is None:
                # Token not in top 20, this shouldn't happen in normal operation
                token_idx = 0
            
            # Convert token index to 5 bits
            token_bits = format(token_idx, '05b')
            bits.append(token_bits)
            
            current_text += token
        
        # Combine all bits and convert back to text
        all_bits = ''.join(bits)
        decoded = self.bits_to_text(all_bits)
        return decoded.rstrip('\x00')  # Remove any null padding
    
    def _decode_huffman(self, prompt: str, generated_tokens: List[str]) -> str:
        """Decode using Huffman method."""
        bits = []
        current_text = prompt
        
        for token in generated_tokens:
            # Get top 20 tokens and their probabilities
            token_probs = self.model_fn(prompt, current_text)
            tokens = [tp[0] for tp in token_probs[:self.top_k]]
            probs = [tp[1] for tp in token_probs[:self.top_k]]
            
            # Build Huffman tree for current token probabilities
            root, codes = self.build_huffman_tree(probs)
            
            # Find the token index
            token_idx = None
            for i, t in enumerate(tokens):
                if t == token:
                    token_idx = i
                    break
            
            if token_idx is not None and token_idx in codes:
                bits.append(codes[token_idx])
            
            current_text += token
        
        # Combine all bits and convert back to text
        all_bits = ''.join(bits)
        decoded = self.bits_to_text(all_bits)
        return decoded.rstrip('\x00')  # Remove any null padding


# Example usage and testing
def mock_model_fn(prompt: str, prefix: str) -> List[Tuple[str, float]]:
    """
    Mock language model function for testing.
    Returns top 20 tokens with probabilities.
    """
    # Simple mock that returns consistent tokens with decreasing probabilities
    tokens = []
    for i in range(20):
        token = f"token_{i}"
        prob = 1.0 / (i + 1)  # Decreasing probability
        tokens.append((token, prob))
    return tokens


if __name__ == "__main__":
    # Test the system
    steg_llm = SteganographicLLM(mock_model_fn)
    
    prompt = "Hello, this is a test prompt."
    hidden_text = "Secret message!"
    
    print(f"Original hidden text: {hidden_text}")
    print(f"Binary representation: {steg_llm.text_to_bits(hidden_text)}")
    
    # Test basic encoding
    print("\n--- Basic Encoding ---")
    encoded_basic = steg_llm.encode_basic(prompt, hidden_text)
    print(f"Generated tokens: {encoded_basic}")
    
    decoded_basic = steg_llm.decode(prompt, encoded_basic, 'basic')
    print(f"Decoded text: {repr(decoded_basic)}")
    print(f"Match: {decoded_basic.strip() == hidden_text}")
    
    # Test Huffman encoding
    print("\n--- Huffman Encoding ---")
    encoded_huffman = steg_llm.encode_huffman(prompt, hidden_text)
    print(f"Generated tokens: {encoded_huffman}")
    
    decoded_huffman = steg_llm.decode(prompt, encoded_huffman, 'huffman')
    print(f"Decoded text: {repr(decoded_huffman)}")
    print(f"Match: {decoded_huffman.strip() == hidden_text}")
    print(f"Huffman decoded repr: {repr(decoded_huffman)}")
    print(f"Length difference: {len(decoded_huffman)} vs {len(hidden_text)}")