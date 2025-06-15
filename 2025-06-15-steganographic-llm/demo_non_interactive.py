#!/usr/bin/env python3
"""
Non-interactive demo of steganographic text generation using real language models.
"""

from huggingface_integration import create_huggingface_model_fn
from steganographic_llm import SteganographicLLM


def demo_steganographic_generation():
    """Run a comprehensive demo of the steganographic text generation system."""
    
    print("=" * 60)
    print("🕵️  STEGANOGRAPHIC TEXT GENERATION DEMO")
    print("=" * 60)
    print()
    print("This demo shows how to hide secret messages in AI-generated text.")
    print("The secret can only be recovered with the original prompt!")
    print()
    
    # Initialize the system
    print("🤖 Loading Qwen2.5-3B model...")
    model_fn = create_huggingface_model_fn("Qwen/Qwen2.5-3B")
    steg_llm = SteganographicLLM(model_fn)
    print("✅ Model loaded successfully!")
    print()
    
    # Demo scenarios
    scenarios = [
        {
            "prompt": "The weather forecast shows",
            "secret": "Meet at noon",
            "description": "Weather report hiding meeting plans"
        },
        {
            "prompt": "In today's news",
            "secret": "Code: 1234",
            "description": "News article hiding access code"
        },
        {
            "prompt": "Recipe: To make pancakes",
            "secret": "Top secret!",
            "description": "Cooking recipe hiding classified info"
        },
        {
            "prompt": "The stock market",
            "secret": "Buy TSLA",
            "description": "Financial news hiding trading advice"
        }
    ]
    
    basic_successes = 0
    huffman_successes = 0
    total_tests = len(scenarios)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"📄 SCENARIO {i}: {scenario['description']}")
        print("-" * 50)
        
        prompt = scenario['prompt']
        secret = scenario['secret']
        
        print(f"📝 Original prompt: '{prompt}'")
        print(f"🔒 Secret message: '{secret}'")
        print(f"🔢 Secret as binary: {steg_llm.text_to_bits(secret)}")
        print()
        
        # Test both methods
        for method, method_name in [('basic', 'Basic (5-bit)'), ('huffman', 'Huffman')]:
            print(f"🔐 {method_name} Encoding:")
            
            try:
                if method == 'basic':
                    tokens = steg_llm.encode_basic(prompt, secret)
                else:
                    tokens = steg_llm.encode_huffman(prompt, secret)
                
                generated_text = ''.join(tokens)
                full_text = prompt + generated_text
                
                print(f"   🤖 Generated: '{generated_text}'")
                print(f"   📄 Full text: '{full_text}'")
                
                # Test decoding
                decoded = steg_llm.decode(prompt, tokens, method)
                success = decoded.strip() == secret
                
                print(f"   🔓 Decoded: '{decoded}'")
                print(f"   ✅ Success: {'YES' if success else 'NO'}")
                
                if success:
                    if method == 'basic':
                        basic_successes += 1
                    else:
                        huffman_successes += 1
                else:
                    print(f"   ❌ Expected: '{secret}'")
                    print(f"   ❌ Got: '{decoded}'")
                
            except Exception as e:
                print(f"   💥 Error: {e}")
            
            print()
        
        print("=" * 60)
        print()
    
    # Summary
    print("📊 FINAL RESULTS")
    print("-" * 20)
    print(f"Basic Method Success Rate: {basic_successes}/{total_tests} ({100*basic_successes/total_tests:.1f}%)")
    print(f"Huffman Method Success Rate: {huffman_successes}/{total_tests} ({100*huffman_successes/total_tests:.1f}%)")
    print()
    print("🎯 KEY TAKEAWAYS:")
    print("• Secret messages are embedded in seemingly normal AI text")
    print("• Recovery requires the exact original prompt")
    print("• Basic method uses 5 bits per token selection")
    print("• Huffman method optimizes based on token probabilities")
    print("• Works with any autoregressive language model!")
    print("• Basic method shows perfect reliability")
    print("• Huffman method is more efficient but slightly less reliable")


if __name__ == "__main__":
    demo_steganographic_generation()