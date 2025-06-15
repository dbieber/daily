#!/usr/bin/env python3
"""
Interactive demo of steganographic text generation using real language models.
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
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"📄 SCENARIO {i}: {scenario['description']}")
        print("-" * 40)
        
        prompt = scenario['prompt']
        secret = scenario['secret']
        
        print(f"Original prompt: '{prompt}'")
        print(f"Secret message: '{secret}'")
        print(f"Secret as binary: {steg_llm.text_to_bits(secret)}")
        print()
        
        # Test both methods
        for method, method_name in [('basic', 'Basic (7-bit)'), ('huffman', 'Huffman')]:
            print(f"🔐 {method_name} Encoding:")
            
            try:
                if method == 'basic':
                    tokens = steg_llm.encode_basic(prompt, secret)
                else:
                    tokens = steg_llm.encode_huffman(prompt, secret)
                
                generated_text = ''.join(tokens)
                full_text = prompt + generated_text
                
                print(f"   Generated: '{generated_text}'")
                print(f"   Full text: '{full_text}'")
                
                # Test decoding
                decoded = steg_llm.decode(prompt, tokens, method)
                success = decoded.strip() == secret
                
                print(f"   Decoded: '{decoded}'")
                print(f"   Success: {'✅' if success else '❌'}")
                
                if not success:
                    print(f"   Expected: '{secret}'")
                    print(f"   Got: '{decoded}'")
                
            except Exception as e:
                print(f"   Error: {e}")
            
            print()
        
        print("=" * 60)
        print()
    
    # Interactive section
    print("🎮 INTERACTIVE MODE")
    print("-" * 20)
    print("Try your own prompt and secret message!")
    print("(Press Enter with empty input to skip)")
    print()
    
    while True:
        prompt = input("Enter your prompt: ").strip()
        if not prompt:
            break
            
        secret = input("Enter your secret message: ").strip()
        if not secret:
            break
        
        print(f"\n🔐 Encoding '{secret}' with prompt '{prompt}'")
        print()
        
        try:
            # Basic method
            tokens = steg_llm.encode_huffman(prompt, secret)
            generated = ''.join(tokens)
            full_text = prompt + generated
            
            print(f"Generated text: '{full_text}'")
            print(f"Hidden message recoverable with original prompt: '{prompt}'")
            
            # Verify decoding
            decoded = steg_llm.decode(prompt, tokens, 'huffman')
            print(f"Verification - Decoded: '{decoded}' ({'✅' if decoded.strip() == secret else '❌'})")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()
        continue_demo = input("Try another? (y/n): ").strip().lower()
        if continue_demo != 'y':
            break
        print()
    
    print("🎯 DEMO COMPLETE")
    print("Key takeaways:")
    print("• Secret messages are embedded in seemingly normal AI text")
    print("• Recovery requires the exact original prompt")
    print("• Basic method uses 7 bits per token selection")
    print("• Huffman method optimizes based on token probabilities")
    print("• Works with any autoregressive language model!")


if __name__ == "__main__":
    demo_steganographic_generation()