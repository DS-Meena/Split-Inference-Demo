from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

output_dir = "client_app/assets"
tokenizer.save_pretrained(output_dir)

print(f"Tokenizer files saved to {output_dir}")