import flet as ft
import socket
import pickle
import numpy as np
from transformers import AutoTokenizer
from head_model_arch import NumPyEmbedding

SERVER_IP = "172.17.255.44"
SERVER_PORT = 12345

MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  

# Load the Head model
vocalb_size = tokenizer.vocab_size
embedding_dim = 768                 # 768 for gpt2, 1024 for gpt2-large
max_position_embeddings = 1024      # 1024 for gpt2, 2048 for gpt2-large

client_model = NumPyEmbedding(
    token_embeddings_path="assets/token_embeddings.npy",
    position_embeddings_path="assets/position_embeddings.npy",
    max_position_embeddings=max_position_embeddings
)

def main(page: ft.Page):
    page.title = "GPT-2 Split Inference Client"

    prompt_input = ft.TextField(label="Enter your prompt", multiline=True, expand=True)
    generate_button = ft.ElevatedButton("Generate")
    generated_text_output = ft.Text("")

    def generate_text(e):
        print("Generate Text: Entry")
        prompt = prompt_input.value
        generated_text_output.value = ""   # clear previous output
        page.update()

        # 1. Tokenizations and embeddings on the Client
        inputs = tokenizer(prompt, return_tensors="np", padding=True)
        print("Tokenization complete, inputs:", inputs)

        embeddings_np = client_model(inputs['input_ids'])
        print("Embeddings generated, shape:", embeddings_np.shape)

        attention_mask_np = inputs['attention_mask']
        data_to_send = {
            'embeddings': embeddings_np,
            'attention_mask': attention_mask_np
        }
        
        # 2. Serialize and send embeddings to the server
        try: 
            print("Connecting to server...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((SERVER_IP, SERVER_PORT))

            # Serialize embeddings
            pickled_embeddings = pickle.dumps(data_to_send)

            # Share embeddings along with its size
            size = len(pickled_embeddings)
            size_in_bytes = size.to_bytes(4, byteorder='big')
            client_socket.sendall(size_in_bytes)
            client_socket.sendall(pickled_embeddings)

            print("Embeddings sent to server.")
            # Receive and display genreated tokens from the server
            while True:
                token = client_socket.recv(1024).decode('utf-8')
                if not token:   # No more tokens
                    break
                generated_text_output.value += token
                page.update()
        
        except ConnectionRefusedError:
            generated_text_output.value = "Connection to server failed. Is the server running?"
        except Exception as ex:
            generated_text_output.value = f"An error occurred: {ex}"
        finally:
            client_socket.close()
    
    generate_button.on_click = generate_text
    page.add(
        ft.Column(
            [
                prompt_input,
                generate_button,
                ft.Text("Generated Text:"),
                generated_text_output
            ]
        )
    )

if __name__ == "__main__":
    ft.app(target=main)
