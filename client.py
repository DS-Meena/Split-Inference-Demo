import flet as ft
import socket
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SERVER_IP = "172.17.255.44"
SERVER_PORT = 12345

MODEL_NAME = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

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
        inputs = tokenizer(prompt, return_tensors="pt")
        print("Tokenization complete, inputs:", inputs)

        with torch.no_grad():
            embeddings = model.get_input_embeddings()(inputs['input_ids'])
        print("Embeddings generated, shape:", embeddings.shape)

        attention_mask = inputs['attention_mask']
        data_to_send = {
            'embeddings': embeddings,
            'attention_mask': attention_mask
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
