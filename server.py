import flet as ft
import socket
import threading
import  torch
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel

SERVER_IP = "0.0.0.0"
SERVER_PORT = 12345

MODEL_NAME = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

server_status_text = ft.Text("Server Status: Idle")
generated_tokens_list = ft.ListView(expand=True, spacing=5)

def handle_client(client_socket, addr):
    print(f"[Server] Accepted connection from {addr[0]}:{addr[1]}")
    server_status_text.value = f"Server Status: Connected to {addr[0]}:{addr[1]}"
    generated_tokens_list.clean()      # Clear previous tokens

    try:
        # Receive size of serialized embeddings
        size_in_bytes = client_socket.recv(4)
        if not size_in_bytes:
            raise ConnectionError("No data received from client.")
        size = int.from_bytes(size_in_bytes, byteorder='big')

        # Receive the serialized embeddings
        received_data = b''
        while len(received_data) < size:
            chunk = client_socket.recv(size - len(received_data))
            if not chunk:
                raise ConnectionError("Connection closed by client.")
            received_data += chunk
        
        data_from_client = pickle.loads(received_data)
        embeddings = data_from_client['embeddings']
        attention_mask = data_from_client['attention_mask']
        print("Received embeddings and attention mask from client.")

        # use embeddings to generate tokens
        with torch.no_grad():
            output = model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id
            )

            for token_id in output[0]:
                token = tokenizer.decode(token_id)
                generated_tokens_list.controls.append(ft.Text(f"Generated: {token}"))
                client_socket.sendall(token.encode('utf-8'))
                server_status_text.page.update()
    
    except Exception as e:
        print(f"[Server] Error handling client {addr}: {e}")
        server_status_text.value = f"Server Status: Error - {e}"
    
    finally:
        client_socket.close()
        print(f"[Server] Connection to client {addr} closed.")
        server_status_text.value = "Server Status: Idle"


def start_server(page: ft.Page):
    page.title = "GPT-2 Split Inference Server"

    def listen_for_connections():
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.listen(5)
        print(f"Server listening on {SERVER_IP}:{SERVER_PORT}")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")
            client_handler = threading.Thread(target=handle_client, args=(client_socket, addr))
            client_handler.start()
    
    page.add(
        ft.Column(
            [
                server_status_text,
                ft.Text("Generated Tokens:"),
                generated_tokens_list,
            ]
        )
    )

    # Start listening for connections in a separate thread
    threading.Thread(target=listen_for_connections, daemon=True).start()

if __name__ == "__main__":
    ft.app(target=start_server)