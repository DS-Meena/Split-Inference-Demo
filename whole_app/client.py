# client.py
import flet as ft
import numpy as np
import asyncio
import json

from custom_bpe_tokenizer import CustomGPT2Tokenizer 

# Client configuration
PLATFORM_CHANNEL_NAME = 'com.example.gpt2tflite/tflite'

# Tokenizer setup
try:
    tokenizer = CustomGPT2Tokenizer(
        vocab_file="assets/vocab.json",
        merges_file="assets/merges.txt",
        special_tokens_map_file="assets/special_tokens_map.json",
        tokenizer_config_file="assets/tokenizer_config.json"
    )
except FileNotFoundError as e:
    print(f"Error loading tokenizer files: {e}. Ensure 'assets' folder is correct.")
    tokenizer = None

FIXED_SEQUENCE_LENGTH = 64
TFLITE_INPUT_DTYPE = np.int32 
TFLITE_INPUT_SHAPE = [1, FIXED_SEQUENCE_LENGTH] 
TFLITE_OUTPUT_SHAPE = [1, tokenizer.vocab_size] if tokenizer else [1, 50257] # Fallback for tokenizer fail
TFLITE_OUTPUT_DTYPE = np.float32

# Event to signal when Dart inference result is available
tflite_result_event = asyncio.Event() 
tflite_inference_output = None

def main(page: ft.Page):
    page.title = "GPT-2 TFLite On-Device Client"

    prompt_input = ft.TextField(label="Enter your prompt", multiline=True, min_lines=3, expand=True)
    generate_button = ft.ElevatedButton("Generate Text", disabled=True)
    generated_text_output = ft.Text("")
    status_text = ft.Text("Initializing app...")

    page.session.set('model_loaded', False)

    async def handle_platform_method_call(e):
        global tflite_inference_output

        if e.method_name == 'tfliteResult':
            tflite_inference_output = e.method_arguments
            tflite_result_event.set()

        elif e.method_name == 'statusUpdate':
            status_text.value = e.method_arguments.get('message', '')
            page.update()

        elif e.method_name == 'modelLoaded':
            page.session.set('model_loaded', e.method_arguments.get('success', False))
            if page.session.get('model_loaded'):
                status_text.value = "TFLite model loaded successfully"
            else:
                status_text.value = f"Failed to load TFLite model: {e.method_arguments.get('error', 'Unknown error')}"
            generate_button.disabled = not page.session.get('model_loaded')
            page.update()

        else:
            print(f"Unknown method call from Dart: {e.method_name}")

    page.on_platform_method_call = handle_platform_method_call

    async def generate_text_on_device(e):
        global tflite_inference_output

        prompt = prompt_input.value
        generated_text_output.value = ""
        status_text.value = "Generating..."
        generate_button.disabled = True
        page.update()

        if not page.session.get('model_loaded'):
            status_text.value = "Model not loaded. Please wait or check logs."
            generate_button.disabled = False
            page.update()
            return
        
        if tokenizer is None:
            status_text.value = "Tokenizer not loaded. Cannot proceed with generation."
            generate_button.disabled = False
            page.update()
            return

        try:
            tokenized_inputs = tokenizer(
                prompt,
                padding='max_length',
                max_length=FIXED_SEQUENCE_LENGTH,
                truncation=True,
                return_tensors='np'
            )
            input_ids = tokenized_inputs['input_ids']
            input_ids_bytes = input_ids.astype(TFLITE_INPUT_DTYPE).tobytes()

            print("Invoking Dart method 'runInference'...")
            # Use send_method_call to directly invoke the Dart method with arguments
            page.send_method_call(
                'runInference',
                {
                    'input_ids_bytes': list(input_ids_bytes), # Convert bytes to list for serialization
                    'input_shape': list(TFLITE_INPUT_SHAPE),
                    'output_shape': list(TFLITE_OUTPUT_SHAPE),
                }
            )

            tflite_result_event.clear()
            await asyncio.wait_for(tflite_result_event.wait(), timeout=60)

            if tflite_inference_output and tflite_inference_output.get('success'):
                output_list = tflite_inference_output['output']
                output_np = np.array(output_list, dtype=TFLITE_OUTPUT_DTYPE).reshape(TFLITE_OUTPUT_SHAPE)

                next_token_id = np.argmax(output_np[0, :])
                next_token_str = tokenizer.decode([next_token_id])

                generated_text_output.value = prompt + next_token_str
                status_text.value = "Generation complete."
            else:
                status_text.value = f"Inference failed: {tflite_inference_output.get('error', 'Unknown error')}"
                print(f"Dart Inference Error: {tflite_inference_output.get('error', 'Unknown error')}")

        except asyncio.TimeoutError:
            status_text.value = "Inference timed out. Dart might not be responding."
            print("Python Timeout error: Dart did not respond within the expected time.")
        except Exception as ex:
            status_text.value = f"Error during on-device generation: {ex}"
            print(f"Python Generation Error: {ex}")
        finally:
            page.update()
            generate_button.disabled = False
    
    generate_button.on_click = generate_text_on_device

    # Function to trigger model loading
    async def load_model_on_start():
        print("Python: Signalling Dart to load model on app start.")
        status_text.value = "Loading TFLite model..."
        generate_button.disabled = True
        page.update()
        # Use send_method_call to directly invoke 'loadModel' in Dart
        page.send_method_call('loadModel', {}) 

    # We use a simple `ft.Container` and its `did_mount` method to trigger the initial load.
    # This avoids the need for `page.once_loaded` and `platform_data`.
    class InitialSetupTrigger(ft.Container):
        def build(self):
            return ft.Container() # Can be an empty container, or a hidden text etc.

        def did_mount(self):
            # Run the async load_model_on_start function
            asyncio.create_task(load_model_on_start())

    page.add(
        ft.Column(
            [
                InitialSetupTrigger(), # Add the trigger control here
                status_text,
                prompt_input,
                generate_button,
                ft.Text("Generated Text:"),
                generated_text_output,
            ]
        )
    )

if __name__=="__main__":
    ft.app(target=main)
