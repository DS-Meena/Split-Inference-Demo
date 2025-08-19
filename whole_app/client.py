import flet as ft
import numpy as np
import asyncio
import json

from custom_bpe_tokenizer import CustomGPT2Tokenizer # Assuming this is correctly implemented and available

# Client configuration
PLATFORM_CHANNEL_NAME = 'com.example.gpt2tflite/tflite'

# Tokenizer setup
try:
    # Attempt to initialize the tokenizer (handle potential FileNotFoundError)
    tokenizer = CustomGPT2Tokenizer(
        vocab_file="assets/vocab.json",
        merges_file="assets/merges.txt",
        special_tokens_map_file="assets/special_tokens_map.json",
        tokenizer_config_file="assets/tokenizer_config.json"
    )
except FileNotFoundError as e:
    print(f"Error loading tokenizer files: {e}. Ensure 'assets' folder is correct.")
    # You might want to exit here or load a dummy tokenizer if the app can't proceed without it.
    tokenizer = None

FIXED_SEQUENCE_LENGTH = 64
# Ensure TFLITE_INPUT_DTYPE matches the expected type by your TFLite model,
# typically tf.int32 for token IDs.
TFLITE_INPUT_DTYPE = np.int32 
TFLITE_INPUT_SHAPE = [1, FIXED_SEQUENCE_LENGTH] 
TFLITE_OUTPUT_SHAPE = [1, tokenizer.vocab_size] if tokenizer else [1, 1000] # Fallback if tokenizer fails
TFLITE_OUTPUT_DTYPE = np.float32

# Event to signal when Dart inference result is available
tflite_result_event = asyncio.Event() 
tflite_inference_output = None

# Flet UI and logic
def main(page: ft.Page):
    page.title = "GPT-2 TFLite On-Device Client"

    prompt_input = ft.TextField(label="Enter your prompt", multiline=True, min_lines=3, expand=True)
    generate_button = ft.ElevatedButton("Generate Text", disabled=True)
    generated_text_output = ft.Text("")
    status_text = ft.Text("Initializing app...")

    #  Initialize shared state in page.session
    page.session.set('model_loaded', False)

    #  Handle platform method calls from Dart
    async def handle_platform_method_call(e): # Removed the type hint here
        global tflite_inference_output

        if e.method_name == 'tfliteResult':
            tflite_inference_output = e.method_arguments
            tflite_result_event.set() # Set the event to signal completion

        elif e.method_name == 'statusUpdate':
            status_text.value = e.method_arguments.get('message', '')
            page.update()

        elif e.method_name == 'modelLoaded':
            # Update the 'model_loaded' flag in session
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

    # Async function for generating text
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
            # Tokenize input prompt and prepare for TFLite
            tokenized_inputs = tokenizer(
                prompt,
                padding='max_length',
                max_length=FIXED_SEQUENCE_LENGTH,
                truncation=True,
                return_tensors='np' # Return numpy arrays
            )
            input_ids = tokenized_inputs['input_ids']

            # TFLite input expects a flat buffer of bytes
            input_ids_bytes = input_ids.astype(TFLITE_INPUT_DTYPE).tobytes()

            print("Invoking Dart method 'runInference'...")

            # Set inference data for Dart
            page.platform_data['inference_input_data'] = {
                'input_ids_bytes': input_ids_bytes,
                'input_shape': list(TFLITE_INPUT_SHAPE),
                'output_shape': list(TFLITE_OUTPUT_SHAPE),
            }
            # Set flag to trigger inference in Dart
            page.platform_data['py_flags']['run_inference_request'] = True
            page.update()

            # Wait for Dart to send back the result
            tflite_result_event.clear() # Clear event before waiting
            await asyncio.wait_for(tflite_result_event.wait(), timeout=60) # Increased timeout

            if tflite_inference_output and tflite_inference_output.get('success'):
                output_list = tflite_inference_output['output']

                # Convert output list back to numpy array
                output_np = np.array(output_list, dtype=TFLITE_OUTPUT_DTYPE).reshape(TFLITE_OUTPUT_SHAPE)

                # TODO: Implement a proper next-token sampling and generation loop
                # This is a basic example of taking the argmax of the logits for the next token.
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
            # Reset flags after inference
            page.platform_data['py_flags']['run_inference_request'] = False
            page.platform_data['inference_input_data'] = {}
            page.update()
            generate_button.disabled = False

    generate_button.on_click = generate_text_on_device

    # Custom control for initial setup, inheriting from ft.Container
    class InitialSetup(ft.Container):
        def build(self):
            return ft.Container() # It can be an empty container, just needs to be added

        def did_mount(self):
            print("Python: Initial setup triggered via did_mount. Signalling Dart to load model.")
            self.page.platform_data['py_flags']['load_model_request'] = True
            status_text.value = "Loading TFLite model..."
            generate_button.disabled = True
            self.page.update()


    page.add(
        ft.Column(
            [
                InitialSetup(), # Add the custom control here
                status_text,
                prompt_input,
                generate_button,
                ft.Text("Generated Text:"),
                generated_text_output,
            ]
        )
    )

if __name__ == "__main__":
    ft.app(target=main)
