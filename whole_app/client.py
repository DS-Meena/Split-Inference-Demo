import flet as ft
import numpy as np
import asyncio
import json

from custom_bpe_tokenizer import CustomGPT2Tokenizer

# Client configuration
# Match the channel name defined in main.dart
PLATFORM_CHANNEL_NAME = 'com.example.gpt2tflite/tflite'

# Tokenizer setup
tokenizer = CustomGPT2Tokenizer(
    vocab_file="assets/vocab.json",
    merges_file="assets/merges.txt",
    special_tokens_map_file="assets/special_tokens_map.json",
    tokenizer_config_file="assets/tokenizer_config.json"
)

FIXED_SEQUENCE_LENGTH = 64
TFLITE_INPUT_SHAPE = [1, FIXED_SEQUENCE_LENGTH]
TFLITE_INPUT_DTYPE = np.int32
TFLITE_OUTPUT_SHAPE = [1, tokenizer.vocab_size]
TFLITE_OUTPUT_DTYPE = np.float32

# Flet UI and logic
def main(page: ft.Page):
    page.title = "GPT-2 TFLite On-Device Client"

    prompt_input = ft.TextField(label="Enter your prompt", multiline=True, min_lines=3)
    generate_button = ft.ElevatedButton("Generate Text")
    generated_text_output = ft.Text("")
    status_text = ft.Text("Initializing app...")

    # A flag to indicate if the TFLite model is loaded in Dart
    page._tflite_loaded = False

    # Event for receving TFLite results from Dart
    tflite_result_event = asyncio.Event()
    tflite_inference_output = None

    async def handle_platform_method_call(e):
        # Receives method calls from Dart (e.g. TFLite inference results)
        nonlocal tflite_inference_output

        if e.method_name == 'tfliteResult':
            tflite_inference_output = e.method_arguments['output']
            tflite_result_event.set()
        
        elif e.method_name == 'statusUpdate':
            status_text.value = e.method_arguments['message']
            page.update()
        
        else:
            print(f"Unknown method call from Dart: {e.method_name}")
    
    page.on_platform_method_call = handle_platform_method_call

    async def generate_text_on_device(e):
        nonlocal tflite_inference_output

        prompt = prompt_input.value
        generated_text_output.value = ""
        status_text.value = "Generating..."
        generate_button.disabled = True
        page.update()

        if not page._tflite_loaded:
            status_text.value = "Model not loaded. Please wait or check logs."
            generate_button.disabled = False
            page.update()
            return
        
        try:
            # Take input prompt and prepare input
            # Tokenize 
            tokenized_inputs = tokenizer.tokenize(prompt, padding=True, max_length=FIXED_SEQUENCE_LENGTH)
            input_ids = tokenized_inputs['input_ids']

            # TFLite input expects a flat buffer of bytes
            input_ids_bytes = input_ids.astype(TFLITE_INPUT_DTYPE).tobytes()

            # Run inference (via Platform Channel)
            # Reset event and output variable before calling Dart
            tflite_result_event.clear()
            tflite_inference_output = None

            print("Invoking Dart method 'runInference'...")
            response_map = await page.Platform.invoke_method(
                'runInference',
                {
                    'input_ids_bytes': input_ids_bytes,
                    'input_shape': list(TFLITE_INPUT_SHAPE),
                    'output_shape': list(TFLITE_OUTPUT_SHAPE)
                }
            )

            if response_map['success']:
                output_list = response_map['output']

                # Inference TFLite Output
                # Convert list back to numpy array of the correct data type
                output_np = np.array(output_list, dtype=TFLITE_OUTPUT_DTYPE).reshape(TFLITE_OUTPUT_SHAPE)

                # Assuming output_np are logits for the next token from the model
                next_token_id = np.argmax(output_np[0, :])
                next_token_str = tokenizer.decode([next_token_id])

                # TODO: GENERATE WHOLE TEXT, USING FOR LOOP
                generated_text_output.value = prompt + next_token_str

                status_text.value = "Generation complete."
            
            else:
                status_text.value = f"Inference failed: {response_map['error']}"
                print(f"Dart Inference Error: {response_map['error']}")
        
        except Exception as ex:
            status_text.value = f"Error during on device generation: {ex}"
            print(f"Python Generation Error: {ex}")
        
        finally:
            generate_button.disabled = False
            page.update()
    
    async def load_tflite_model_async():
        # Attempt to load the TFLite model via Dart Platform channel
        status_text.value = "Loading TFLite model..."
        generate_button.disabled = True
        page.update()

        try:
            print("Invoking Dart method 'loadModel'...")
            load_success = await page.invoke_method(
                'loadModel',
                arguments={},
                channel_name=PLATFORM_CHANNEL_NAME
                )

            if load_success:
                status_text.value = "TFLite model loaded successfully!"
                page._tflite_loaded = True
            else:
                status_text.value = "Failed to load TFLite model in Dart."
                page._tflite_loaded = False
            
        except Exception as e:
            status_text.value = f"Error invoking laodModel: {e}. Check if Dart/Native code is correctly configured."
            page._tflite_loaded = False
            print(f"Error invoking loadModel from python: {e}")
        
        finally:
            generate_button.disabled = not page._tflite_loaded
            page.update()
        
    # Initial setup when the page connects
    async def on_page_connect(e):
        print("Page connected! Attempting to laod TFLite mode...")
        await load_tflite_model_async()
    
    print("connection here")
    page.on_connect = on_page_connect
    
    # Flet UI Layout
    page.add(
        ft.Column(
            [
                status_text,
                prompt_input,
                generate_button,
                ft.Text("Generated Text:"),
                generated_text_output,
            ]
        )
    )

if __name__=="__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER)
