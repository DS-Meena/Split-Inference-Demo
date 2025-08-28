# whole_app

A new Flutter project.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

## To Generate more text 

```dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter_gpt_tokenizer/flutter_gpt_tokenizer.dart';

class TextGenerator {
    late Interpreter _interpreter;
    late Tokenizer _tokenizer;
    late Map<String, int> _vocab;
    late List<List<String>> _merges;
    late int _outputSequenceLength;
    late int _vocabSize;

    Future<void> loadModel() async {
        // Load the TFLite model
        _interpreter = await Interpreter.fromAsset('assets/gpt2-large.tflite');
        print('Input Tensor shape: ${_interpreter.getInputTensor(0).shape}');
        print('Output Tensor shape: ${_interpreter.getOutputTensor(0).shape}');
        print('Output Tensor type: ${_interpreter.getOutputTensor(0).type}');

        // Load the vocabulary file
        final String vocabString = await rootBundle.loadString('assets/vocab.json');
        _vocab = Map<String, int>.from(json.decode(vocabString));
        print('vocab is following: $_vocab');

        final String mergesString = await rootBundle.loadString('assets/merges.txt');
        _merges = mergesString.split('\n').map((line) => line.split(' ')).toList();
        print('merges is following: $_merges');

        // Get and store the output tensor shape
        final outputTensor = _interpreter.getOutputTensor(0);
        _outputSequenceLength = outputTensor.shape[1]; // Should be 64
        _vocabSize = outputTensor.shape[2]; // Should be 50257
        print('Output Tensor shape: ${outputTensor.shape}');
    }

    Future<String> generateText(String prompt, {int maxLength = 10}) async {
        print("Enter generateText");

        _tokenizer = Tokenizer();
        List<int> inputTokens = await _tokenizer.encode(prompt, modelName: "text-davinci-002");
        inputTokens = List<int>.from(inputTokens);
        List<int> generatedTokens = [];
        const expectedLength = 64;

        // to store the output
        var output = List.filled(1 * _outputSequenceLength * _vocabSize, 0.0).reshape([1, _outputSequenceLength, _vocabSize]);
        var output_tmp = List.filled(1 * 2 * 12 * 64 * 64, 0).reshape([1, 2, 12, 64, 64]);
        var map = <int, Object>{};
        map[0] = output;
        for (int i=1; i<=12; i++) {
            map[i] = output_tmp;
        }

        for (int i=0; i<maxLength; i++) {
          
          // prepare Input tensor for this iteration
          List<int> currentInput = List<int>.from(inputTokens);

          if (currentInput.length < expectedLength) {
              currentInput.addAll(List.filled(expectedLength - currentInput.length, 0));
          } else if (currentInput.length > expectedLength) {
              currentInput = currentInput.sublist(0, expectedLength);
          }

          // final inputTensorBuffer = TensorBuffer.createFixedSize([1, expectedLength], TfLiteType.int32);
          // inputTensorBuffer.loadArray(currentInput, [1, expectedLength]);

          // Run inference
          print('Encoded tokens before: $currentInput');

          // Do inference (using multiple outputs options, only index 0 output matters)
          final inputs = [[currentInput]];
          _interpreter.runForMultipleInputs(inputs, map);

          // get the next token
          double maxLogit = -double.infinity;
          int predictedTokenId = 0;
          for (int j = 0; j < _vocabSize; j++) {
              if (output[0][0][j] > maxLogit) { // Assuming output is Float32, adjust type if needed
                  maxLogit = output[0][0][j];
                  predictedTokenId = j;
              }
          }

          // add the output to input for next call
          inputTokens.add(predictedTokenId);
          generatedTokens.add(predictedTokenId);

          if (inputTokens.length > _outputSequenceLength) {
              break;
          }
        }

        // Decode the generated Ids to text
        Uint32List generatedIdsUint32 = Uint32List.fromList(generatedTokens);

        String actualGeneratedText = await _tokenizer.decode(
            generatedIdsUint32,
            modelName: "text-davinci-002",
            allowedSpecialTokens: ['<|endoftext|>'],
        );

        print('actualGeneratedText: $actualGeneratedText');
        
        return actualGeneratedText;
    }

    void dispose() {
        _interpreter.close();
    }
}
```