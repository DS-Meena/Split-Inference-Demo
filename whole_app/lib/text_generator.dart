import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter_gpt_tokenizer/flutter_gpt_tokenizer.dart';

class TextGenerator {
    late Interpreter _interpreter;
    late Tokenizer _tokenizer;
    late int _vocabSize;
    
    // For handling past_key_values across multiple inference steps
    // final Map<int, Object> _pastKeyValues = {};

    Future<void> loadModel() async {
        // Load the TFLite model
        _interpreter = await Interpreter.fromAsset('assets/sahane.tflite');
        print('Interpreter loaded successfully.');

        // Print input/output tensors for debugging
        for (var i = 0; i < _interpreter.getInputTensors().length; i++) {
            print('Input Tensor $i shape: ${_interpreter.getInputTensor(i).shape}');
        }
        for (var i = 0; i < _interpreter.getOutputTensors().length; i++) {
            print('Output Tensor $i shape: ${_interpreter.getOutputTensor(i).shape}');
            print('Output Tensor $i type: ${_interpreter.getOutputTensor(i).type}');
        }

        // Initialize output tensor shapes for reuse (logits is on 2nd index)
        _vocabSize = _interpreter.getOutputTensor(2).shape[2];
        print('Output Tensor shape: ${_interpreter.getOutputTensor(2).shape}');

        // Initialize empty past_key_values for the first inference run
        // for (int i = 0; i <= 12; i++) {
        //     _pastKeyValues[i] = List.filled(2 * 1 * 12 * 1 * 64, 0.0).reshape([2, 1, 12, 1, 64]);
        // }
    }

    Future<String> generateText(String prompt, {int maxLength = 50}) async {
        print("Enter generateText");

        _tokenizer = Tokenizer();
        List<int> inputTokens = await _tokenizer.encode(prompt, modelName: "text-davinci-002");
        print('Encoded tokens: $inputTokens');

        List<int> generatedIds = [];
        List<int> currentInputTokens = List.from(inputTokens);

        for (int i = 0; i < maxLength; i++) {

            final int sequenceLength = currentInputTokens.length;
            List<int> attentionMask = List.filled(sequenceLength, 1);
            final inputs = [
                Int32List.fromList(attentionMask).reshape([1, sequenceLength]),
                Int32List.fromList(currentInputTokens).reshape([1, sequenceLength]),
            ];

            // Prepare outputs for the interpreter
            var outputs = <int, Object>{};
            // Re-use `_pastKeyValues` for subsequent steps (multiple output layers)
            for (var j = 0; j <= 12; j++) {
                outputs[j] = List.filled(2 * 1 * 12 * sequenceLength * 64, 0.0).reshape([2, 1, 12, sequenceLength, 64]);
            }
            // Main logits output
            outputs[2] = List.filled(1 * sequenceLength * _vocabSize, 0.0).reshape([1, sequenceLength, _vocabSize]);

            _interpreter.runForMultipleInputs(inputs, outputs);

            // Process the logits output to get the next token
            var logits = outputs[2] as List;
            final lastLogits = logits[0][logits[0].length - 1];
            double maxLogit = -double.infinity;
            int predictedTokenId = 0;
            for (int j = 0; j < _vocabSize; j++) {
                if (lastLogits[j] > maxLogit) {
                    maxLogit = lastLogits[j];
                    predictedTokenId = j;
                }
            }
            generatedIds.add(predictedTokenId);
            currentInputTokens.add(predictedTokenId);
            print('Generated token: $predictedTokenId');
        }

        print('GeneratedIds are: $generatedIds');

        // Decode the generated Ids to text
        Uint32List generatedIdsUint32 = Uint32List.fromList(generatedIds);

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
