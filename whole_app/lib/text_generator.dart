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
        _interpreter = await Interpreter.fromAsset('assets/gpt2-full.tflite');
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

    Future<String> generateText(String prompt, {int maxLength = 50}) async {
        print("Enter generateText");

        // Tokenization
        _tokenizer = Tokenizer();
        List<int> inputTokens = await _tokenizer.encode(prompt, modelName: "text-davinci-002");

        inputTokens = List<int>.from(inputTokens);
        const expectedLength = 64;
        if (inputTokens.length < expectedLength) {
            inputTokens.addAll(List.filled(expectedLength - inputTokens.length, 0));
        } else if (inputTokens.length > expectedLength) {
            inputTokens = inputTokens.sublist(0, expectedLength);
        }
        
        print('Encoded tokens before: $inputTokens');

        // Do inference (using multiple outputs options, only index 0 output matters)
        final inputs = [[inputTokens]];

        var output = List.filled(1 * _outputSequenceLength * _vocabSize, 0.0).reshape([1, _outputSequenceLength, _vocabSize]);
        var output_tmp = List.filled(1 * 2 * 12 * 64 * 64, 0).reshape([1, 2, 12, 64, 64]);
        var map = <int, Object>{};
        map[0] = output;
        for (int i=1; i<=12; i++) {
            map[i] = output_tmp;
        }

        _interpreter.runForMultipleInputs(inputs, map);

        print('output of interpreter is $output');

        // process the output to get the generated Ids
        // choose id with max logit value from each output dimension (50)
        List<int> generatedIds = [];
        for (int i = 0; i < _outputSequenceLength; i++) {
            double maxLogit = -double.infinity;
            int predictedTokenId = 0;
            for (int j = 0; j < _vocabSize; j++) {
                if (output[0][i][j] > maxLogit) { // Assuming output is Float32, adjust type if needed
                    maxLogit = output[0][i][j];
                    predictedTokenId = j;
                }
            }
            generatedIds.add(predictedTokenId);
        }
        print('GeneratedIds are: $generatedIds');

        // Decode the generated Ids to text
        Uint32List generatedIdsUint32 = Uint32List.fromList(generatedIds);

        String actualGeneratedText = await _tokenizer.decode(
            generatedIdsUint32,
            modelName: "text-davinci-002",
            allowedSpecialTokens: ['<|endoftext|>'],
        );

        // final actualGeneratedText = _decode(generatedIds);
        print('actualGeneratedText: $actualGeneratedText');
        
        // int endOfTextIndex = generatedIds.indexOf(_vocab['<|endoftext|>'] ?? 0);
        // if (endOfTextIndex == -1) {
        //     endOfTextIndex = generatedIds.length; // No end-of-text found, take all
        // }

        // final finalGeneratedText = _decode(generatedIds.sublist(0, endOfTextIndex)).trim();

        return actualGeneratedText;
    }

    void dispose() {
        _interpreter.close();
    }
}