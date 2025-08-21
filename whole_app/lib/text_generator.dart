import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class TextGenerator {
    late Interpreter _interpreter;
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

    String generateText(String prompt, {int maxLength = 50}) {
        print("Enter generateText");
        // Convert prompt to input tensor using vocabulary
        final inputTokens = _encode(prompt, expectedLength: 64);
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

        final actualGeneratedText = _decode(generatedIds);
        print('actualGeneratedText: $actualGeneratedText');

        int endOfTextIndex = generatedIds.indexOf(_vocab['<|endoftext|>'] ?? 0);
        if (endOfTextIndex == -1) {
            endOfTextIndex = generatedIds.length; // No end-of-text found, take all
        }

        final finalGeneratedText = _decode(generatedIds.sublist(0, endOfTextIndex)).trim();

        return finalGeneratedText;
    }

    // Helper functions
    List<int> _encode(String text, {required int expectedLength}) {
        List<String> tokens = text.runes.map((rune) => String.fromCharCode(rune)).toList();
        print("tokens are $tokens");

        // Apply merge rules
        for (final merge in _merges) {
            if (merge.length != 2) {
                continue;
            } 
            final newCompoundToken = merge.join('');

            // Continuously apply the merge rule until no more instances of the pair are found
            while(true) {
                int bestIndex = -1;
                for (int i=0; i<tokens.length - 1; i++) {
                    if (tokens[i] == merge[0] && tokens[i+1] == merge[1]) {
                        bestIndex = i;
                        break;
                    }
                }

                if (bestIndex != -1) {
                    tokens.replaceRange(bestIndex, bestIndex+2, [newCompoundToken]);
                } else {
                    break;
                }
            }
        }

        print("tokens after $tokens");
        // remove space " " from tokens
        tokens.removeWhere((token) => token == " ");
        
        List<int> encodedTokens = tokens.map((token) => _vocab[token] ?? _vocab['<|endoftext|>'] ?? 0).toList();
        print("encodedTokens are $encodedTokens");


        if (encodedTokens.length < expectedLength) {
            encodedTokens.addAll(List.filled(expectedLength - encodedTokens.length, 0));
        } else if (encodedTokens.length > expectedLength) {
            encodedTokens = encodedTokens.sublist(0, expectedLength);
        }

        return encodedTokens;
    }

    String _decode(List<int> tokenIds) {
        final List<String> tokens = tokenIds.map((id) => _idToToken(id)).toList();
        return tokens.join(' ');
    }

    String _idToToken(int id) {
        return _vocab.entries.firstWhere(
            (entry) => entry.value == id,
            orElse: () => MapEntry('<|endoftext|>', _vocab['<|endoftext|>']!),
        ).key;
    }

    void dispose() {
        _interpreter.close();
    }
}