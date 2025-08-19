import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class TextGenerator {
    late Interpreter _interpreter;
    late Map<String, int> _vocab;
    late List<List<String>> _merges;

    Future<void> loadModel() async {
        // Load the TFLite model
        _interpreter = await Interpreter.fromAsset('assets/gpt2.tflite');

        // Load the vocabulary file
        final String vocabString = await rootBundle.loadString('assets/vocab.json');
        _vocab = Map<String, int>.from(json.decode(vocabString));
        
        vocabString.split('\n').forEach((line) {
            final parts = line.split(' ');
            if (parts.length == 2) {
                _vocab[parts[0]] = int.parse(parts[1]);
            }
        });

        final String mergesString = await rootBundle.loadString('assets/merges.txt');
        _merges = mergesString.split('\n').map((line) => line.split(' ')).toList();
    }

    String generateText(String prompt, {int maxLength = 50}) {
        // Convert prompt to input tensor using vocabulary
        final inputTokens = _encode(prompt);

        final output = List.filled(1 * maxLength, 0).reshape([1, maxLength]);

        _interpreter.run([inputTokens], output);

        // convert output tensor to generated text using vocabulary
        final generatedIds = output[0].cast<int>();
        final generatedText = _decode(generatedIds);

        return generatedText;
    }

    // Helper functions
    List<int> _encode(String text) {
        List<String> tokens = text.runes.map((rune) => String.fromCharCode(rune)).toList();

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

        // convert tokens to IDs
        return tokens.map((token) => _vocab[token] ?? _vocab['<|endoftext|>']!).toList();
    }

    String _decode(List<int> tokenIds) {
        final List<String> tokens = tokenIds.map((id) => _idToToken(id)).toList();
        return tokens.join('');
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