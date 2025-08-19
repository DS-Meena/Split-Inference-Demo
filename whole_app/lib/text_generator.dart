import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class TextGenerator {
    late Interpreter _interpreter;
    late Map<String, int> _vocab;

    Future<void> loadModel() async {
        // Load the TFLite model
        _interpreter = await Interpreter.fromAsset('assets/gpt2.tflite');

        // Load the vocabulary file
        final String vocabString = await rootBundle.loadString('assets/vocab.txt');
        _vocab = {};

        vocabString.split('\n').forEach((line) {
            final parts = line.split(' ');
            if (parts.length == 2) {
                _vocab[parts[0]] = int.parse(parts[1]);
            }
        });
    }

    String generateText(String prompt, {int maxLength = 50}) {
        // Convert prompt to input tensor using vocabulary
        final input = _tokenize(prompt);

        final output = List.filled(1 * maxLength, 0).reshape([1, maxLength]);

        _interpreter.run(input, output);

        // convert output tensor to generated text using vocabulary
        final generatedTokens = output[0].map<String>((e) => _detokenize(e as int)).toList();
        
        return generatedTokens.join(' ');
    }

    // Helper functions
    List<int> _tokenize(String text) {
        return text.split(' ').map((word) => _vocab[word] ?? 0).toList();
    }

    String _detokenize(int token) {
        return _vocab.entries.firstWhere((entry) => entry.value == token, orElse: () => MapEntry('', 0)).key;
    }

    void dispose() {
        _interpreter.close();
    }
}