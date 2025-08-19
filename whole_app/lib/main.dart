import 'package:flutter/material.dart';
import 'package:whole_app/text_generator.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final TextEditingController _promptController = TextEditingController();
  String _generatedText = '';
  final TextGenerator _textGenerator = TextGenerator();

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    await _textGenerator.loadModel();
    setState(() {});   // Refresh UI after model load
  }

  void _generateText() {
    final prompt = _promptController.text;
    final generated = _textGenerator.generateText(prompt);

    setState(() {
      _generatedText = generated;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('GPT2 Text Generator')),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              TextField(
                controller: _promptController,
                decoration: const InputDecoration(labelText: 'Enter your prompt'),
              ),
              ElevatedButton(
                onPressed: _generateText,
                child: const Text('Generate Text'),
              ),
              const SizedBox(height: 20),
              Text('Generated Text: $_generatedText'),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _promptController.dispose();
    _textGenerator.dispose();
    super.dispose();
  }
}