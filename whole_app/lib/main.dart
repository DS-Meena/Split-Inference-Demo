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

  // Use a Future to track the loading state of the model
  late Future<void> _loadingModelFuture;

  @override
  void initState() {
    super.initState();
    _loadingModelFuture = _textGenerator.loadModel();
  }

  void _generateText() {
    final prompt = _promptController.text;
    final generated = _textGenerator.generateText(prompt);

    setState(() {
      _generatedText = generated.toString();
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

              // Use FutureBuilder to show a loading indicator while the model is loading
              FutureBuilder(
                future: _loadingModelFuture,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.done) {
                    if (snapshot.hasError) {
                      return Text('Error loading model: ${snapshot.error}');
                    } else {
                      return ElevatedButton(
                        onPressed: _generateText,
                        child: const Text('Generate Text'),
                      );
                    }
                  } else {
                    // Model is still loading
                    return const CircularProgressIndicator();
                  }
                },
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