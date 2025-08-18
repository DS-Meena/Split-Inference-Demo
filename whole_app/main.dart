import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flet/flet.dart'; // Flet's Dart code
import 'package:tflite_flutter/tflite_flutter.dart'; // TFLite plugin

// Global interpreter instance
Interpreter? _interpreter;
bool _modelLoaded = false;
final MethodChannel _platformChannel = const MethodChannel('com.example.gpt2tflite/tflite');

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Handle method calls from Python
  _platformChannel.setMethodCallHandler((call) async {
    switch (call.method) {
      case 'loadModel':
        return await _loadModel();
      case 'runInference':
        final inputIds = Uint8List.fromList(call.arguments['input_ids_bytes']);
        final inputShape = List<int>.from(call.arguments['input_shape']);
        final outputShape = List<int>.from(call.arguments['output_shape']);
        return await _runInference(inputIds, inputShape, outputShape);
      default:
        return Future.error('Method not found: ${call.method}');
    }
  });

  runApp(const FletApp());
}

Future<bool> _loadModel() async {
  try {
    if (_interpreter != null) {
      _interpreter!.close(); // Close existing interpreter if any
    }
    // Load model from assets
    _interpreter = await Interpreter.fromAsset('gpt2.tflite');
    _modelLoaded = true;
    print("TFLite Model loaded successfully in Dart.");
    return true;
  } catch (e) {
    _modelLoaded = false;
    print("Error loading TFLite model: $e");
    return false;
  }
}

Future<Map<String, dynamic>> _runInference(Uint8List inputIdsBytes, List
