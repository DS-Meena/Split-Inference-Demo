// main.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flet/flet.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:provider/provider.dart';

Interpreter? _interpreter;
bool _modelLoaded = false;
final MethodChannel _platformChannel = const MethodChannel('com.example.gpt2tflite/tflite');

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  runApp(
    Provider<FletPage>(
      create: (_) => FletPage(
        pageUrl: "http://localhost:8550", // Or your Flet server URL
      ),
      child: const FletApp(),
    ),
  );
}

Future<bool> _loadModel(FletPage page) async {
  try {
    if (_interpreter != null) {
      _interpreter!.close();
    }
    _interpreter = await Interpreter.fromAsset('gpt2.tflite');
    _modelLoaded = true;
    print("TFLite Model loaded successfully in Dart.");
    // Send model loaded status back to Python
    page.sendMethodCall('modelLoaded', {'success': true, 'message': "TFLite model loaded successfully!"});
    return true;
  } catch (e) {
    _modelLoaded = false;
    print("Error loading TFLite model: $e");
    // Send model loaded failure status back to Python
    page.sendMethodCall('modelLoaded', {'success': false, 'error': e.toString(), 'message': "Error loading TFLite model: $e"});
    return false;
  }
}

Future<Map<String, dynamic>> _runInference(FletPage page, Map<String, dynamic> arguments) async {
    if (!_modelLoaded || _interpreter == null) {
        // Send inference failure status back to Python
        page.sendMethodCall('tfliteResult', {'success': false, 'error': 'TFLite model not loaded.'});
        return {'success': false, 'error': 'TFLite model not loaded.'};
    }

    try {
        // arguments are passed directly via the method call
        final inputIdsBytes = arguments['input_ids_bytes'] as List<dynamic>;
        final inputShape = List<int>.from(arguments['input_shape']);
        final outputShape = List<int>.from(arguments['output_shape']);

        final inputBuffer = Uint8List.fromList(inputIdsBytes.cast<int>());
        final outputBuffer = Float32List(outputShape.reduce((a, b) => a * b)).buffer;

        _interpreter!.resizeInput(0, inputShape);
        _interpreter!.allocateTensors();

        _interpreter!.run(inputBuffer, outputBuffer);

        final List<double> outputList = outputBuffer.asFloat32List().toList();

        return {'success': true, 'output': outputList};
    } catch(e) {
        print("Error during TFLite inference: $e");
        return {'success': false, 'error': e.toString()};
    }
}

class FletApp extends StatefulWidget {
    const FletApp({Key? key}) : super(key: key);

    @override
    State<FletApp> createState() => _FletAppState();
}

class _FletAppState extends State<FletApp> {
    FletPage? _fletPage;

    @override
    void initState() {
        super.initState();
        WidgetsBinding.instance.addPostFrameCallback((_) {
            _fletPage = Provider.of<FletPage>(context, listen: false);
            _setupMethodCallHandler();
        });
    }

    void _setupMethodCallHandler() {
        _platformChannel.setMethodCallHandler((call) async {
            if (_fletPage == null) {
                print("FletPage is not initialized yet.");
                return Future.error('FletPage not ready.');
            }

            switch (call.method) {
                case 'loadModel':
                    print("Dart: Received loadModel call from Python.");
                    await _loadModel(_fletPage!);
                    break;
                case 'runInference':
                    print("Dart: Received runInference call from Python.");
                    final arguments = call.arguments as Map<String, dynamic>;
                    final result = await _runInference(_fletPage!, arguments);
                    _fletPage!.sendMethodCall('tfliteResult', result);
                    break;
                default:
                    return Future.error('Method not found: ${call.method}');
            }
            return null; // Method handled successfully
        });
    }

    @override
    Widget build(BuildContext context) {
        return FletPageWidget(
            page: _fletPage!,
        );
    }
}
