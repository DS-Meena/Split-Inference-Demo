import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flet/flet.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:provider/provider.dart'; // Import Provider

// Global interpreter instance
Interpreter? _interpreter;
bool _modelLoaded = false;
final MethodChannel _platformChannel = const MethodChannel('com.example.gpt2tflite/tflite'); // Match Python's channel name

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // FletApp needs access to the FletPage context,
  // so we wrap it with a Provider.
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
      _interpreter!.close(); // Close existing interpreter if any
    }
    // Load model from assets
    _interpreter = await Interpreter.fromAsset('gpt2.tflite');
    _modelLoaded = true;
    print("TFLite Model loaded successfully in Dart.");
    // Communicate model loading status to Python
    page.sendMethodCall('modelLoaded', {'success': true, 'message': "TFLite model loaded successfully!"});
    return true;
  } catch (e) {
    _modelLoaded = false;
    print("Error loading TFLite model: $e");
    // Communicate model loading failure to Python
    page.sendMethodCall('modelLoaded', {'success': false, 'error': e.toString(), 'message': "Error loading TFLite model: $e"});
    return false;
  }
}

Future<Map<String, dynamic>> _runInference(FletPage page, Map<String, dynamic> arguments) async {
    if (!_modelLoaded || _interpreter == null) {
        page.sendMethodCall('tfliteResult', {'success': false, 'error': 'TFLite model not loaded.'});
        return {'success': false, 'error': 'TFLite model not loaded.'};
    }

    try {
        final inputIdsBytes = arguments['input_ids_bytes'] as List<dynamic>; // Cast to List<dynamic> then to Uint8List
        final inputShape = List<int>.from(arguments['input_shape']);
        final outputShape = List<int>.from(arguments['output_shape']);

        // Allocate input buffer
        final inputBuffer = Uint8List.fromList(inputIdsBytes); // Direct conversion

        // Prepare output buffer. Adjust the output type based on your model's specific output.
        // GPT-2 typically outputs float32 logits.
        final outputBuffer = Float32List(outputShape.reduce((a, b) => a * b)).buffer;

        // Resize input tensors
        _interpreter!.resizeInput(0, inputShape); // Resize the first (and likely only) input tensor
        _interpreter!.allocateTensors(); // Allocate tensors after resizing

        // Run inference
        _interpreter!.run(inputBuffer, outputBuffer);

        // Convert output to a Python-compatible format (e.g., list of floats)
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
    FletPage? _fletPage; // Store the FletPage instance

    @override
    void initState() {
        super.initState();
        // After the first frame is rendered, handle Python requests.
        WidgetsBinding.instance.addPostFrameCallback((_) async {
            _fletPage = Provider.of<FletPage>(context, listen: false);
            _handlePythonRequests();
        });
    }

    Future<void> _handlePythonRequests() async {
        // This loop constantly checks Python flags
        while(true) {
            if (_fletPage != null && _fletPage!.platform_data != null) {
                final Map<String, dynamic>? pyFlags = _fletPage!.platform_data!['py_flags'] as Map<String, dynamic>?;

                if (pyFlags != null) {
                    // Handle LoadModel request
                    if (pyFlags['load_model_request'] == true) {
                        print("Dart: Received loadModel request from Python.");
                        await _loadModel(_fletPage!); // Pass the FletPage instance
                        pyFlags['load_model_request'] = false; // Reset the flag
                        _fletPage!.update(); // Notify Python that the flag is reset
                    }

                    // Handle run inference request
                    if (pyFlags['run_inference_request'] == true) {
                        print("Dart: Received runInference request from Python.");
                        final arguments = _fletPage!.platform_data!['inference_input_data'] as Map<String, dynamic>?;

                        if (arguments != null) {
                            final result = await _runInference(_fletPage!, arguments); // Pass FletPage and arguments
                            _fletPage!.sendMethodCall('tfliteResult', result);
                        } else {
                            _fletPage!.sendMethodCall('tfliteResult', {'success': false, 'error': 'No inference data provided.'});
                        }
                        pyFlags['run_inference_request'] = false;
                        _fletPage!.platform_data!['inference_input_data'] = null; // Clear data
                        _fletPage!.update(); // Notify Python that flags are reset
                    }
                }
            }
            await Future.delayed(const Duration(milliseconds: 100));
        }
    }

    @override
    Widget build(BuildContext context) {
        return FletPageWidget(
            page: _fletPage!,
        );
    }
}
