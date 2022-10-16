import 'dart:async';
import 'dart:html';
import 'dart:js';
import 'dart:math';

import 'package:neural_network/math.dart';
import 'package:neural_network/neural_network.dart';

extension on Window {
  dynamic prompt(final String message) =>
      context.callMethod('prompt', <String>[message]);
}

List<Layer>? blueprint = <Layer>[];
FeedForwardNeuralNetwork? network;
CostFunction? costFunction;

void main() {
  document
      .querySelector('#example-blueprint-xor-button')!
      .addEventListener('click', onExampleBlueprintXorButtonClicked);
  document
      .querySelector('#example-blueprint-sinus-button')!
      .addEventListener('click', onExampleBlueprintSinusButtonClicked);
  document
      .querySelector('#reset-blueprint-button')!
      .addEventListener('click', onResetBlueprintButtonClicked);
  document
      .querySelector('#add-dense-layer-button')!
      .addEventListener('click', onAddDenseLayerButtonClicked);
  document
      .querySelector('#add-sigmoid-layer-button')!
      .addEventListener('click', onAddSigmoidLayerButtonClicked);
  document
      .querySelector('#add-relu-layer-button')!
      .addEventListener('click', onAddReluLayerButtonClicked);
  document
      .querySelector('#finalize-blueprint-button')!
      .addEventListener('click', onFinalizeBlueprintButtonClicked);
  document
      .querySelector('#reset-network-button')!
      .addEventListener('click', onResetNetworkButtonClicked);
  document
      .querySelector('#example-training-xor-button')!
      .addEventListener('click', onExampleTrainingXorButtonClicked);
  document
      .querySelector('#example-training-sinus-button')!
      .addEventListener('click', onExampleTrainingSinusButtonClicked);
  document
      .querySelector('#distance-cost-function')!
      .addEventListener('change', onDistanceCostFunctionChanged);
  document
      .querySelector('#squared-distance-cost-function')!
      .addEventListener('change', onSquaredDistanceCostFunctionChanged);
  document
      .querySelector('#train-network-button')!
      .addEventListener('click', onTrainNetworkButtonClicked);
  document
      .querySelector('#predict-button')!
      .addEventListener('click', onPredictButtonClicked);
}

void showBlueprint() {
  blueprint == null
      ? document.querySelector('#blueprint')!.classes.add('d-none')
      : document.querySelector('#blueprint')!.classes.remove('d-none');
  if (blueprint == null) {
    document.querySelector('#layers')!.children = <Element>[];
    return;
  }

  document.querySelector('#layers')!.children = blueprint!
      .map(
        (final Layer layer) => Element.div()
          ..setAttribute(
            'class',
            'card d-flex flex-column align-items-center justify-content-center',
          )
          ..setAttribute('style', 'width: 200px')
          ..children = <Element>[
            Element.p()
              ..setAttribute('class', 'fw-bold m-0')
              ..text = layer.type,
            if (layer is DynamicLayer)
              Element.p()
                ..setAttribute('class', 'text-center')
                ..setAttribute(
                  'style',
                  'white-space: pre;',
                )
                ..text = '${layer.n_inputs} inputs\n${layer.n_outputs} outputs',
          ],
      )
      .toList();
}

void showNetwork() {
  network == null
      ? document.querySelector('#network')!.classes.add('d-none')
      : document.querySelector('#network')!.classes.remove('d-none');
  showNetworkStructure();
  showNetworkWeights();
}

void showNetworkStructure() {
  final CanvasElement canvas =
      document.querySelector('#canvas')! as CanvasElement;
  final CanvasRenderingContext2D context =
      canvas.getContext('2d')! as CanvasRenderingContext2D;

  final int canvasWidth = canvas.width!;
  final int canvasHeight = canvas.height!;
  context.clearRect(
    -canvasWidth,
    -canvasHeight,
    2 * canvasWidth,
    2 * canvasHeight,
  );

  if (network == null) {
    return;
  }
  (document.querySelector('#layer-labels')! as DivElement).children = <Element>[
    Element.div()
      ..setAttribute('class', 'flex-grow-1 text-center')
      ..text = 'Eingänge',
    for (final Layer layer in network!.layers)
      Element.div()
        ..setAttribute('class', 'flex-grow-1 text-center fw-bold')
        ..text = layer.type,
    Element.div()
      ..setAttribute('class', 'flex-grow-1 text-center')
      ..text = 'Ausgänge',
  ];

  context.translate(canvasWidth / 2, canvasHeight / 2);

  for (int i_layer = 0; i_layer <= network!.n_layers; i_layer++) {
    final Layer? layer =
        i_layer == network!.n_layers ? null : network!.layers[i_layer];
    final int n_starts = network!.n_neuronsAt(i_layer - 1);
    final int n_ends = network!.n_neuronsAt(i_layer);

    final double x_start = x_layer(canvasWidth, network!.n_layers, i_layer - 1);
    final double x_end = x_layer(canvasWidth, network!.n_layers, i_layer);

    for (int i_end = 0; i_end < n_ends; i_end++) {
      final double y_end = y_neuron(canvasHeight, n_ends, i_end);
      if (layer is DenseLayer) {
        for (int i_start = 0; i_start < n_starts; i_start++) {
          final double y_start = y_neuron(canvasHeight, n_starts, i_start);
          final double weight = layer.weights.raw[i_end][i_start + 1];
          context
            ..strokeStyle = scaledColor(sigmoid(weight))
            ..lineWidth = scaledWidth(weight)
            ..beginPath()
            ..moveTo(x_start, y_start)
            ..lineTo(x_end, y_end)
            ..stroke();
        }
      } else {
        final double y_start = y_end;
        context
          ..strokeStyle = 'black'
          ..lineWidth = 5
          ..beginPath()
          ..moveTo(x_start, y_start)
          ..lineTo(x_end, y_end)
          ..stroke();
      }
    }
  }

  for (int i_layer = 0; i_layer < network!.n_layers; i_layer++) {
    final Layer layer = network!.layers[i_layer];
    final double x = x_layer(canvasWidth, network!.n_layers, i_layer);
    final int n_neurons = network!.n_neuronsAt(i_layer);
    for (int i_neuron = 0; i_neuron < n_neurons; i_neuron++) {
      final double y = y_neuron(canvasHeight, n_neurons, i_neuron);
      if (layer is DenseLayer) {
        final double weight = layer.weights.raw[i_neuron][0];
        context
          ..fillStyle = 'white'
          ..strokeStyle = scaledColor(sigmoid(weight))
          ..lineWidth = scaledWidth(weight)
          ..beginPath()
          ..ellipse(x, y, 50, 50, 0, 0, 360, false)
          ..fill()
          ..stroke();
      } else {
        context
          ..fillStyle = 'white'
          ..strokeStyle = 'black'
          ..lineWidth = 5
          ..beginPath()
          ..ellipse(x, y, 50, 50, 0, 0, 360, false)
          ..fill()
          ..stroke();
      }
    }
  }

  for (int i_input = 0; i_input < network!.n_inputs; i_input++) {
    final double x = x_layer(canvasWidth, network!.n_layers, -1);
    final double y = y_neuron(canvasHeight, network!.n_inputs, i_input);
    context
      ..fillStyle = 'black'
      ..beginPath()
      ..ellipse(x, y, 25, 25, 0, 0, 360, false)
      ..fill();
  }
  for (int i_output = 0; i_output < network!.n_outputs; i_output++) {
    final double x = x_layer(canvasWidth, network!.n_layers, network!.n_layers);
    final double y = y_neuron(canvasHeight, network!.n_outputs, i_output);
    context
      ..fillStyle = 'black'
      ..beginPath()
      ..ellipse(x, y, 25, 25, 0, 0, 360, false)
      ..fill();
  }
  context.translate(-canvasWidth / 2, -canvasHeight / 2);
}

void showNetworkWeights() {
  if (network == null) {
    return;
  }

  document.querySelector('#weights')!.children = network!.layers
      .whereType<DenseLayer>()
      .map(
        (final DenseLayer layer) => Element.li()
          ..setAttribute('class', 'mb-3')
          ..setAttribute(
            'style',
            'white-space: pre-wrap; font-family: monospace;',
          )
          ..text = layer.weights.toString(),
      )
      .toList();
}

void onExampleBlueprintXorButtonClicked(final Event event) {
  blueprint!
    ..clear()
    ..add(DenseLayer.random(n_inputs: 2, n_outputs: 2))
    ..add(SigmoidLayer())
    ..add(DenseLayer.random(n_inputs: 2, n_outputs: 1));
  showBlueprint();
}

void onExampleBlueprintSinusButtonClicked(final Event event) {
  blueprint!
    ..clear()
    ..add(DenseLayer.random(n_inputs: 1, n_outputs: 6))
    ..add(SigmoidLayer())
    ..add(DenseLayer.random(n_inputs: 6, n_outputs: 1));
  showBlueprint();
}

void onResetBlueprintButtonClicked(final Event event) {
  blueprint!.clear();
  showBlueprint();
}

void onAddDenseLayerButtonClicked(final Event event) {
  final int n_inputs =
      int.tryParse(window.prompt('Anzahl Eingänge').toString()) ?? 0;
  final int n_outputs =
      int.tryParse(window.prompt('Anzahl Ausgänge').toString()) ?? 0;

  if (n_inputs < 1 || n_outputs < 1) {
    return;
  }
  blueprint!.add(DenseLayer.random(n_inputs: n_inputs, n_outputs: n_outputs));
  showBlueprint();
}

void onAddSigmoidLayerButtonClicked(final Event event) {
  blueprint!.add(SigmoidLayer());
  showBlueprint();
}

void onAddReluLayerButtonClicked(final Event event) {
  blueprint!.add(ReluLayer());
  showBlueprint();
}

void onFinalizeBlueprintButtonClicked(final Event event) {
  network = FeedForwardNeuralNetwork(
    n_inputs: blueprint!.whereType<DynamicLayer>().first.n_inputs,
    n_outputs: blueprint!.reversed.whereType<DynamicLayer>().first.n_outputs,
    layers: blueprint!,
  );
  blueprint = null;
  showBlueprint();
  showNetwork();
}

void onResetNetworkButtonClicked(final Event event) {
  (document.querySelector('#train-epochs')! as InputElement).value = '';
  (document.querySelector('#train-learning-rate')! as InputElement).value = '';
  (document.querySelector('#distance-cost-function')! as InputElement).checked =
      false;
  (document.querySelector('#squared-distance-cost-function')! as InputElement)
      .checked = false;
  (document.querySelector('#train-inputs')! as TextAreaElement).value = '';
  (document.querySelector('#train-outputs')! as TextAreaElement).value = '';
  (document.querySelector('#test-inputs')! as TextAreaElement).value = '';
  (document.querySelector('#test-outputs')! as TextAreaElement).value = '';
  (document.querySelector('#predict-inputs')! as TextAreaElement).value = '';
  (document.querySelector('#predict-outputs')! as TextAreaElement).value = '';
  (document.querySelector('#train-output')! as SpanElement).text = '';
  (document.querySelector('#train-progress')! as DivElement).setAttribute(
    'style',
    'width: 0%; transition: none;',
  );
  network = null;
  costFunction = null;
  showNetwork();
  blueprint = <Layer>[];
  showBlueprint();
}

void onExampleTrainingXorButtonClicked(final Event event) {
  (document.querySelector('#train-epochs')! as InputElement).value = '1000';
  (document.querySelector('#train-learning-rate')! as InputElement).value =
      '0.1';
  (document.querySelector('#squared-distance-cost-function')! as InputElement)
      .checked = true;
  costFunction = SquaredDistanceCostFunction();
  (document.querySelector('#train-inputs')! as TextAreaElement).value =
      <Vector>[
    Vector(values: <double>[0, 0]),
    Vector(values: <double>[0, 1]),
    Vector(values: <double>[1, 0]),
    Vector(values: <double>[1, 1]),
  ].map((final Vector input) => input.toString()).join('\n');
  (document.querySelector('#train-outputs')! as TextAreaElement).value =
      <Vector>[
    Vector(values: <double>[0]),
    Vector(values: <double>[1]),
    Vector(values: <double>[1]),
    Vector(values: <double>[0]),
  ].map((final Vector input) => input.toString()).join('\n');
  (document.querySelector('#test-inputs')! as TextAreaElement).value = <Vector>[
    Vector(values: <double>[0, 0]),
    Vector(values: <double>[0, 1]),
    Vector(values: <double>[1, 0]),
    Vector(values: <double>[1, 1]),
  ].map((final Vector input) => input.toString()).join('\n');
  (document.querySelector('#test-outputs')! as TextAreaElement).value =
      <Vector>[
    Vector(values: <double>[0]),
    Vector(values: <double>[1]),
    Vector(values: <double>[1]),
    Vector(values: <double>[0]),
  ].map((final Vector input) => input.toString()).join('\n');
}

void onExampleTrainingSinusButtonClicked(final Event event) {
  (document.querySelector('#train-epochs')! as InputElement).value = '1000';
  (document.querySelector('#train-learning-rate')! as InputElement).value =
      '0.1';
  (document.querySelector('#squared-distance-cost-function')! as InputElement)
      .checked = true;
  costFunction = SquaredDistanceCostFunction();
  final List<double> trainX =
      List<double>.generate(100, (final int index) => 7 * linearRandom());
  (document.querySelector('#train-inputs')! as TextAreaElement).value = trainX
      .map((final double x) => x.toStringAsFixed(4).padLeft(10))
      .join('\n');
  (document.querySelector('#train-outputs')! as TextAreaElement).value = trainX
      .map(sin)
      .map((final double y) => y.toStringAsFixed(4).padLeft(10))
      .join('\n');
  final List<double> testX =
      List<double>.generate(10, (final int index) => 7 * linearRandom());
  (document.querySelector('#test-inputs')! as TextAreaElement).value = testX
      .map((final double x) => x.toStringAsFixed(4).padLeft(10))
      .join('\n');
  (document.querySelector('#test-outputs')! as TextAreaElement).value = testX
      .map(sin)
      .map((final double y) => y.toStringAsFixed(4).padLeft(10))
      .join('\n');
}

void onDistanceCostFunctionChanged(final Event event) =>
    costFunction = DistanceCostFunction();

void onSquaredDistanceCostFunctionChanged(final Event event) =>
    costFunction = SquaredDistanceCostFunction();

void onTrainNetworkButtonClicked(final Event event) {
  (document.querySelector('#train-network-button')! as ButtonElement)
      .setAttribute('disabled', true);

  final int? n_epochs = int.tryParse(
    (document.querySelector('#train-epochs')! as InputElement).value.toString(),
  );
  final double? learningRate = double.tryParse(
    (document.querySelector('#train-learning-rate')! as InputElement)
        .value
        .toString(),
  );
  final List<String> trainInputs =
      (document.querySelector('#train-inputs')! as TextAreaElement)
          .value
          .toString()
          .split('\n')
          .map((final String input) => input.trim())
          .where((final String input) => input.isNotEmpty)
          .toList();
  final List<String> trainOutputs =
      (document.querySelector('#train-outputs')! as TextAreaElement)
          .value
          .toString()
          .split('\n')
          .map((final String input) => input.trim())
          .where((final String input) => input.isNotEmpty)
          .toList();
  final List<String> testInputs =
      (document.querySelector('#test-inputs')! as TextAreaElement)
          .value
          .toString()
          .split('\n')
          .map((final String input) => input.trim())
          .where((final String input) => input.isNotEmpty)
          .toList();
  final List<String> testOutputs =
      (document.querySelector('#test-outputs')! as TextAreaElement)
          .value
          .toString()
          .split('\n')
          .map((final String input) => input.trim())
          .where((final String input) => input.isNotEmpty)
          .toList();

  n_epochs == null
      ? document.querySelector('#train-epochs')!.classes.add('error')
      : document.querySelector('#train-epochs')!.classes.remove('error');
  learningRate == null
      ? document.querySelector('#train-learning-rate')!.classes.add('error')
      : document.querySelector('#train-learning-rate')!.classes.remove('error');
  trainInputs.isEmpty || trainInputs.length != trainOutputs.length
      ? document.querySelector('#train-inputs')!.classes.add('error')
      : document.querySelector('#train-inputs')!.classes.remove('error');
  trainOutputs.isEmpty || trainInputs.length != trainOutputs.length
      ? document.querySelector('#train-outputs')!.classes.add('error')
      : document.querySelector('#train-outputs')!.classes.remove('error');
  testInputs.isEmpty || testInputs.length != testOutputs.length
      ? document.querySelector('#test-inputs')!.classes.add('error')
      : document.querySelector('#test-inputs')!.classes.remove('error');
  testOutputs.isEmpty || testInputs.length != testOutputs.length
      ? document.querySelector('#test-outputs')!.classes.add('error')
      : document.querySelector('#test-outputs')!.classes.remove('error');
  costFunction == null
      ? document
          .querySelector('#squared-distance-cost-function')!
          .nextElementSibling!
          .classes
          .add('error')
      : document
          .querySelector('#squared-distance-cost-function')!
          .nextElementSibling!
          .classes
          .remove('error');

  if (<String>{
    ...document.querySelector('#train-epochs')!.classes,
    ...document.querySelector('#train-learning-rate')!.classes,
    ...document.querySelector('#train-inputs')!.classes,
    ...document.querySelector('#train-outputs')!.classes,
    ...document.querySelector('#test-inputs')!.classes,
    ...document.querySelector('#test-outputs')!.classes,
    ...document
        .querySelector('#squared-distance-cost-function')!
        .nextElementSibling!
        .classes,
  }.contains('error')) {
    (document.querySelector('#train-network-button')! as ButtonElement)
        .removeAttribute('disabled');
    return;
  }

  late final List<InputOutputPair> trainingData;
  try {
    trainingData = List<InputOutputPair>.generate(
      trainInputs.length,
      (final int index) => InputOutputPair(
        input: Vector(
          values: trainInputs[index]
              .split(' ')
              .where((final String value) => value.isNotEmpty)
              .map(double.tryParse),
        ),
        output: Vector(
          values: trainOutputs[index]
              .split(' ')
              .where((final String value) => value.isNotEmpty)
              .map(double.tryParse),
        ),
      ),
    );
    // ignore: avoid_catching_errors
  } on Error {
    document.querySelector('#train-inputs')!.classes.add('error');
    document.querySelector('#train-outputs')!.classes.add('error');
    (document.querySelector('#train-network-button')! as ButtonElement)
        .removeAttribute('disabled');
    return;
  }

  late final List<InputOutputPair> testData;
  try {
    testData = List<InputOutputPair>.generate(
      testInputs.length,
      (final int index) => InputOutputPair(
        input: Vector(
          values: testInputs[index]
              .split(' ')
              .where((final String value) => value.isNotEmpty)
              .map(double.tryParse),
        ),
        output: Vector(
          values: testOutputs[index]
              .split(' ')
              .where((final String value) => value.isNotEmpty)
              .map(double.tryParse),
        ),
      ),
    );
    // ignore: avoid_catching_errors
  } on Error {
    (document.querySelector('#train-network-button')! as ButtonElement)
        .removeAttribute('disabled');
    document.querySelector('#test-inputs')!.classes.add('error');
    document.querySelector('#test-outputs')!.classes.add('error');
    return;
  }

  trainingData.any(
    (final InputOutputPair inputOutputPair) =>
        inputOutputPair.input.n_values != network!.n_inputs,
  )
      ? document.querySelector('#train-inputs')!.classes.add('error')
      : document.querySelector('#train-inputs')!.classes.remove('error');
  trainingData.any(
    (final InputOutputPair inputOutputPair) =>
        inputOutputPair.output.n_values != network!.n_outputs,
  )
      ? document.querySelector('#train-outputs')!.classes.add('error')
      : document.querySelector('#train-outputs')!.classes.remove('error');
  testData.any(
    (final InputOutputPair inputOutputPair) =>
        inputOutputPair.input.n_values != network!.n_inputs,
  )
      ? document.querySelector('#test-inputs')!.classes.add('error')
      : document.querySelector('#test-inputs')!.classes.remove('error');
  testData.any(
    (final InputOutputPair inputOutputPair) =>
        inputOutputPair.output.n_values != network!.n_outputs,
  )
      ? document.querySelector('#test-outputs')!.classes.add('error')
      : document.querySelector('#test-outputs')!.classes.remove('error');

  if (<String>{
    ...document.querySelector('#train-inputs')!.classes,
    ...document.querySelector('#train-outputs')!.classes,
    ...document.querySelector('#test-inputs')!.classes,
    ...document.querySelector('#test-outputs')!.classes,
  }.contains('error')) {
    (document.querySelector('#train-network-button')! as ButtonElement)
        .removeAttribute('disabled');
    return;
  }

  (document.querySelector('#train-inputs')! as TextAreaElement).value =
      trainingData
          .map((final InputOutputPair inputOutputPair) => inputOutputPair.input)
          .join('\n');
  (document.querySelector('#train-outputs')! as TextAreaElement).value =
      trainingData
          .map(
            (final InputOutputPair inputOutputPair) => inputOutputPair.output,
          )
          .join('\n');
  (document.querySelector('#test-inputs')! as TextAreaElement).value = testData
      .map((final InputOutputPair inputOutputPair) => inputOutputPair.input)
      .join('\n');
  (document.querySelector('#test-outputs')! as TextAreaElement).value = testData
      .map((final InputOutputPair inputOutputPair) => inputOutputPair.output)
      .join('\n');

  network!
      .train(
    n_epochs: n_epochs!,
    learningRate: learningRate!,
    trainingData: trainingData,
    testData: testData,
    costFunction: costFunction!,
  )
      .listen((final TrainingSummary summary) {
    (document.querySelector('#train-output')! as SpanElement).text =
        summary.toString();
    (document.querySelector('#train-progress')! as DivElement).setAttribute(
      'style',
      'width: ${summary.progress * 100}%; transition: none;',
    );
    showNetwork();
    if (summary.progress == 1) {
      (document.querySelector('#train-network-button')! as ButtonElement)
          .removeAttribute('disabled');
    }
  });
}

void onPredictButtonClicked(final Event event) {
  (document.querySelector('#predict-button')! as ButtonElement)
      .setAttribute('disabled', true);

  final List<Vector> inputs =
      (document.querySelector('#predict-inputs')! as TextAreaElement)
          .value
          .toString()
          .split('\n')
          .map((final String input) => input.trim())
          .where((final String input) => input.isNotEmpty)
          .map(
            (final String input) => Vector(
              values: input
                  .split(' ')
                  .where((final String value) => value.isNotEmpty)
                  .map(double.tryParse),
            ),
          )
          .toList();
  inputs.isEmpty
      ? document.querySelector('#predict-inputs')!.classes.add('error')
      : document.querySelector('#predict-inputs')!.classes.remove('error');

  if (document.querySelector('#predict-inputs')!.classes.contains('error')) {
    (document.querySelector('#predict-button')! as ButtonElement)
        .removeAttribute('disabled');
    return;
  }

  (document.querySelector('#predict-inputs')! as TextAreaElement).value =
      inputs.join('\n');

  unawaited(
    Future.wait(
      inputs.map((final Vector input) => network!.predict(input: input)),
    ).then(
      (final List<Vector> outputs) {
        (document.querySelector('#predict-button')! as ButtonElement)
            .removeAttribute('disabled');
        (document.querySelector('#predict-outputs')! as TextAreaElement).value =
            outputs.join('\n');
      },
    ),
  );
}

double x_layer(final int canvasWidth, final int n_layers, final int i_layer) =>
    canvasWidth * (i_layer - (n_layers - 1) / 2) / (n_layers + 2);

double y_neuron(
  final int canvasHeight,
  final int n_neurons,
  final int i_neuron,
) =>
    canvasHeight * (i_neuron - (n_neurons - 1) / 2) / n_neurons;

String color(final double r, final double g, final double b) => '#'
    '${(r * 255).round().toRadixString(16).padLeft(2, '0')}'
    '${(g * 255).round().toRadixString(16).padLeft(2, '0')}'
    '${(b * 255).round().toRadixString(16).padLeft(2, '0')}';

String scaledColor(final num x) => color(
      sqrt(clamp(0, -1.75 * x + 1, 1)),
      0.25,
      sqrt(clamp(0, 1.75 * x - 0.75, 1)),
    );

double scaledWidth(final num x) => 20 * sqrt((x - 0.5).abs());
