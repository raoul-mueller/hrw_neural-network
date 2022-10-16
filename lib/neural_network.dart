import 'math.dart';

/// A feed forward neural network.
///
/// This network consists of layers.
/// The network can be used to predict an output based on an input.
/// The network can be trained using backpropagation.
class FeedForwardNeuralNetwork {
  /// The constructor.
  FeedForwardNeuralNetwork({
    required this.n_inputs,
    required this.n_outputs,
    required final Iterable<Layer> layers,
  })  : this.layers = List<Layer>.unmodifiable(layers),
        this.n_layers = layers.length {
    final List<DynamicLayer> dynamicLayers =
        layers.whereType<DynamicLayer>().toList();
    if (dynamicLayers.isEmpty) {
      throw IllegalConfigurationException();
    }
    if (dynamicLayers.first.n_inputs != this.n_inputs) {
      throw IllegalConfigurationException();
    }
    if (dynamicLayers.last.n_outputs != this.n_outputs) {
      throw IllegalConfigurationException();
    }
    for (int i = 0; i < dynamicLayers.length - 1; i++) {
      if (dynamicLayers[i].n_outputs != dynamicLayers[i + 1].n_inputs) {
        throw IllegalConfigurationException();
      }
    }
  }

  /// The network's layers.
  final List<Layer> layers;

  /// The layer count.
  final int n_layers;

  /// The input vector's n_values.
  final int n_inputs;

  /// The output vector's n_output.
  final int n_outputs;

  /// Passes an input vector forwards through each layer.
  Stream<Vector> forward({required final Vector input}) async* {
    yield input;
    Vector activation = input;
    for (final Layer layer in this.layers) {
      activation = layer.forward(input: activation);
      yield activation;
    }
  }

  /// Passes an input vector forwards through each layer.
  /// Additionally, calculates deltas for the network's dynamic layers
  /// while passing the gradient backwards through each layer.
  Stream<BackwardSummary> backward({
    required final InputOutputPair inputOutputPair,
    required final CostFunction costFunction,
  }) async* {
    final List<Vector> activations =
        await this.forward(input: inputOutputPair.input).toList();

    BackwardSummary backwardSummary = costFunction.backward(
      actual: activations.last,
      expected: inputOutputPair.output,
    );
    yield backwardSummary;

    for (int i = this.n_layers - 1; i >= 0; i--) {
      backwardSummary = this.layers[i].backward(
            gradient: backwardSummary.gradient,
            input: activations[i],
            output: activations[i + 1],
          );
      yield backwardSummary;
    }
  }

  /// Predicts an output vector based on an input vector.
  Future<Vector> predict({required final Vector input}) =>
      this.forward(input: input).last;

  /// Validates the network's performance with test data while training the
  /// network on training data.
  /// Each of the network's dynamic layers is adjusted n_epoch times.
  Stream<TrainingSummary> train({
    required final int n_epochs,
    required final double learningRate,
    required final Iterable<InputOutputPair> trainingData,
    required final Iterable<InputOutputPair> testData,
    required final CostFunction costFunction,
  }) async* {
    List<double> costs = await Future.wait(<Future<double>>[
      this.evaluate(data: trainingData, costFunction: costFunction),
      this.evaluate(data: testData, costFunction: costFunction)
    ]);
    yield TrainingSummary(
      progress: 0,
      trainDataCost: costs[0],
      testDataCost: costs[1],
    );
    await Future<void>.delayed(Duration.zero);

    for (int epoch = 0; epoch < n_epochs; epoch++) {
      final List<List<BackwardSummary>> backwardSummaries = (await Future.wait(
        trainingData.map(
          (final InputOutputPair inputOutputPair) => this
              .backward(
                inputOutputPair: inputOutputPair,
                costFunction: costFunction,
              )
              .toList(),
        ),
      ))
          .map(
            (final List<BackwardSummary> backwardSummaries) =>
                backwardSummaries.reversed.toList(),
          )
          .toList();

      for (int i_layer = 0; i_layer < this.n_layers; i_layer++) {
        final Layer layer = this.layers[i_layer];
        final Iterable<BackwardSummary> layerBackwardSummaries =
            backwardSummaries.map(
          (final List<BackwardSummary> backwardSummaries) =>
              backwardSummaries[i_layer],
        );
        if (layer is DenseLayer) {
          final Matrix delta = layerBackwardSummaries
              .cast<DenseBackwardSummary>()
              .map(
                (final DenseBackwardSummary backwardSummary) =>
                    backwardSummary.delta,
              )
              .reduce(
                (final Matrix a, final Matrix b) => a.combine(
                  operation: (final double own, final double others) =>
                      own + others,
                  other: b,
                ),
              )
              .apply(operation: (final double x) => x / trainingData.length);
          layer.applyDelta(learningRate: learningRate, delta: delta);
        }
      }

      costs = await Future.wait(<Future<double>>[
        this.evaluate(data: trainingData, costFunction: costFunction),
        this.evaluate(data: testData, costFunction: costFunction)
      ]);
      yield TrainingSummary(
        progress: (epoch + 1) / n_epochs,
        trainDataCost: costs[0],
        testDataCost: costs[1],
      );
      await Future<void>.delayed(Duration.zero);
    }
  }

  /// Calculates the data's average cost.
  Future<double> evaluate({
    required final Iterable<InputOutputPair> data,
    required final CostFunction costFunction,
  }) async {
    final List<Vector> predictions = (await Future.wait(
      data.map(
        (final InputOutputPair inputOutputPair) =>
            this.predict(input: inputOutputPair.input),
      ),
    ))
        .toList();

    return List<Vector>.generate(
          data.length,
          (final int index) => costFunction.forward(
            actual: predictions[index],
            expected: data.toList()[index].output,
          ),
        )
            .map(
              (final Vector cost) =>
                  cost.raw.reduce(
                    (final double own, final double others) => own + others,
                  ) /
                  cost.n_values,
            )
            .reduce((final double own, final double others) => own + others) /
        data.length;
  }

  /// The neuron count in the layer with the given index.
  int n_neuronsAt(final int i_layer) {
    if (i_layer < 0) {
      return this.n_inputs;
    }
    if (i_layer >= this.n_layers) {
      return n_outputs;
    }

    final Layer layer = this.layers[i_layer];
    if (layer is DynamicLayer) {
      return layer.n_outputs;
    }
    final Iterable<DynamicLayer> dynamicLayersAfter =
        this.layers.skip(i_layer).whereType<DynamicLayer>();
    return dynamicLayersAfter.isEmpty
        ? this.n_outputs
        : dynamicLayersAfter.first.n_inputs;
  }
}

/// A layer is part of a feed forward neural network.
abstract class Layer {
  /// Passes an input vector forwards through the layer.
  Vector forward({required final Vector input});

  /// The layer type.
  String get type;

  /// Calculates the gradient.
  BackwardSummary backward({
    required final Vector input,
    required final Vector output,
    required final Vector gradient,
  });
}

/// A layer that can be trained.
abstract class DynamicLayer extends Layer {
  /// The constructor.
  DynamicLayer({required this.n_inputs, required this.n_outputs});

  /// The input vector's n_values.
  final int n_inputs;

  /// The output vector's n_values.
  final int n_outputs;
}

/// A dense layer (aka fully connected layer)
class DenseLayer extends DynamicLayer {
  /// The constructor.
  DenseLayer({required this.weights})
      : super(n_inputs: weights.n_columns - 1, n_outputs: weights.n_rows);

  /// A dense layer with random weights and biases.
  DenseLayer.random({required super.n_inputs, required super.n_outputs})
      : weights = Matrix.random(n_rows: n_outputs, n_columns: n_inputs + 1);

  @override
  String get type => 'Dense';

  /// The weights and biases (biases are in first column).
  Matrix weights;

  @override
  Vector forward({required final Vector input}) =>
      input.prepend(value: 1).dotMatrix(other: this.weights);

  @override
  DenseBackwardSummary backward({
    required final Vector input,
    required final Vector output,
    required final Vector gradient,
  }) =>
      DenseBackwardSummary(
        gradient:
            gradient.dotMatrix(other: this.weights.transposed).stripFirst(),
        delta: Matrix(rows: <Vector>[gradient])
            .transposed
            .dotMatrix(other: Matrix(rows: <Vector>[input.prepend(value: 1)])),
      );

  /// Applies a delta with a learning rate.
  void applyDelta({
    required final double learningRate,
    required final Matrix delta,
  }) =>
      this.weights = this.weights.combine(
            operation: (final double own, final double others) =>
                own - learningRate * others,
            other: delta,
          );
}

/// A non-trainable layer.
abstract class StaticLayer extends Layer {}

/// A sigmoid activation layer.
class SigmoidLayer extends StaticLayer {
  @override
  String get type => 'Sigmoid';

  @override
  Vector forward({required final Vector input}) =>
      input.apply(operation: sigmoid);

  @override
  BackwardSummary backward({
    required final Vector input,
    required final Vector output,
    required final Vector gradient,
  }) =>
      BackwardSummary(
        gradient:
            output.apply(operation: (final double x) => x * (1 - x)).combine(
                  operation: (final double own, final double others) =>
                      own * others,
                  other: gradient,
                ),
      );
}

/// A rectified linear unit activation layer.
class ReluLayer extends StaticLayer {
  @override
  String get type => 'ReLU';

  @override
  Vector forward({required final Vector input}) =>
      input.apply(operation: (final double x) => x > 0 ? x : 0);

  @override
  BackwardSummary backward({
    required final Vector input,
    required final Vector output,
    required final Vector gradient,
  }) =>
      BackwardSummary(
        gradient: input.combine(
          operation: (final double own, final double others) =>
              own > 0 ? others : 0,
          other: gradient,
        ),
      );
}

/// A cost function.
abstract class CostFunction {
  /// Calculates the cost vector based on the actual and the expected output.
  Vector forward({
    required final Vector actual,
    required final Vector expected,
  });

  /// Calculates the gradient based on the actual and the expected output.
  BackwardSummary backward({
    required final Vector actual,
    required final Vector expected,
  });
}

/// A distance cost function.
class DistanceCostFunction extends CostFunction {
  @override
  Vector forward({
    required final Vector actual,
    required final Vector expected,
  }) =>
      expected
          .combine(
            operation: (final double own, final double others) => own - others,
            other: actual,
          )
          .apply(operation: (final double x) => x < 0 ? -x : x);

  @override
  BackwardSummary backward({
    required final Vector actual,
    required final Vector expected,
  }) =>
      BackwardSummary(
        gradient: actual
            .combine(
              operation: (final double own, final double others) =>
                  own - others,
              other: expected,
            )
            .combine(
              operation: (final double own, final double others) =>
                  own == 0 ? 0 : own / own.abs() * others,
              other: actual,
            ),
      );
}

/// A squared distance cost function.
class SquaredDistanceCostFunction extends CostFunction {
  @override
  Vector forward({
    required final Vector actual,
    required final Vector expected,
  }) =>
      expected
          .combine(
            operation: (final double own, final double others) => own - others,
            other: actual,
          )
          .apply(operation: (final double x) => x * x);

  @override
  BackwardSummary backward({
    required final Vector actual,
    required final Vector expected,
  }) =>
      BackwardSummary(
        gradient: actual
            .combine(
              operation: (final double own, final double others) =>
                  own - others,
              other: expected,
            )
            .apply(operation: (final double x) => 2 * x),
      );
}

/// An input and an output.
class InputOutputPair {
  /// The constructor.
  InputOutputPair({required this.input, required this.output});

  /// The input vector.
  final Vector input;

  /// The output vector.
  final Vector output;
}

/// A training summary is yielded during training as feedback that the train
/// process is still running.
class TrainingSummary {
  /// The constructor.
  TrainingSummary({
    required this.progress,
    required this.trainDataCost,
    required this.testDataCost,
  });

  /// The progress (current epoch divided by total epochs).
  final double progress;

  /// The train data's cost.
  final double trainDataCost;

  /// The test data's cost.
  final double testDataCost;

  @override
  String toString() {
    final String progress = (this.progress * 100).toStringAsFixed(2).padLeft(9);
    final String trainingCost =
        this.trainDataCost.toStringAsFixed(4).padLeft(10);
    final String testCost = this.testDataCost.toStringAsFixed(4).padLeft(10);
    return 'progress: $progress%'
        ' | training cost: $trainingCost'
        ' | test cost: $testCost';
  }
}

/// A backward summary contains at least the layer's gradient.
class BackwardSummary {
  /// The constructor.
  BackwardSummary({required this.gradient});

  /// The layer's gradient.
  final Vector gradient;
}

/// A backward summary for a dense layer.
class DenseBackwardSummary extends BackwardSummary {
  /// The constructor.
  DenseBackwardSummary({required super.gradient, required this.delta});

  /// The layer's data to improve it.
  final Matrix delta;
}

/// Thrown if the configured network is illegal (mainly if vector sizes do
/// not match).
class IllegalConfigurationException implements Exception {}
