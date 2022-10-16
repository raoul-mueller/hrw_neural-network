import 'dart:math';

/// This function can be used to set the random generator's seed for
/// reproducible random numbers.
void seed(final int s) => linearRandom = Random(s).nextDouble;

/// A random number generator whose output matches a constant curve with a
/// min of 0 and a max of 0.999...
double Function() linearRandom = Random().nextDouble;

/// A random number generator whose output matches the gaussian bell curve
/// with µ=0 and σ=1.
/// https://en.wikipedia.org/wiki/Normal_distribution
///
/// Implementation based on https://stackoverflow.com/a/7771542/13878330.
double gaussianRandom() =>
    sqrt(-2 * log(linearRandom())) * cos(2 * pi * linearRandom());

/// The sigmoid function.
/// https://en.wikipedia.org/wiki/Sigmoid_function
double sigmoid(final double x) => 1 / (1 + exp(-x));

/// The median value.
double clamp(final double own, final double others, final double c) =>
    max(own, min(others, c));

/// A 0-dimensional tensor.
class Scalar {
  /// The constructor.
  Scalar({required final dynamic value}) {
    if (value is Scalar) {
      this.value = value.value;
    } else if (value is num) {
      this.value = value.toDouble();
    } else {
      throw ArgumentError('neither a double nor a scalar', 'value');
    }
  }

  /// A random scalar.
  factory Scalar.random() => Scalar(value: gaussianRandom());

  /// The scalar's value.
  late final double value;

  /// A unary operation.
  Scalar apply({required final double Function(double x) operation}) =>
      Scalar(value: operation(this.value));

  /// A binary operation.
  Scalar combine({
    required final double Function(double a, double b) operation,
    required final Scalar other,
  }) =>
      Scalar(value: operation(this.value, other.value));

  @override
  String toString() => this.value.toStringAsFixed(4).padLeft(10);
}

/// A 1-dimensional tensor.
class Vector {
  /// The constructor.
  Vector({required final dynamic values}) {
    if (values is Vector) {
      this.values = List<Scalar>.unmodifiable(values.values);
    } else if (values is Iterable) {
      this.values = List<Scalar>.unmodifiable(
        values.map((final dynamic value) => Scalar(value: value)),
      );
    } else {
      throw ArgumentError('neither a list nor a vector', 'values');
    }
    this.raw = List<double>.unmodifiable(
      this.values.map((final Scalar value) => value.value),
    );
  }

  /// A random vector.
  factory Vector.random({required final int n_values}) {
    if (n_values < 1) {
      throw ArgumentError();
    }
    return Vector(
      values: List<Scalar>.generate(n_values, (final _) => Scalar.random()),
    );
  }

  /// The vector's values (as scalars).
  late final List<Scalar> values;

  /// The vector's values (as nums).
  late final List<double> raw;

  /// The vector's size.
  int get n_values => this.values.length;

  /// Prepends a scalar or double to the vector.
  Vector prepend({required final dynamic value}) =>
      Vector(values: <Scalar>[Scalar(value: value), ...this.values]);

  /// Strips the vector's first value.
  Vector stripFirst() => Vector(values: this.values.skip(1));

  /// Dot multiplication with another vector.
  Scalar dotVector({required final Vector other}) {
    if (this.n_values != other.n_values) {
      throw ArgumentError('incompatible vector', 'other');
    }
    return Scalar(
      value: List<double>.generate(
        this.n_values,
        (final int index) => this.raw[index] * other.raw[index],
      ).reduce((final double own, final double others) => own + others),
    );
  }

  /// Dot multiplication with a matrix.
  Vector dotMatrix({required final Matrix other}) => Vector(
        values:
            other.rows.map((final Vector row) => this.dotVector(other: row)),
      );

  /// A unary operation.
  Vector apply({required final double Function(double x) operation}) => Vector(
        values: this
            .values
            .map((final Scalar value) => value.apply(operation: operation)),
      );

  /// A binary operation.
  Vector combine({
    required final double Function(double a, double b) operation,
    required final Vector other,
  }) {
    if (this.n_values != other.n_values) {
      throw ArgumentError('incompatible vector', 'other');
    }
    return Vector(
      values: List<Scalar>.generate(
        this.n_values,
        (final int index) => this
            .values[index]
            .combine(operation: operation, other: other.values[index]),
      ),
    );
  }

  @override
  String toString() =>
      this.values.map((final Scalar value) => value.toString()).join();
}

/// A 2-dimensional tensor.
class Matrix {
  /// The constructor.
  Matrix({required final dynamic rows}) {
    if (rows is Matrix) {
      this.rows = List<Vector>.unmodifiable(rows.rows);
    } else if (rows is Iterable) {
      this.rows = List<Vector>.unmodifiable(
        rows.map((final dynamic values) => Vector(values: values)),
      );
    } else {
      throw ArgumentError('neither a list nor a matrix', 'rows');
    }
    for (final Vector row in this.rows) {
      if (row.n_values != this.rows.first.n_values) {
        throw ArgumentError('incompatible rows', 'rows');
      }
    }
    this.raw = List<List<double>>.unmodifiable(
      this.rows.map((final Vector row) => row.raw),
    );
  }

  /// A random matrix.
  factory Matrix.random({
    required final int n_rows,
    required final int n_columns,
  }) {
    if (n_rows < 1) {
      throw ArgumentError();
    }
    if (n_columns < 1) {
      throw ArgumentError();
    }
    return Matrix(
      rows: List<Vector>.generate(
        n_rows,
        (final _) => Vector.random(n_values: n_columns),
      ),
    );
  }

  /// The matrix's rows (as Vectors).
  late final List<Vector> rows;

  /// The matrix's rows (as lists of nums).
  late final List<List<double>> raw;

  /// The matrix's size (regarding its rows).
  int get n_rows => this.rows.length;

  /// The matrix's size (regarding its columns).
  int get n_columns => this.rows.first.n_values;

  /// The matrix's transposed form (rows and columns swapped).
  Matrix get transposed => Matrix(
        rows: List<List<double>>.generate(
          this.n_columns,
          (final int row) => List<double>.generate(
            this.n_rows,
            (final int column) => this.raw[column][row],
          ),
        ),
      );

  /// Dot multiplication with another matrix.
  Matrix dotMatrix({required final Matrix other}) {
    if (this.n_columns != other.n_rows) {
      throw ArgumentError('incompatible matrices', 'other');
    }
    return Matrix(
      rows: this
          .rows
          .map((final Vector row) => row.dotMatrix(other: other.transposed)),
    );
  }

  /// A unary operation.
  Matrix apply({required final double Function(double x) operation}) => Matrix(
        rows: this
            .rows
            .map((final Vector row) => row.apply(operation: operation)),
      );

  /// A binary operation.
  Matrix combine({
    required final double Function(double a, double b) operation,
    required final Matrix other,
  }) {
    if (this.n_rows != other.n_rows) {
      throw ArgumentError('incompatible matrix', 'other');
    }
    return Matrix(
      rows: List<Vector>.generate(
        this.n_rows,
        (final int index) => this
            .rows[index]
            .combine(operation: operation, other: other.rows[index]),
      ),
    );
  }

  @override
  String toString() =>
      this.rows.map((final Vector row) => row.toString()).join('\n');
}
