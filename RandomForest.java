


class RandomForest extends SupervisedLearner {
  int numTrees;

  RandomForest(int nt) {
    numTrees = nt;
  }

  String name() { return "RandomForest"; }

  // Resamples data from the original set (Bootstrap aggregation)
  void bagging(Matrix features, Matrix labels) {

  }

  void train(Matrix features, Matrix labels) {

  }

  void predict(double[] in, double[] out) {

  }
}
