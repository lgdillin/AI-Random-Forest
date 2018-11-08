import java.util.ArrayList;
import java.util.Random;


class RandomForest extends SupervisedLearner {
  Random rand;
  int numTrees;
  double labelType;
  ArrayList<DecisionTree> trees;

  RandomForest(Random r, int nt) {
    rand = r;
    numTrees = nt;

    trees = new ArrayList<DecisionTree>();
    for(int i = 0; i < numTrees; ++i) {
      trees.add(new DecisionTree(rand));
    }
  }

  String name() { return "RandomForest"; }

  // Resamples data from the original set (Bootstrap aggregation)
  void bagging(Matrix features, Matrix labels, Matrix newFeatures, Matrix newLabels) {
    for(int i = 0; i < features.rows(); ++i) {
      int row = rand.nextInt(features.rows());

      Vec.copy(newFeatures.newRow(), features.row(row));
      Vec.copy(newLabels.newRow(), labels.row(row));
    }
  }

  void train(Matrix features, Matrix labels) {
    this.labelType = labels.valueCount(0);

    Matrix newFeatures = new Matrix();
    Matrix newLabels = new Matrix();

    for(int i = 0; i < numTrees; ++i) {
      // Re-sample data
      newFeatures.copyMetaData(features);
      newLabels.copyMetaData(labels);

      bagging(features, labels, newFeatures, newLabels);

      DecisionTree dt = trees.get(i);
      dt.train(newFeatures, newLabels);
    }
  }

  void predict(double[] in, double[] out) {
    Matrix votes = new Matrix(numTrees, 1);

    double[] dtPrediction = new double[1];
    for(int i = 0; i < numTrees; ++i) {
      DecisionTree dt = trees.get(i);
      dt.predict(in, dtPrediction);

      Vec.copy(votes.row(i), dtPrediction);
    }

    if(labelType == 0) {
      Vec.setAll(out, votes.columnMean(0));
    } else {
      Vec.setAll(out, votes.mostCommonValue(0));
    }
  }
}
