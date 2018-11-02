


abstract class Node {
  abstract boolean isLeaf();
}

class InteriorNode extends Node {
  int attribute; // Which attribute to divide on
  double pivot; // Which value to divide on
  Node a;
  Node b;

  InteriorNode() {

  }

  boolean isLeaf() { return false; }
}

class LeafNode extends Node {
  double[] label;

  LeafNode() {

  }

  boolean isLeaf() { return true; }
}

class DecisionTree extends SupervisedLearner {
  Node root;

  DecisionTree() {

  }

  String name() { return ""; }

  Node buildTree(Matrix features, Matrix labels) {
    if(features.rows() != labels.rows()) throw new RuntimeException("Rows unequal");

    int col = pickDividingColumn(features, labels);
    double pivot = pickPivot(features, labels);

    int vals = features.valueCount(col);

    // copies the meta-data
    Matrix featA = new Matrix(features);
    Matrix featB = new Matrix(features);
    Matrix labelA = new Matrix(labels);
    Matrix labelB = new Matrix(labels);

    for(int i = 0; i < features.rows(); ++i) {

      // Data is continuous
      if(vals == 0) {

        // Decide if the data is greater than or less than pivot
        if(features.row(i)[col] < pivot) {
          featA.takeRow(features.removeRow(i));
          labelA.takeRow(labels.removeRow(i));
        } else {
          featB.takeRow(features.removeRow(i));
          labelB.takeRow(labels.removeRow(i));
        }

      // Data is non-continuous
      } else {

      }
    }
  }

  void train(Matrix features, Matrix labels) {
    root = buildTree(features, labels);
  }

  void predict(double[] in, double[] out) {

  }
}
