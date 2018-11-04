import java.util.Random;


abstract class Node {
  int attribute; // Which attribute to divide on
  double pivot; // Which value to divide on
  double[] label;
  Node a, b;

  abstract boolean isLeaf();
}

class InteriorNode extends Node {
  InteriorNode(Node a, Node b, int col, double pivot) {
    label = null;
    this.a = a;
    this.b = b;
    attribute = col;
    this.pivot = pivot;
  }

  boolean isLeaf() { return false; }
}

class LeafNode extends Node {

  LeafNode(Matrix labels) {
    a = null;
    b = null;

    label = new double[labels.rows()];
    double mode;
    if(labels.valueCount(0) == 0)
      mode = labels.columnMean(0);
    else
      mode = labels.mostCommonValue(0);
    Vec.setAll(label, mode);
  }

  boolean isLeaf() { return true; }
}

class DecisionTree extends SupervisedLearner {
  Random rand;
  Node root;

  DecisionTree(Random r) {
    rand = r;
  }

  String name() { return "DecisionTree"; }

  int pickDividingColumn(Matrix features, Matrix labels) {
    // try a whole bunch of divisions()
    // return best one
    // we should do entropy reduction eventually

    int col = rand.nextInt(features.cols());
    //int row = rand.nextInt(features.rows());
    //int pivot = features.row(row)[col];
    return col;
  }

  double pickPivot(Matrix features, Matrix labels, int col) {
    //int col = rand.nextInt(features.cols());
    int row = rand.nextInt(features.rows());
    double pivot = features.row(row)[col];
    return pivot;
  }

  Node buildTree(Matrix features, Matrix labels) {
    if(features.rows() != labels.rows()) throw new RuntimeException("Rows unequal");

    int col = pickDividingColumn(features, labels);
    double pivot = pickPivot(features, labels, col);

    Matrix featA = null, featB = null, labelA = null, labelB = null;
    for(int patience = 1; patience >= 0; --patience) {
      int vals = features.valueCount(col);

      // copies the meta-data
      featA = new Matrix(features);
      featB = new Matrix(features);
      labelA = new Matrix(labels);
      labelB = new Matrix(labels);

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

        // Data is non-continuous (categorical)
        } else {
          if(features.row(i)[col] == pivot) {
            featA.takeRow(features.removeRow(i));
            labelA.takeRow(labels.removeRow(i));
          } else {
            featB.takeRow(features.removeRow(i));
            labelB.takeRow(labels.removeRow(i));
          }
        }
      }

      // we did succeed at dividing
      if(featA.rows() != 0 || featB.rows() != 0) {
        // We assume that the leaf node consutctor compues the mean of all the cont. values in the labels
        // and the mode of the categorical variables
        break;
      }
    }

    System.out.println(col);

    // We failed to divide the data
    if(featA.rows() == 0 || featB.rows() == 0) {
      // We assume that the leaf node consutctor compues the mean of all the cont. values in the labels
      // and the mode of the categorical variables
      return new LeafNode(labels);
    }

    Node a = buildTree(featA, labelA);
    Node b = buildTree(featB, labelB);
    return new InteriorNode(a, b, col, pivot);
  }

  void train(Matrix features, Matrix labels) {
    root = buildTree(features, labels);
  }

  // Starts at the root, and runs down the tree until it reaches
  void predict(double[] in, double[] out) {
    Node n = root;

    while(true) {
      if(!n.isLeaf()) {
        if(in[n.attribute] < n.pivot)
          n = n.a;
        else
          n = n.b;
      } else {
        // When we hit the leaf node, copy the labels into out
        Vec.copy(out, n.label);
      }
    }
  }
}
