


abstract class Node {
  abstract boolean isLeaf();
}

class InteriorNode extends Node {
  int attribute; // Which attribute to divide on
  double pivot; // Which value to divide on
  Node a;
  Node b;

  boolean isLeaf() { return false; }
}

class LeafNode extends Node {
  double[] label;

  boolean isLeaf() { return true; } 
}

class DecisionTree extends SupervisedLearner {
  String name() {
    return "";
  }

  void train(Matrix features, Matrix labels) {

  }

  void predict(double[] in, double[] out) {

  }

  Node root;
}
