import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

import java.util.Random;

public class Improver {

    public static void main(String[] args) throws Exception {

        // Load cleaned dataset
        DataSource source = new DataSource("datasets/heart_disease_preprocessed.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("=== MODEL IMPROVEMENTS ===");

        improveJ48(data);
        improveRandomForest(data);
    }

    // ----- IMPROVE J48 -----
    public static void improveJ48(Instances data) throws Exception {
        System.out.println("\n=== Improved J48 (Pruned) ===");

        J48 tree = new J48();
        tree.setUnpruned(false);          // enable pruning
        tree.setConfidenceFactor(0.1f);   // stronger pruning
        tree.setMinNumObj(5);             // minimum samples per leaf

        evaluate(tree, data);
    }

    // ----- IMPROVE RANDOM FOREST -----
    public static void improveRandomForest(Instances data) throws Exception {
        System.out.println("\n=== Improved Random Forest ===");

        RandomForest rf = new RandomForest();
        rf.setNumTrees(200);         // more trees
        rf.setMaxDepth(20);          // deeper trees
        rf.setNumFeatures(5);        // number of features per split

        evaluate(rf, data);
    }

    // Reusable evaluator
    public static void evaluate(weka.classifiers.Classifier model, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidat
