import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

import java.util.Random;

public class Evaluator {

    public static void main(String[] args) throws Exception {

        // Load cleaned dataset
        DataSource source = new DataSource("datasets/heart_disease_preprocessed.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("=== FINAL MODEL COMPARISON ===\n");

        evaluateBaselineModels(data);
        evaluateImprovedModels(data);
    }

    // ---- BASELINE: J48, RF, NB ----
    public static void evaluateBaselineModels(Instances data) throws Exception {
        System.out.println("=== BASELINE MODELS ===");

        evaluateModel("J48 (Baseline)", new J48(), data);
        evaluateModel("Random Forest (Baseline)", new RandomForest(), data);
        evaluateModel("Naive Bayes (Baseline)", new NaiveBayes(), data);
    }

    // ---- IMPROVED MODELS ----
    public static void evaluateImprovedModels(Instances data) throws Exception {
        System.out.println("\n=== IMPROVED MODELS ===");

        // Improved J48
        J48 j48Imp = new J48();
        j48Imp.setUnpruned(false);
        j48Imp.setConfidenceFactor(0.1f);
        j48Imp.setMinNumObj(5);
        evaluateModel("J48 (Improved)", j48Imp, data);

        // Improved Random Forest
        RandomForest rfImp = new RandomForest();
        rfImp.setNumTrees(200);
        rfImp.setMaxDepth(20);
        rfImp.setNumFeatures(5);
        evaluateModel("Random Forest (Improved)", rfImp, data);
    }

    // Reusable evaluation function
    public static void evaluateModel(String label, weka.classifiers.Classifier model, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));

        System.out.println("\n--- " + label + " ---");
        System.out.printf("Accuracy: %.2f%%\n", eval.pctCorrect());
        System.out.println("F1-score per class:");
        System.out.println(eval.toClassDetailsString());
        System.out.println("Confusion Matrix:");
        System.out.println(eval.toMatrixString());
    }
}
