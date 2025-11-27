import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.CostMatrix;

import java.util.Random;

public class Improver {

    public static void main(String[] args) throws Exception {

        System.out.println("======================================================================");
        System.out.println("=== HEART DISEASE - IMPROVEMENT EXPERIMENTS (STEP 3 - PHAM HUYNH DUC) ===");
        System.out.println("======================================================================");

        String arffPath = (args.length > 0)
                ? args[0]
                : "dataset/heart_disease_preprocessed.arff"; // đúng với thư mục của bạn

        // 1. Load dataset
        DataSource source = new DataSource(arffPath);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        System.out.println("Loaded dataset: " + data.relationName());
        System.out.println("Instances: " + data.numInstances());
        System.out.println("Attributes: " + data.numAttributes());
        System.out.println("Class: " + data.classAttribute().name());

        // 2. Build improved model: Cost-Sensitive RandomForest
        System.out.println("\n[Step] Building Cost-Sensitive RandomForest...");
        Classifier csRandomForest = buildCostSensitiveRandomForest(data);

        // 3. Evaluate with 10-fold CV
        System.out.println("[Step] Evaluating model with 10-fold cross-validation...");
        evaluateModel(csRandomForest, data, "CostSensitive RandomForest");

        System.out.println("\n======================================================================");
        System.out.println("=== END OF IMPROVEMENT EXPERIMENTS ===");
        System.out.println("======================================================================");
    }

    private static Classifier buildCostSensitiveRandomForest(Instances train) throws Exception {
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);   // ít cây hơn cho nhanh
        rf.setMaxDepth(0);

        // cost matrix: rows = actual, cols = predicted
        // giả sử class 0 = No, class 1 = Yes
        CostMatrix costMatrix = new CostMatrix(2);
        costMatrix.setElement(0, 0, 0.0); // đúng No
        costMatrix.setElement(1, 1, 0.0); // đúng Yes
        costMatrix.setElement(0, 1, 1.0); // FP
        costMatrix.setElement(1, 0, 5.0); // FN (phạt nặng hơn)

        CostSensitiveClassifier csc = new CostSensitiveClassifier();
        csc.setClassifier(rf);
        csc.setCostMatrix(costMatrix);
        csc.setMinimizeExpectedCost(true);
        csc.buildClassifier(train);

        return csc;
    }

    private static void evaluateModel(Classifier cls, Instances data, String name) throws Exception {
        long start = System.currentTimeMillis();

        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(cls, data, 10, new Random(1));

        long end = System.currentTimeMillis();
        long runtimeMs = end - start;

        System.out.println("\n--------------------------------------------------------");
        System.out.println("Improved Model: " + name);
        System.out.println("--------------------------------------------------------");
        System.out.printf("Accuracy: %.2f%%%n", eval.pctCorrect());
        System.out.printf("Weighted Precision: %.4f%n", eval.weightedPrecision());
        System.out.printf("Weighted Recall:    %.4f%n", eval.weightedRecall());
        System.out.printf("Weighted F1-score:  %.4f%n", eval.weightedFMeasure());
        System.out.println("Runtime: " + runtimeMs + " ms");

        System.out.println("\nPer-class details:");
        System.out.println(eval.toClassDetailsString());

        System.out.println("Confusion Matrix:");
        System.out.println(eval.toMatrixString());
    }
}
