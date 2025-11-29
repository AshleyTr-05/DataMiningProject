import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.CostMatrix;


import weka.classifiers.bayes.NaiveBayes;              // Step 4 – thêm model
import weka.classifiers.lazy.IBk;                      // kNN
import weka.classifiers.functions.Logistic;            // Logistic Regression
import weka.classifiers.functions.SMO;                 // SVM

import weka.filters.Filter;                             // Step 4 – feature selection


import java.util.Random;

public class Improver {

    public static void main(String[] args) throws Exception {

        printLine();
        System.out.println("=== HEART DISEASE - IMPROVEMENT EXPERIMENTS (STEP 3 - PHAM HUYNH DUC) ===");
        printLine();

        String arffPath = (args.length > 0)
                ? args[0]
                : "datasets/heart_disease_preprocessed.arff";

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
        System.out.println();
        System.out.println("[Step] Building Cost-Sensitive RandomForest...");
        Classifier csRandomForest = buildCostSensitiveRandomForest(data);

        // 3. Evaluate with 10-fold CV
        System.out.println("[Step] Evaluating model with 10-fold cross-validation...");
        evaluateModel(csRandomForest, data, "CostSensitive RandomForest");

        // ===================== STEP 4 – ADD MORE EXPERIMENTS =====================
        System.out.println();
        System.out.println("======================================================================");
        System.out.println("[Step 4] Running additional improvement experiments...");
        System.out.println("======================================================================");

        // 4.1 – Thử thêm nhiều model baseline khác (Logistic, NB, kNN, SVM)
        runAdditionalModels(data);

        // 4.2 – Feature selection + RandomForest
        runFeatureSelectionExperiment(data);

        printLine();
        System.out.println("=== END OF IMPROVEMENT EXPERIMENTS ===");
        printLine();
    }

    private static void printLine() {
        System.out.println("======================================================================");
    }

    private static Classifier buildCostSensitiveRandomForest(Instances train) throws Exception {
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);   // số cây
        rf.setMaxDepth(0);          // 0 = không giới hạn, rừng sẽ tự regularize

        // Cost matrix: rows = actual, cols = predicted
        // Giả sử: class 0 = No, class 1 = Yes
        CostMatrix costMatrix = new CostMatrix(2);
        costMatrix.setElement(0, 0, 0.0); // đúng No
        costMatrix.setElement(1, 1, 0.0); // đúng Yes
        costMatrix.setElement(0, 1, 1.0); // false positive
        costMatrix.setElement(1, 0, 5.0); // false negative 

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

        System.out.println();
        System.out.println("--------------------------------------------------------");
        System.out.println("Improved Model: " + name);
        System.out.println("--------------------------------------------------------");
        System.out.printf("Accuracy: %.2f%%%n", eval.pctCorrect());
        System.out.printf("Weighted Precision: %.4f%n", eval.weightedPrecision());
        System.out.printf("Weighted Recall:    %.4f%n", eval.weightedRecall());
        System.out.printf("Weighted F1-score:  %.4f%n", eval.weightedFMeasure());
        System.out.println("Runtime: " + runtimeMs + " ms");

        System.out.println();
        System.out.println("Per-class details:");
        System.out.println(eval.toClassDetailsString());

        System.out.println("Confusion Matrix:");
        System.out.println(eval.toMatrixString());
    }

    // ===================== STEP 4 – ADD ANOTHER MODEL =====================

    /**
     * Step 4 – Thử thêm các mô hình baseline khác trên full feature set:
     * Logistic Regression, Naive Bayes, kNN (k=5), SVM (SMO).
     */
    private static void runAdditionalModels(Instances data) throws Exception {
        System.out.println();
        System.out.println("[Step 4.1] Evaluating additional baseline models on full feature set...");

        Classifier[] models = new Classifier[] {
                new Logistic(),
                new NaiveBayes(),
                new IBk(5),   // kNN với k = 5
                new SMO()     // SVM
        };

        String[] names = new String[] {
                "Step 4 – Logistic Regression",
                "Step 4 – Naive Bayes",
                "Step 4 – kNN (k=5)",
                "Step 4 – SVM (SMO)"
        };

        for (int i = 0; i < models.length; i++) {
            System.out.println();
            System.out.println("[Step 4.1] --------------------------------------------");
            System.out.println("[Step 4.1] Model: " + names[i]);
            evaluateModel(models[i], data, names[i]);
        }
    }

    // ===================== STEP 4 – FEATURE SELECTION =====================

    /**
     * Step 4 – Dùng InfoGain + Ranker chọn top-k thuộc tính, sau đó train lại
     * RandomForest trên tập thuộc tính đã chọn.
     */
    private static void runFeatureSelectionExperiment(Instances data) throws Exception {
        System.out.println();
        System.out.println("[Step 4.2] Running feature selection (InfoGain + Ranker)...");

        int k = 8; // ví dụ chọn top 8 thuộc tính quan trọng
        AttributeSelection filter = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker search = new Ranker();
        search.setNumToSelect(k);

        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);

        Instances reducedData = Filter.useFilter(data, filter);
        if (reducedData.classIndex() == -1) {
            reducedData.setClassIndex(reducedData.numAttributes() - 1);
        }

        System.out.println("[Step 4.2] Original attributes: " + data.numAttributes());
        System.out.println("[Step 4.2] Attributes after selection (top " + k + "): " + reducedData.numAttributes());
        System.out.println("[Step 4.2] Relation (after selection): " + reducedData.relationName());

        // Dùng lại RandomForest cơ bản
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        rf.setMaxDepth(0);

        System.out.println();
        System.out.println("[Step 4.2] Evaluating RandomForest on selected features...");
        evaluateModel(rf, reducedData, "Step 4 – RandomForest + InfoGain (top " + k + " attrs)");
    }
}
