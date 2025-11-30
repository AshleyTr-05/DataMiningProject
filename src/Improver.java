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

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;

import weka.filters.Filter;

import java.util.Random;
import java.util.concurrent.*;
import java.util.List;
import java.util.ArrayList;

public class Improver {
    
    // Number of threads for parallel execution
    static int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    static ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

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

        // Shutdown executor
        executor.shutdown();
        
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
        rf.setMaxDepth(0);          // 0 = không giới hạn
        rf.setNumExecutionSlots(NUM_THREADS); // Use all CPU threads

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

    // ===================== STEP 4 – ADD ANOTHER MODEL (PARALLEL) =====================

    /**
     * Step 4 – Thử thêm các mô hình baseline khác trên full feature set (PARALLEL):
     * Logistic Regression, Naive Bayes, kNN (k=5), SVM (SMO).
     */
    private static void runAdditionalModels(Instances data) throws Exception {
        System.out.println();
        System.out.println("[Step 4.1] Evaluating additional baseline models in PARALLEL (" + NUM_THREADS + " threads)...");

        List<Future<String>> futures = new ArrayList<>();
        
        // Submit all models in parallel
        futures.add(executor.submit(() -> {
            Logistic log = new Logistic();
            evaluateModel(log, new Instances(data), "Step 4 – Logistic Regression");
            return "Logistic done";
        }));
        
        futures.add(executor.submit(() -> {
            NaiveBayes nb = new NaiveBayes();
            evaluateModel(nb, new Instances(data), "Step 4 – Naive Bayes");
            return "NaiveBayes done";
        }));
        
        futures.add(executor.submit(() -> {
            IBk knn = new IBk(5);
            evaluateModel(knn, new Instances(data), "Step 4 – kNN (k=5)");
            return "kNN done";
        }));
        
        futures.add(executor.submit(() -> {
            SMO svm = new SMO();
            evaluateModel(svm, new Instances(data), "Step 4 – SVM (SMO)");
            return "SVM done";
        }));

        // Wait for all to complete
        for (Future<String> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
            }
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
