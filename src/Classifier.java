import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

import java.util.Random;
import java.util.concurrent.*;
import java.util.List;
import java.util.ArrayList;

public class Classifier {
    
    // Number of threads for parallel execution
    static int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    static ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

    public static void main(String[] args) throws Exception {
        System.out.println("=".repeat(80));
        System.out.println("=== HEART DISEASE CLASSIFICATION SYSTEM (PARALLEL MODE) ===");
        System.out.println("=== Using " + NUM_THREADS + " CPU threads ===");
        System.out.println("=".repeat(80));

        // 1. Load preprocessed ARFF dataset
        // Support both absolute path and relative path
        String datasetPath;
        if (args.length > 0) {
            // Use command line argument if provided
            datasetPath = args[0];
        } else {
            // Default: use relative path
            datasetPath = "datasets/heart_disease_preprocessed.arff";
        }
        
        System.out.println("\n--- Loading Dataset ---");
        System.out.println("Dataset path: " + datasetPath);
        
        DataSource source = new DataSource(datasetPath);
        Instances data = source.getDataSet();
        
        System.out.println("Dataset loaded successfully!");

        // 2. Set class attribute (last attribute)
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Print dataset information
        printDatasetInfo(data);

        // 3. Train and evaluate multiple classifiers in PARALLEL
        System.out.println("\n" + "=".repeat(80));
        System.out.println("=== TRAINING AND EVALUATING CLASSIFIERS (PARALLEL) ===");
        System.out.println("=".repeat(80));

        // Create a list of classifier tasks
        List<Future<String>> futures = new ArrayList<>();
        
        // J48 Decision Tree
        futures.add(executor.submit(() -> {
            System.out.println("\n" + "█".repeat(80));
            System.out.println("█ 1. J48 DECISION TREE");
            System.out.println("█".repeat(80));
            evaluateClassifier(new J48(), new Instances(data), "J48 Decision Tree");
            return "J48 done";
        }));

        // Naive Bayes
        futures.add(executor.submit(() -> {
            System.out.println("\n" + "█".repeat(80));
            System.out.println("█ 2. NAIVE BAYES");
            System.out.println("█".repeat(80));
            evaluateClassifier(new NaiveBayes(), new Instances(data), "Naive Bayes");
            return "NaiveBayes done";
        }));

        // Support Vector Machine (SMO)
        futures.add(executor.submit(() -> {
            System.out.println("\n" + "█".repeat(80));
            System.out.println("█ 3. SUPPORT VECTOR MACHINE (SVM)");
            System.out.println("█".repeat(80));
            evaluateClassifier(new SMO(), new Instances(data), "SVM (SMO)");
            return "SVM done";
        }));

        // k-Nearest Neighbors (k=3)
        futures.add(executor.submit(() -> {
            System.out.println("\n" + "█".repeat(80));
            System.out.println("█ 4. k-NEAREST NEIGHBORS (k=3)");
            System.out.println("█".repeat(80));
            IBk knn = new IBk(3);
            evaluateClassifier(knn, new Instances(data), "k-NN (k=3)");
            return "kNN done";
        }));

        // Random Forest (with multi-threading enabled)
        futures.add(executor.submit(() -> {
            System.out.println("\n" + "█".repeat(80));
            System.out.println("█ 5. RANDOM FOREST");
            System.out.println("█".repeat(80));
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100);
            rf.setNumExecutionSlots(NUM_THREADS); // Use all CPU threads
            evaluateClassifier(rf, new Instances(data), "Random Forest");
            return "RandomForest done";
        }));

        // Wait for all classifiers to complete
        for (Future<String> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                System.err.println("Error in classifier: " + e.getMessage());
            }
        }
        
        // Shutdown executor
        executor.shutdown();

        // Final summary
        System.out.println("\n" + "=".repeat(80));
        System.out.println("ALL CLASSIFIERS COMPLETED SUCCESSFULLY");
        System.out.println("=".repeat(80));
    }

    /**
     * Print dataset information
     */
    private static void printDatasetInfo(Instances data) {
        System.out.println("\n--- Dataset Information ---");
        System.out.printf("Dataset Name: %s%n", data.relationName());
        System.out.printf("Number of Instances: %d%n", data.numInstances());
        System.out.printf("Number of Attributes: %d%n", data.numAttributes());
        System.out.printf("Class Attribute: %s%n", data.classAttribute().name());
        System.out.printf("Class Values: ");
        for (int i = 0; i < data.classAttribute().numValues(); i++) {
            System.out.printf("%s ", data.classAttribute().value(i));
        }
        System.out.println();
    }

    /**
     * Train and evaluate a classifier using 10-fold cross-validation
     */
    private static void evaluateClassifier(weka.classifiers.Classifier classifier, 
                                          Instances data, 
                                          String classifierName) throws Exception {
        
        // Print algorithm description
        printAlgorithmDescription(classifierName);

        // Build classifier
        System.out.println("\n--- Training Classifier ---");
        long startTime = System.currentTimeMillis();
        classifier.buildClassifier(data);
        long trainingTime = System.currentTimeMillis() - startTime;
        System.out.printf("Training completed in %d ms%n", trainingTime);

        // Evaluate with 10-fold cross-validation
        System.out.println("\n--- Performing 10-Fold Cross-Validation ---");
        Evaluation eval = new Evaluation(data);
        startTime = System.currentTimeMillis();
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        long evalTime = System.currentTimeMillis() - startTime;
        System.out.printf("Evaluation completed in %d ms%n", evalTime);

        // Print detailed results
        printDetailedResults(eval, classifierName);
    }

    /**
     * Print algorithm description
     */
    private static void printAlgorithmDescription(String classifierName) {
        System.out.println("\n┌" + "─".repeat(78) + "┐");
        System.out.println("│ ALGORITHM DESCRIPTION" + " ".repeat(55) + "│");
        System.out.println("└" + "─".repeat(78) + "┘");

        switch (classifierName) {
            case "J48 Decision Tree":
                System.out.println("Algorithm: J48 (C4.5 Decision Tree)");
                System.out.println("Description:");
                System.out.println("  - Uses information gain to select the best attribute for splitting");
                System.out.println("  - Builds a tree structure where each node represents a decision");
                System.out.println("  - Performs pruning to avoid overfitting");
                System.out.println("  - Handles both continuous and discrete attributes");
                System.out.println("Advantages:");
                System.out.println("  - Easy to interpret and visualize");
                System.out.println("  - Handles non-linear relationships well");
                System.out.println("  - Requires little data preprocessing");
                break;

            case "Naive Bayes":
                System.out.println("Algorithm: Naive Bayes");
                System.out.println("Description:");
                System.out.println("  - Based on Bayes' theorem with naive independence assumption");
                System.out.println("  - Assumes all features are independent given the class");
                System.out.println("  - Calculates P(Class|Features) for classification");
                System.out.println("  - Fast and efficient for large datasets");
                System.out.println("Advantages:");
                System.out.println("  - Simple and fast training");
                System.out.println("  - Works well with small training datasets");
                System.out.println("  - Handles high-dimensional data effectively");
                break;

            case "SVM (SMO)":
                System.out.println("Algorithm: Support Vector Machine (Sequential Minimal Optimization)");
                System.out.println("Description:");
                System.out.println("  - Finds optimal hyperplane to separate classes");
                System.out.println("  - Maximizes the margin between support vectors");
                System.out.println("  - Uses kernel trick for non-linear classification");
                System.out.println("  - SMO is an efficient algorithm for training SVM");
                System.out.println("Advantages:");
                System.out.println("  - Effective in high-dimensional spaces");
                System.out.println("  - Memory efficient (uses support vectors only)");
                System.out.println("  - Versatile with different kernel functions");
                break;

            case "k-NN (k=3)":
                System.out.println("Algorithm: k-Nearest Neighbors (k=3)");
                System.out.println("Description:");
                System.out.println("  - Instance-based learning algorithm");
                System.out.println("  - Classifies based on majority vote of k nearest neighbors");
                System.out.println("  - Uses distance metrics (typically Euclidean)");
                System.out.println("  - k=3 means considers 3 closest neighbors");
                System.out.println("Advantages:");
                System.out.println("  - Simple and intuitive");
                System.out.println("  - No training phase (lazy learning)");
                System.out.println("  - Adapts easily as new data is added");
                break;

            case "Random Forest":
                System.out.println("Algorithm: Random Forest");
                System.out.println("Description:");
                System.out.println("  - Ensemble method using multiple decision trees");
                System.out.println("  - Each tree is trained on random subset of data");
                System.out.println("  - Final prediction is majority vote of all trees");
                System.out.println("  - Uses feature bagging to reduce overfitting");
                System.out.println("Advantages:");
                System.out.println("  - High accuracy and robustness");
                System.out.println("  - Handles large datasets with high dimensionality");
                System.out.println("  - Provides feature importance rankings");
                break;
        }
    }

    /**
     * Print detailed evaluation results
     */
    private static void printDetailedResults(Evaluation eval, String classifierName) throws Exception {
        System.out.println("\n┌" + "─".repeat(78) + "┐");
        System.out.println("│ CLASSIFICATION RESULTS" + " ".repeat(54) + "│");
        System.out.println("└" + "─".repeat(78) + "┘");

        // Overall statistics
        System.out.println("\n=== Overall Performance Metrics ===");
        System.out.printf("Correctly Classified Instances: %d (%.4f%%)%n",
                (int) eval.correct(), eval.pctCorrect());
        System.out.printf("Incorrectly Classified Instances: %d (%.4f%%)%n",
                (int) eval.incorrect(), eval.pctIncorrect());
        System.out.printf("Total Number of Instances: %d%n", (int) eval.numInstances());

        // Detailed accuracy metrics
        System.out.println("\n=== Detailed Accuracy Metrics ===");
        System.out.printf("Accuracy: %.4f%%%n", eval.pctCorrect());
        System.out.printf("Kappa Statistic: %.4f%n", eval.kappa());
        System.out.printf("Mean Absolute Error: %.4f%n", eval.meanAbsoluteError());
        System.out.printf("Root Mean Squared Error: %.4f%n", eval.rootMeanSquaredError());
        System.out.printf("Relative Absolute Error: %.4f%%%n", eval.relativeAbsoluteError());
        System.out.printf("Root Relative Squared Error: %.4f%%%n", eval.rootRelativeSquaredError());

        // Confusion Matrix
        System.out.println("\n=== Confusion Matrix ===");
        double[][] confusionMatrix = eval.confusionMatrix();
        System.out.println("                 Predicted");
        System.out.println("              No      Yes");
        System.out.println("        ┌─────────────────┐");
        System.out.printf("Actual No│ %6.0f  %6.0f │%n", confusionMatrix[0][0], confusionMatrix[0][1]);
        System.out.printf("      Yes│ %6.0f  %6.0f │%n", confusionMatrix[1][0], confusionMatrix[1][1]);
        System.out.println("        └─────────────────┘");

        // Calculate metrics for each class
        System.out.println("\n=== Class-Specific Performance ===");
        for (int i = 0; i < eval.getHeader().classAttribute().numValues(); i++) {
            String className = eval.getHeader().classAttribute().value(i);
            System.out.printf("\nClass: %s%n", className);
            System.out.printf("  Precision (PPV): %.4f%n", eval.precision(i));
            System.out.printf("  Recall (TPR):    %.4f%n", eval.recall(i));
            System.out.printf("  F-Measure:       %.4f%n", eval.fMeasure(i));
            System.out.printf("  ROC Area:        %.4f%n", eval.areaUnderROC(i));
            System.out.printf("  PRC Area:        %.4f%n", eval.areaUnderPRC(i));
        }

        // Weighted averages
        System.out.println("\n=== Weighted Average Metrics ===");
        System.out.printf("Weighted Precision: %.4f%n", eval.weightedPrecision());
        System.out.printf("Weighted Recall:    %.4f%n", eval.weightedRecall());
        System.out.printf("Weighted F-Measure: %.4f%n", eval.weightedFMeasure());
        System.out.printf("Weighted ROC Area:  %.4f%n", eval.weightedAreaUnderROC());
        System.out.printf("Weighted PRC Area:  %.4f%n", eval.weightedAreaUnderPRC());

        System.out.println("\n" + "─".repeat(80));
    }
}
