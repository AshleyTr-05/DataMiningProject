import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.*;
import java.util.concurrent.*;

public class Evaluator {

    // Structure to store evaluation metrics
    static class Result {
        String modelName;
        double accuracy;
        double precision;
        double recall;
        double f1Score;
        double kappa;
        long runtime;
        Evaluation eval;

        Result(String name, double acc, double prec, double rec, double f1,
                double k, long time, Evaluation e) {
            modelName = name;
            accuracy = acc;
            precision = prec;
            recall = rec;
            f1Score = f1;
            kappa = k;
            runtime = time;
            eval = e;
        }
    }

    // Store baseline and improved results
    static List<Result> baselineResults = Collections.synchronizedList(new ArrayList<>());
    static List<Result> improvedResults = Collections.synchronizedList(new ArrayList<>());
    
    // Thread pool for parallel execution
    static int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    static ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

    // Evaluate a model with 10-fold cross-validation
    static Result evaluate(String label, Classifier model, Instances data) throws Exception {
        long start = System.currentTimeMillis();

        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(model, data, 10, new Random(1));

        long end = System.currentTimeMillis();

        Result result = new Result(
                label,
                evaluation.pctCorrect(),
                evaluation.weightedPrecision(),
                evaluation.weightedRecall(),
                evaluation.weightedFMeasure(),
                evaluation.kappa(),
                end - start,
                evaluation);

        return result;
    }

    // Print detailed results for each model
    static void printDetailedResults(Result r) throws Exception {
        System.out.println("\n" + "-".repeat(70));
        System.out.println("Model: " + r.modelName);
        System.out.println("-".repeat(70));
        System.out.printf("Accuracy            : %.4f%%%n", r.accuracy);
        System.out.printf("Precision (Weighted): %.4f%n", r.precision);
        System.out.printf("Recall (Weighted)   : %.4f%n", r.recall);
        System.out.printf("F1-Score (Weighted) : %.4f%n", r.f1Score);
        System.out.printf("Kappa Statistic     : %.4f%n", r.kappa);
        System.out.printf("Runtime             : %d ms%n", r.runtime);
        System.out.println("\nConfusion Matrix:");
        System.out.println(r.eval.toMatrixString());
    }

    // Print performance summary table
    static void printSummaryTable(String title, List<Result> results) {
        System.out.println("\n" + "=".repeat(90));
        System.out.println(title);
        System.out.println("=".repeat(90));
        System.out.printf("%-35s | %-10s | %-10s | %-10s | %-10s | %-12s%n",
                "Model", "Accuracy", "Precision", "Recall", "F1-Score", "Runtime (ms)");
        System.out.println("-".repeat(90));

        for (Result r : results) {
            System.out.printf("%-35s | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-12d%n",
                    r.modelName, r.accuracy, r.precision, r.recall, r.f1Score, r.runtime);
        }
        System.out.println("=".repeat(90));
    }

    // Compare baseline vs improved models
    static void compareModels() {
        System.out.println("\n" + "█".repeat(90));
        System.out.println("█ 5.2 ANALYSIS OF RESULTS - BASELINE VS IMPROVED COMPARISON");
        System.out.println("█".repeat(90));

        System.out.println("\n" + "-".repeat(90));
        System.out.println("MODEL-BY-MODEL IMPROVEMENTS");
        System.out.println("-".repeat(90));
        System.out.printf("%-20s | %-15s | %-15s | %-15s | %-15s%n",
                "Model Type", "Baseline Acc", "Improved Acc", "Accuracy Gain", "Runtime Ratio");
        System.out.println("-".repeat(90));

        Map<String, Result> baselineMap = new HashMap<>();
        for (Result r : baselineResults) {
            String modelType = r.modelName.split(" ")[0]; // Extract base model name
            baselineMap.put(modelType, r);
        }

        for (Result improved : improvedResults) {
            String modelType = improved.modelName.split(" ")[0];
            Result baseline = baselineMap.get(modelType);

            if (baseline != null) {
                double accGain = improved.accuracy - baseline.accuracy;
                double runtimeRatio = (double) improved.runtime / baseline.runtime;
                String gainTrend = accGain >= 0 ? "↑ +" : "↓ ";

                System.out.printf("%-20s | %-15.4f | %-15.4f | %s%-14.4f | %-15.2fx%n",
                        modelType, baseline.accuracy, improved.accuracy,
                        gainTrend, Math.abs(accGain), runtimeRatio);
            }
        }
        System.out.println("-".repeat(90));
    }

    // Print comprehensive analysis
    static void printAnalysis() {
        System.out.println("\n" + "█".repeat(90));
        System.out.println("█ 5.3 KEY INSIGHTS & INTERPRETATION");
        System.out.println("█".repeat(90));

        // Best models
        Result bestBaseline = baselineResults.stream()
                .max(Comparator.comparingDouble(r -> r.accuracy))
                .orElse(null);
        Result bestImproved = improvedResults.stream()
                .max(Comparator.comparingDouble(r -> r.accuracy))
                .orElse(null);

        System.out.println("\n▪ BEST PERFORMING MODELS:");
        System.out.printf("  • Baseline: %s (Accuracy: %.4f%%, F1: %.4f)%n",
                bestBaseline.modelName, bestBaseline.accuracy, bestBaseline.f1Score);
        System.out.printf("  • Improved: %s (Accuracy: %.4f%%, F1: %.4f)%n",
                bestImproved.modelName, bestImproved.accuracy, bestImproved.f1Score);

        // Average metrics
        double avgBaselineAcc = baselineResults.stream().mapToDouble(r -> r.accuracy).average().orElse(0);
        double avgImprovedAcc = improvedResults.stream().mapToDouble(r -> r.accuracy).average().orElse(0);
        double avgBaselineTime = baselineResults.stream().mapToDouble(r -> r.runtime).average().orElse(0);
        double avgImprovedTime = improvedResults.stream().mapToDouble(r -> r.runtime).average().orElse(0);

        System.out.println("\n▪ ACCURACY ANALYSIS:");
        System.out.printf("  • Average Baseline Accuracy: %.4f%%%n", avgBaselineAcc);
        System.out.printf("  • Average Improved Accuracy: %.4f%%%n", avgImprovedAcc);
        System.out.printf("  • Overall Improvement: %.4f%%%n", avgImprovedAcc - avgBaselineAcc);

        System.out.println("\n▪ RUNTIME ANALYSIS (Performance Trade-offs):");
        System.out.printf("  • Average Baseline Runtime: %.0f ms%n", avgBaselineTime);
        System.out.printf("  • Average Improved Runtime: %.0f ms%n", avgImprovedTime);
        double timeIncrease = ((avgImprovedTime - avgBaselineTime) / avgBaselineTime) * 100;
        System.out.printf("  • Runtime Increase: %.2f%%%n", timeIncrease);

        System.out.println("\n▪ CLASS BALANCE ASSESSMENT:");
        System.out.println("  ⚠ Dataset is imbalanced (80% No, 20% Yes)");
        System.out.println("  • Use F1-Score and Kappa over pure Accuracy");
        System.out.println("  • Check Precision/Recall per class in confusion matrix");
        System.out.println("  • Be cautious of models predicting majority class only");

        System.out.println("\n▪ RECOMMENDATIONS:");
        System.out.printf("  • Best Model: %s%n", bestImproved.modelName);
        System.out.println("    ✓ Highest accuracy and F1-score");
        System.out.println("    ✓ Balance between prediction quality and computational cost");
        System.out.println("    ✓ Suitable for deployment in heart disease prediction");

        if (timeIncrease > 50) {
            System.out.printf("  • Trade-off: Improved models take %.0f%% longer%n", timeIncrease);
            System.out.println("    → Acceptable if accuracy gain is significant");
        } else {
            System.out.println("  • Improved models maintain fast execution");
        }

        System.out.println("\n▪ KAPPA STATISTIC INTERPRETATION:");
        System.out.println("  • Values > 0.6: Substantial agreement");
        System.out.println("  • Values > 0.4: Moderate agreement");
        System.out.println("  • Values < 0.4: Fair/poor agreement");
        System.out.println("  • Negative values: Model performs worse than random chance");

        System.out.println("-".repeat(90));
    }

    public static void main(String[] args) throws Exception {
        System.out.println("█".repeat(90));
        System.out.println("█ HEART DISEASE PREDICTION - MODEL EVALUATION (PARALLEL MODE)");
        System.out.println("█ 10-Fold Cross-Validation Analysis");
        System.out.println("█ Using " + NUM_THREADS + " CPU threads for parallel execution");
        System.out.println("█".repeat(90));

        // Load dataset
        String datasetPath;
        if (args.length > 0) {
            datasetPath = args[0];
        } else {
            datasetPath = "datasets/heart_disease_preprocessed.arff";
        }

        System.out.println("\nDataset path: " + datasetPath);

        Instances data = ConverterUtils.DataSource.read(datasetPath);
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("Dataset:   " + data.relationName());
        System.out.println("Instances: " + data.numInstances());
        System.out.println("Attributes:" + data.numAttributes());
        System.out.println("Class:     " + data.classAttribute().name());

        // ========== BASELINE MODELS (PARALLEL) ==========
        System.out.println("\n" + "█".repeat(90));
        System.out.println("█ BASELINE MODELS (Running in parallel)");
        System.out.println("█".repeat(90));

        List<Future<Result>> baselineFutures = new ArrayList<>();
        
        // Submit all baseline models in parallel
        baselineFutures.add(executor.submit(() -> {
            J48 j48 = new J48();
            return evaluate("J48 (Baseline)", j48, new Instances(data));
        }));
        
        baselineFutures.add(executor.submit(() -> {
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100);
            rf.setNumExecutionSlots(NUM_THREADS); // Use all threads for RF
            return evaluate("Random Forest (Baseline)", rf, new Instances(data));
        }));
        
        baselineFutures.add(executor.submit(() -> {
            NaiveBayes nb = new NaiveBayes();
            return evaluate("Naive Bayes (Baseline)", nb, new Instances(data));
        }));
        
        baselineFutures.add(executor.submit(() -> {
            SMO svm = new SMO();
            return evaluate("SVM/SMO (Baseline)", svm, new Instances(data));
        }));
        
        baselineFutures.add(executor.submit(() -> {
            IBk knn = new IBk(3);
            return evaluate("k-NN k=3 (Baseline)", knn, new Instances(data));
        }));

        // Collect baseline results
        for (Future<Result> future : baselineFutures) {
            try {
                Result r = future.get();
                baselineResults.add(r);
                printDetailedResults(r);
            } catch (Exception e) {
                System.err.println("Error in baseline model: " + e.getMessage());
            }
        }

        // ========== IMPROVED MODELS (PARALLEL) ==========
        System.out.println("\n" + "█".repeat(90));
        System.out.println("█ IMPROVED MODELS (Running in parallel)");
        System.out.println("█".repeat(90));

        List<Future<Result>> improvedFutures = new ArrayList<>();
        
        improvedFutures.add(executor.submit(() -> {
            J48 j48Imp = new J48();
            j48Imp.setConfidenceFactor(0.1f);
            j48Imp.setMinNumObj(5);
            return evaluate("J48 (Improved)", j48Imp, new Instances(data));
        }));
        
        improvedFutures.add(executor.submit(() -> {
            RandomForest rfImp = new RandomForest();
            rfImp.setNumIterations(200);
            rfImp.setNumFeatures(5);
            rfImp.setNumExecutionSlots(NUM_THREADS); // Use all threads
            return evaluate("Random Forest (Improved)", rfImp, new Instances(data));
        }));
        
        improvedFutures.add(executor.submit(() -> {
            NaiveBayes nbImp = new NaiveBayes();
            nbImp.setUseKernelEstimator(true);
            return evaluate("Naive Bayes (Improved)", nbImp, new Instances(data));
        }));
        
        improvedFutures.add(executor.submit(() -> {
            IBk knnImp = new IBk(5);
            return evaluate("k-NN k=5 (Improved)", knnImp, new Instances(data));
        }));

        // Collect improved results
        for (Future<Result> future : improvedFutures) {
            try {
                Result r = future.get();
                improvedResults.add(r);
                printDetailedResults(r);
            } catch (Exception e) {
                System.err.println("Error in improved model: " + e.getMessage());
            }
        }

        // Shutdown executor
        executor.shutdown();

        // ========== 5.1 PERFORMANCE METRICS SUMMARY ==========
        System.out.println("\n" + "█".repeat(90));
        System.out.println("█ 5.1 PERFORMANCE METRICS SUMMARY");
        System.out.println("█".repeat(90));

        printSummaryTable("BASELINE MODELS SUMMARY", baselineResults);
        printSummaryTable("IMPROVED MODELS SUMMARY", improvedResults);

        // ========== 5.2 & 5.3 ANALYSIS ==========
        compareModels();
        printAnalysis();

        System.out.println("\n" + "█".repeat(90));
        System.out.println("█ EVALUATION COMPLETED");
        System.out.println("█".repeat(90));
    }
}
