/**
 * Main.java - Run complete data mining pipeline in one go
 * 
 * This class orchestrates all steps:
 * 1. Preprocessing (CSV -> ARFF)
 * 2. Classification (Train & Evaluate multiple classifiers)
 * 3. Evaluation (Compare baseline vs improved models)
 * 4. Improvement experiments (Cost-sensitive, feature selection, etc.)
 */
public class Main {

    public static void main(String[] args) throws Exception {
        
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║           HEART DISEASE DATA MINING - COMPLETE PIPELINE                       ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        
        // Default paths (relative to project root, not src folder)
        // When running from src/, we need to go up one level
        String baseDir = System.getProperty("user.dir");
        
        // Check if we're running from src directory
        if (baseDir.endsWith("src")) {
            baseDir = baseDir.substring(0, baseDir.length() - 4); // Remove "/src"
        }
        
        String inputCsv = baseDir + "/datasets/heart_disease.csv";
        String outputArff = baseDir + "/datasets/heart_disease_preprocessed.arff";
        
        // Override with command line arguments if provided
        if (args.length > 0) {
            inputCsv = args[0];
        }
        if (args.length > 1) {
            outputArff = args[1];
        }
        
        long totalStart = System.currentTimeMillis();
        
        // Variables to store runtime of each step
        long step1Time = 0;
        long step2Time = 0;
        long step3Time = 0;
        long step4Time = 0;
        
        // ==================== STEP 1: PREPROCESSING ====================
        System.out.println("\n");
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║  STEP 1: DATA PREPROCESSING                                                    ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        
        long step1Start = System.currentTimeMillis();
        
        try {
            System.out.println("\nInput CSV:  " + inputCsv);
            System.out.println("Output ARFF: " + outputArff);
            System.out.println();
            
            String processedArff = Preprocessor.preprocess(inputCsv, outputArff);
            
            long step1End = System.currentTimeMillis();
            step1Time = step1End - step1Start;
            System.out.println("\n✓ Preprocessing completed in " + step1Time + " ms");
            System.out.println("✓ Output file: " + processedArff);
            
        } catch (Exception e) {
            System.err.println("✗ Preprocessing failed: " + e.getMessage());
            e.printStackTrace();
            return;
        }
        
        // ==================== STEP 2: CLASSIFICATION ====================
        System.out.println("\n");
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║  STEP 2: CLASSIFICATION (Multiple Algorithms)                                  ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        
        long step2Start = System.currentTimeMillis();
        
        try {
            // Call Classifier.main with the preprocessed ARFF path
            String[] classifierArgs = {outputArff};
            Classifier.main(classifierArgs);
            
            long step2End = System.currentTimeMillis();
            step2Time = step2End - step2Start;
            System.out.println("\n✓ Classification completed in " + step2Time + " ms");
            
        } catch (Exception e) {
            System.err.println("✗ Classification failed: " + e.getMessage());
            e.printStackTrace();
        }
        
        // ==================== STEP 3: EVALUATION ====================
        System.out.println("\n");
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║  STEP 3: MODEL EVALUATION (Baseline vs Improved)                               ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        
        long step3Start = System.currentTimeMillis();
        
        try {
            // Call Evaluator.main with the preprocessed ARFF path
            String[] evaluatorArgs = {outputArff};
            Evaluator.main(evaluatorArgs);
            
            long step3End = System.currentTimeMillis();
            step3Time = step3End - step3Start;
            System.out.println("\n✓ Evaluation completed in " + step3Time + " ms");
            
        } catch (Exception e) {
            System.err.println("✗ Evaluation failed: " + e.getMessage());
            e.printStackTrace();
        }
        
        // ==================== STEP 4: IMPROVEMENT EXPERIMENTS ====================
        System.out.println("\n");
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║  STEP 4: IMPROVEMENT EXPERIMENTS                                               ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        
        long step4Start = System.currentTimeMillis();
        
        try {
            // Call Improver.main with the preprocessed ARFF path
            String[] improverArgs = {outputArff};
            Improver.main(improverArgs);
            
            long step4End = System.currentTimeMillis();
            step4Time = step4End - step4Start;
            System.out.println("\n✓ Improvement experiments completed in " + step4Time + " ms");
            
        } catch (Exception e) {
            System.err.println("✗ Improvement experiments failed: " + e.getMessage());
            e.printStackTrace();
        }
        
        // ==================== FINAL SUMMARY ====================
        long totalEnd = System.currentTimeMillis();
        long totalTime = totalEnd - totalStart;
        
        System.out.println("\n");
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                        PIPELINE EXECUTION COMPLETE                             ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();
        
        // Print runtime summary table
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                         RUNTIME SUMMARY BY MODULE                              ║");
        System.out.println("╠════════════════════════════════════════════════════════════════════════════════╣");
        System.out.printf("║  %-50s │ %12s │ %8s ║%n", "Module", "Time (ms)", "Time (s)");
        System.out.println("╠════════════════════════════════════════════════════════════════════════════════╣");
        System.out.printf("║  %-50s │ %12d │ %8.2f ║%n", "Step 1: Data Preprocessing", step1Time, step1Time / 1000.0);
        System.out.printf("║  %-50s │ %12d │ %8.2f ║%n", "Step 2: Classification", step2Time, step2Time / 1000.0);
        System.out.printf("║  %-50s │ %12d │ %8.2f ║%n", "Step 3: Model Evaluation", step3Time, step3Time / 1000.0);
        System.out.printf("║  %-50s │ %12d │ %8.2f ║%n", "Step 4: Improvement Experiments", step4Time, step4Time / 1000.0);
        System.out.println("╠════════════════════════════════════════════════════════════════════════════════╣");
        System.out.printf("║  %-50s │ %12d │ %8.2f ║%n", "TOTAL", totalTime, totalTime / 1000.0);
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();
        
        // Print percentage breakdown
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                         RUNTIME PERCENTAGE BREAKDOWN                           ║");
        System.out.println("╠════════════════════════════════════════════════════════════════════════════════╣");
        if (totalTime > 0) {
            double pct1 = (step1Time * 100.0) / totalTime;
            double pct2 = (step2Time * 100.0) / totalTime;
            double pct3 = (step3Time * 100.0) / totalTime;
            double pct4 = (step4Time * 100.0) / totalTime;
            
            System.out.printf("║  Step 1: Preprocessing         %6.2f%% ", pct1);
            printProgressBar(pct1);
            System.out.println(" ║");
            
            System.out.printf("║  Step 2: Classification        %6.2f%% ", pct2);
            printProgressBar(pct2);
            System.out.println(" ║");
            
            System.out.printf("║  Step 3: Evaluation            %6.2f%% ", pct3);
            printProgressBar(pct3);
            System.out.println(" ║");
            
            System.out.printf("║  Step 4: Improvement           %6.2f%% ", pct4);
            printProgressBar(pct4);
            System.out.println(" ║");
        }
        System.out.println("╚════════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();
        
        System.out.println("Steps completed:");
        System.out.println("  ✓ Step 1: Data Preprocessing (CSV → ARFF)");
        System.out.println("  ✓ Step 2: Classification (J48, Naive Bayes, SVM, k-NN, Random Forest)");
        System.out.println("  ✓ Step 3: Model Evaluation (Baseline vs Improved comparison)");
        System.out.println("  ✓ Step 4: Improvement Experiments (Cost-sensitive, Feature Selection)");
        System.out.println();
        System.out.println("Output files:");
        System.out.println("  • " + outputArff);
        System.out.println();
    }
    
    /**
     * Print a progress bar based on percentage
     */
    private static void printProgressBar(double percentage) {
        int barLength = 30;
        int filled = (int) Math.round(percentage / 100.0 * barLength);
        StringBuilder bar = new StringBuilder("[");
        for (int i = 0; i < barLength; i++) {
            if (i < filled) {
                bar.append("█");
            } else {
                bar.append("░");
            }
        }
        bar.append("]");
        System.out.print(bar.toString());
    }
}