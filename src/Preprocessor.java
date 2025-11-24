import weka.core.Instances;
import weka.core.Attribute;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;

import java.io.File;

public class Preprocessor {

    public static String preprocess(String inputCsvPath, String outputArffPath) throws Exception {

        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(inputCsvPath)); // <-- datasets/heart_disease.csv
        Instances data = loader.getDataSet();

        // set class attribute to last column
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // print Basic data set summary
        printDatasetSummary(data);

        // print Missing and Zero values report (BEFORE cleaning)
        System.out.println();
        System.out.println("=== BEFORE CLEANING ===");
        printMissingAndZeroReport(data);

        // STEP 1: handle suspicious zeros by marking them as missing
        handleZeroAsMissing(data);

        // print Missing and Zero values report (AFTER cleaning)
        System.out.println();
        System.out.println("=== AFTER ZERO→MISSING HANDLING ===");
        printMissingAndZeroReport(data);

        fillMissingValues(data);

        printMissingAndZeroReport(data);

        System.out.println();
        normalizeNumericAttributes(data);
        System.out.println("=== NORMALIZATION COMPLETED ===");

        // save to ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputArffPath));
        saver.writeBatch();

        System.out.println();
        System.out.println("=== ARFF file saved ===");
        System.out.println("Output ARFF: " + outputArffPath);

        return outputArffPath;
    }

    public static void printDatasetSummary(Instances data) {
        System.out.println("=== Dataset Summary ===");
        System.out.println("Number of instances: " + data.numInstances());
        System.out.println("Number of attributes: " + data.numAttributes());
        System.out.println("Class attribute: " + data.classAttribute().name());
        System.out.println("Attribute list:");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            System.out.printf("  [%d] %s (%s)%n",
                    i,
                    attr.name(),
                    attr.isNumeric() ? "numeric" : "nominal");
        }
    }

    // --- Helper: report missing and zero values for each attribute ---
    private static void printMissingAndZeroReport(Instances data) {
        System.out.println("=== MISSING / ZERO VALUE REPORT ===");
        int numInstances = data.numInstances();

        for (int j = 0; j < data.numAttributes(); j++) {
            Attribute attr = data.attribute(j);
            int missingCount = 0;
            int zeroCount = 0;

            for (int i = 0; i < numInstances; i++) {
                if (data.instance(i).isMissing(j)) {
                    missingCount++;
                } else if (attr.isNumeric()) {
                    double val = data.instance(i).value(j);
                    if (val == 0.0) {
                        zeroCount++;
                    }
                }
            }

            System.out.printf("Attribute: %-20s | Missing: %4d | Zero values (numeric only): %4d%n",
                    attr.name(), missingCount, (attr.isNumeric() ? zeroCount : 0));
        }
        System.out.println();
        System.out.println("Note: Zero values might or might not be invalid.");
        System.out.println("      Later we will decide which attributes treat 0 as 'missing'.");
    }

    // --- NEW STEP: treat 0 as missing for selected numeric attributes ---
    private static void handleZeroAsMissing(Instances data) {
        System.out.println();
        System.out.println("=== HANDLING ZERO VALUES AS MISSING FOR SELECTED ATTRIBUTES ===");

        // List of attribute names where 0 is considered invalid/missing
        String[] zeroAsMissingAttrs = {
                "chol", "cholesterol", // possible names for cholesterol
                "trestbps", "restingbp", // resting blood pressure
                "thalach", "maxhr", "max_heart_rate"
        };

        int totalReplaced = 0;

        for (String name : zeroAsMissingAttrs) {
            Attribute attr = data.attribute(name);
            if (attr == null || !attr.isNumeric()) {
                // Attribute with this name does not exist in this dataset
                continue;
            }

            int replacedForThisAttr = 0;
            for (int i = 0; i < data.numInstances(); i++) {
                if (!data.instance(i).isMissing(attr)) {
                    double val = data.instance(i).value(attr);
                    if (val == 0.0) {
                        data.instance(i).setMissing(attr);
                        replacedForThisAttr++;
                    }
                }
            }

            System.out.printf("Attribute %-20s: replaced %4d zero(s) with missing%n",
                    attr.name(), replacedForThisAttr);
            totalReplaced += replacedForThisAttr;
        }

        System.out.println("Total zeros converted to missing: " + totalReplaced);
    }

    private static void fillMissingValues(Instances data) {
        System.out.println("=== FILLING MISSING VALUES (Mean for numeric, Mode for nominal) ===");

        int totalFilled = 0;

        for (int j = 0; j < data.numAttributes(); j++) {
            Attribute attr = data.attribute(j);

            // Skip class attribute
            if (j == data.classIndex())
                continue;

            if (attr.isNumeric()) {
                // Compute mean
                double sum = 0;
                int count = 0;

                for (int i = 0; i < data.numInstances(); i++) {
                    if (!data.instance(i).isMissing(j)) {
                        sum += data.instance(i).value(j);
                        count++;
                    }
                }

                double mean = (count > 0) ? sum / count : 0;

                int filledForAttr = 0;
                for (int i = 0; i < data.numInstances(); i++) {
                    if (data.instance(i).isMissing(j)) {
                        data.instance(i).setValue(j, mean);
                        filledForAttr++;
                    }
                }

                System.out.printf("Numeric attribute %-20s | Filled: %4d | Mean used: %.2f%n",
                        attr.name(), filledForAttr, mean);
                totalFilled += filledForAttr;

            } else if (attr.isNominal()) {
                // Compute mode (most frequent category)
                int[] counts = new int[attr.numValues()];

                for (int i = 0; i < data.numInstances(); i++) {
                    if (!data.instance(i).isMissing(j)) {
                        int idx = (int) data.instance(i).value(j);
                        counts[idx]++;
                    }
                }

                // Find the mode (max count category)
                int modeIndex = 0;
                for (int k = 1; k < counts.length; k++) {
                    if (counts[k] > counts[modeIndex]) {
                        modeIndex = k;
                    }
                }

                int filledForAttr = 0;
                for (int i = 0; i < data.numInstances(); i++) {
                    if (data.instance(i).isMissing(j)) {
                        data.instance(i).setValue(j, modeIndex);
                        filledForAttr++;
                    }
                }

                System.out.printf("Nominal attribute %-20s | Filled: %4d | Mode used: %s%n",
                        attr.name(), filledForAttr, attr.value(modeIndex));
                totalFilled += filledForAttr;
            }
        }

        System.out.println("Total missing values filled: " + totalFilled);
    }

    // --- NEW STEP: normalize numeric attributes to [0, 1] ---
    private static void normalizeNumericAttributes(Instances data) {
        System.out.println("=== NORMALIZING NUMERIC ATTRIBUTES TO [0, 1] ===");

        for (int j = 0; j < data.numAttributes(); j++) {
            Attribute attr = data.attribute(j);

            // Skip non-numeric attributes and class attribute
            if (!attr.isNumeric() || j == data.classIndex()) {
                continue;
            }

            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;

            // 1. Find min and max
            for (int i = 0; i < data.numInstances(); i++) {
                double val = data.instance(i).value(j);
                if (val < min)
                    min = val;
                if (val > max)
                    max = val;
            }

            // Avoid divide by zero: if all values are same, set them to 0
            if (min == max) {
                System.out.printf("Attribute %-20s has constant value %.2f; setting all to 0.0%n",
                        attr.name(), min);
                for (int i = 0; i < data.numInstances(); i++) {
                    data.instance(i).setValue(j, 0.0);
                }
                continue;
            }

            // 2. Apply min-max normalization
            for (int i = 0; i < data.numInstances(); i++) {
                double oldVal = data.instance(i).value(j);
                double newVal = (oldVal - min) / (max - min); // in [0, 1]
                data.instance(i).setValue(j, newVal);
            }

            System.out.printf("Attribute %-20s normalized using min=%.2f, max=%.2f%n",
                    attr.name(), min, max);
        }
    }

    public static void main(String[] args) throws Exception {
        String inputCsv = "datasets/heart_disease.csv";
        String outputArff = "datasets/heart_disease_preprocessed.arff";

        preprocess(inputCsv, outputArff);
    }
}
