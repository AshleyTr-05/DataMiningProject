import weka.core.Instances;
import weka.core.Attribute;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.instance.RemoveDuplicates;

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
        System.out.println("=== AFTER ZERO TO MISSING HANDLING ===");
        printMissingAndZeroReport(data);

        // STEP 2: Remove duplicates
        data = removeDuplicates(data);

        fillMissingValues(data);

        printMissingAndZeroReport(data);

        // Normalize numeric attributes
        System.out.println();
        normalizeNumericAttributes(data);
        System.out.println("=== NORMALIZATION COMPLETED ===");

        // STEP 3: Convert categorical to numerical
        data = convertCategoricalToNumerical(data);

        // Final status report
        printFinalStatusReport(data);

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
        System.out.println("=== MISSING OR ZERO VALUE REPORT ===");
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

    // --- STEP: treat 0 as missing for selected numeric attributes ---
    private static void handleZeroAsMissing(Instances data) {
        System.out.println();
        System.out.println("=== HANDLING ZERO VALUES AS MISSING FOR SELECTED ATTRIBUTES ===");

        // List of attribute names where 0 is considered invalid/missing
        String[] zeroAsMissingAttrs = {
                "age",
                "blood_pressure",
                "cholesterol_level",
                "bmi",
                "triglyceride_level",
                "fasting_blood_sugar",
                "crp_level",
                "homocysteine_level",
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

    // --- STEP: Remove duplicate rows ---
    private static Instances removeDuplicates(Instances data) throws Exception {
        System.out.println();
        System.out.println("=== REMOVING DUPLICATE ROWS ===");
        int originalSize = data.numInstances();

        RemoveDuplicates duplicateFilter = new RemoveDuplicates();
        duplicateFilter.setInputFormat(data);
        Instances uniqueData = Filter.useFilter(data, duplicateFilter);

        int duplicatesRemoved = originalSize - uniqueData.numInstances();
        System.out.printf("Original instances: %d%n", originalSize);
        System.out.printf("Unique instances: %d%n", uniqueData.numInstances());
        System.out.printf("Duplicates removed: %d%n", duplicatesRemoved);

        return uniqueData;
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

    // --- STEP: normalize numeric attributes to [0, 1] ---
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

    // --- STEP: Convert categorical (nominal) attributes to numerical ---
    private static Instances convertCategoricalToNumerical(Instances data) throws Exception {
        System.out.println();
        System.out.println("=== CONVERTING CATEGORICAL TO NUMERICAL (Binary Encoding) ===");

        // Count nominal attributes (excluding class)
        int nominalCount = 0;
        for (int j = 0; j < data.numAttributes(); j++) {
            if (data.attribute(j).isNominal() && j != data.classIndex()) {
                nominalCount++;
            }
        }

        System.out.printf("Found %d categorical attributes (excluding class)%n", nominalCount);

        if (nominalCount == 0) {
            System.out.println("No categorical attributes to convert.");
            return data;
        }

        // Apply NominalToBinary filter
        NominalToBinary nominalToBinary = new NominalToBinary();
        nominalToBinary.setInputFormat(data);
        Instances transformedData = Filter.useFilter(data, nominalToBinary);

        System.out.printf("Attributes before conversion: %d%n", data.numAttributes());
        System.out.printf("Attributes after conversion: %d%n", transformedData.numAttributes());
        System.out.println("Categorical attributes have been converted to binary (0/1) format.");

        return transformedData;
    }

    // --- Final comprehensive status report ---
    private static void printFinalStatusReport(Instances data) {
        System.out.println();
        System.out.println("=".repeat(70));
        System.out.println("=== FINAL PREPROCESSING STATUS REPORT ===");
        System.out.println("=".repeat(70));

        System.out.println("\n1. DATASET OVERVIEW:");
        System.out.printf("   - Total instances: %d%n", data.numInstances());
        System.out.printf("   - Total attributes: %d%n", data.numAttributes());
        System.out.printf("   - Class attribute: %s%n", data.classAttribute().name());

        System.out.println("\n2. MISSING VALUES:");
        int totalMissing = 0;
        for (int j = 0; j < data.numAttributes(); j++) {
            for (int i = 0; i < data.numInstances(); i++) {
                if (data.instance(i).isMissing(j)) {
                    totalMissing++;
                }
            }
        }
        System.out.printf("   - Total missing values: %d %n", totalMissing);

        System.out.println("\n3. DUPLICATES:");
        System.out.println("   - All duplicate rows have been removed ");

        System.out.println("\n4. ATTRIBUTE TYPES:");
        int numericCount = 0;
        int nominalCount = 0;
        for (int j = 0; j < data.numAttributes(); j++) {
            if (data.attribute(j).isNumeric()) {
                numericCount++;
            } else if (data.attribute(j).isNominal()) {
                nominalCount++;
            }
        }
        System.out.printf("   - Numeric attributes: %d%n", numericCount);
        System.out.printf("   - Nominal attributes: %d (including class)%n", nominalCount);

        System.out.println("\n5. NORMALIZATION:");
        System.out.println("   - All numeric attributes normalized to [0, 1] ");

        System.out.println("\n6. CATEGORICAL CONVERSION:");
        System.out.println("   - Categorical attributes converted to numerical (binary) ");

        System.out.println("\n" + "=".repeat(70));
        System.out.println(" ALL PREPROCESSING STEPS COMPLETED SUCCESSFULLY");
        System.out.println("=".repeat(70));
    }

    public static void main(String[] args) throws Exception {

        String inputCsv = (args.length > 0) ? args[0] : "datasets/heart_disease.csv";
        String outputArff = (args.length > 1) ? args[1] : "datasets/heart_disease_preprocessed.arff";

        System.out.println("Input CSV: " + inputCsv);
        System.out.println("Output ARFF: " + outputArff);

        preprocess(inputCsv, outputArff);
    }
}
