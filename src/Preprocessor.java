import weka.core.Instances;
import weka.core.Attribute;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.instance.RemoveDuplicates;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class Preprocessor {

    public static String preprocess(String inputCsvPath, String outputArffPath) throws Exception {

        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(inputCsvPath));
        Instances data = loader.getDataSet();

        // set class attribute to last column
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        if (data.classAttribute().isNumeric()) {
            System.out.println();
            System.out.println("=== CLASS ATTRIBUTE IS NUMERIC: CONVERTING TO NOMINAL ===");
            System.out.println("Class attribute name: " + data.classAttribute().name());

            NumericToNominal num2nom = new NumericToNominal();
            // Weka uses 1-based indices for filters; "last" = class
            num2nom.setAttributeIndices("" + (data.classIndex() + 1));
            num2nom.setInputFormat(data);
            data = Filter.useFilter(data, num2nom);

            // Re-set class index (since data reference changed)
            data.setClassIndex(data.numAttributes() - 1);

            System.out.println("Class attribute converted to nominal.");
        }

        // print Basic dataset summary
        printDatasetSummary(data);

        // print Missing and Zero values report (BEFORE cleaning)
        System.out.println();
        System.out.println("=== BEFORE CLEANING ===");
        printMissingAndZeroReport(data);

        // STEP 1: handle suspicious zeros by marking them as missing
        handleZeroAsMissing(data);

        // print Missing and Zero values report (AFTER zero->missing)
        System.out.println();
        System.out.println("=== AFTER ZERO TO MISSING HANDLING ===");
        printMissingAndZeroReport(data);

        // STEP 2: Remove duplicates
        data = removeDuplicates(data);

        // STEP 3: Fill missing values
        fillMissingValues(data);

        // Print report again after filling
        printMissingAndZeroReport(data);

        // STEP 4: Normalize numeric attributes
        System.out.println();
        normalizeNumericAttributes(data);
        System.out.println("=== NORMALIZATION COMPLETED ===");

        // STEP 5: Convert categorical to numerical (safe version)
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

    // --- Dataset summary ---
    public static void printDatasetSummary(Instances data) {
        System.out.println("=== Dataset Summary ===");
        System.out.println("Number of instances: " + data.numInstances());
        System.out.println("Number of attributes: " + data.numAttributes());
        System.out.println("Class attribute: " + data.classAttribute().name());
        System.out.println("Attribute list:");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            String type = attr.isNumeric() ? "numeric" : (attr.isNominal() ? "nominal" : "other");
            String extra = attr.isNominal() ? (" | numValues=" + attr.numValues()) : "";
            System.out.printf("  [%d] %s (%s%s)%n",
                    i, attr.name(), type, extra);
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
    // Works in two modes:
    // 1) If this looks like the heart_disease dataset -> use specific medical attributes.
    // 2) Otherwise -> generic heuristic for any dataset.
    private static void handleZeroAsMissing(Instances data) {
        System.out.println();
        System.out.println("=== HANDLING ZERO VALUES AS MISSING FOR SELECTED ATTRIBUTES ===");

        // 1) Preferred known attribute names (heart_disease-style datasets)
        String[] preferredZeroAsMissingAttrs = {
                "age",
                "blood_pressure",
                "cholesterol_level",
                "bmi",
                "triglyceride_level",
                "fasting_blood_sugar",
                "crp_level",
                "homocysteine_level"
        };

        Set<Integer> selectedAttrIndices = new HashSet<>();

        // Try to select by explicit names first (if they exist in this dataset)
        for (String name : preferredZeroAsMissingAttrs) {
            Attribute attr = data.attribute(name);
            if (attr != null && attr.isNumeric()) {
                selectedAttrIndices.add(attr.index());
            }
        }

        // 2) If nothing selected (different dataset, e.g. T_data_*), use a generic heuristic:
        //    - numeric attribute
        //    - not class
        //    - has at least one 0 and at least one non-zero
        if (selectedAttrIndices.isEmpty()) {
            for (int j = 0; j < data.numAttributes(); j++) {
                if (j == data.classIndex()) continue;
                Attribute attr = data.attribute(j);
                if (!attr.isNumeric()) continue;

                boolean hasZero = false;
                boolean hasNonZero = false;

                for (int i = 0; i < data.numInstances(); i++) {
                    if (data.instance(i).isMissing(j)) continue;
                    double val = data.instance(i).value(j);
                    if (val == 0.0) {
                        hasZero = true;
                    } else {
                        hasNonZero = true;
                    }
                    if (hasZero && hasNonZero) break;
                }

                if (hasZero && hasNonZero) {
                    selectedAttrIndices.add(j);
                }
            }
        }

        int totalReplaced = 0;

        // 3) Actually replace zeros with missing in the selected attributes
        for (int j : selectedAttrIndices) {
            Attribute attr = data.attribute(j);
            int replacedForThisAttr = 0;

            for (int i = 0; i < data.numInstances(); i++) {
                if (!data.instance(i).isMissing(j)) {
                    double val = data.instance(i).value(j);
                    if (val == 0.0) {
                        data.instance(i).setMissing(j);
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

    // --- STEP: Fill missing values ---
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

                System.out.printf("Numeric attribute %-20s | Filled: %4d | Mean used: %.4f%n",
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
                System.out.printf("Attribute %-20s has constant value %.4f; setting all to 0.0%n",
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

            System.out.printf("Attribute %-20s normalized using min=%.4f, max=%.4f%n",
                    attr.name(), min, max);
        }
    }

    // --- STEP: Convert categorical (nominal) attributes to numerical ---
    //          Safe version that avoids OutOfMemory by removing huge-cardinality attributes
    private static Instances convertCategoricalToNumerical(Instances data) throws Exception {
        System.out.println();
        System.out.println("=== CONVERTING CATEGORICAL TO NUMERICAL (Binary Encoding) ===");

        // 1. Log nominal attributes and detect high-cardinality ones
        System.out.println("=== NOMINAL ATTRIBUTES (excluding class) ===");
        int classIndex = data.classIndex();
        int MAX_VALUES_FOR_BINARY = 50;   // threshold; tune if needed

        ArrayList<Integer> highCardinalityIndices = new ArrayList<>();

        for (int j = 0; j < data.numAttributes(); j++) {
            Attribute attr = data.attribute(j);
            if (attr.isNominal() && j != classIndex) {
                int numValues = attr.numValues();
                System.out.printf("  - %s | index=%d | numValues=%d%n",
                        attr.name(), j, numValues);
                if (numValues > MAX_VALUES_FOR_BINARY) {
                    highCardinalityIndices.add(j);
                }
            }
        }

        // 2. Remove high-cardinality nominal attributes to avoid dimension explosion
        Instances workingData = new Instances(data);
        if (!highCardinalityIndices.isEmpty()) {
            System.out.println("Removing high-cardinality nominal attributes before NominalToBinary:");
            for (int idx : highCardinalityIndices) {
                Attribute attr = data.attribute(idx);
                System.out.println("  * " + attr.name());
            }

            // Build a fresh index array using names (to avoid shifting index problems)
            Set<String> highCardNames = new HashSet<>();
            for (int j : highCardinalityIndices) {
                highCardNames.add(data.attribute(j).name());
            }

            ArrayList<Integer> toRemoveInWorking = new ArrayList<>();
            for (int j = 0; j < workingData.numAttributes(); j++) {
                if (highCardNames.contains(workingData.attribute(j).name())) {
                    toRemoveInWorking.add(j);
                }
            }

            int[] removeIdxArray = new int[toRemoveInWorking.size()];
            for (int k = 0; k < toRemoveInWorking.size(); k++) {
                removeIdxArray[k] = toRemoveInWorking.get(k);
            }

            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndicesArray(removeIdxArray);
            removeFilter.setInvertSelection(false);
            removeFilter.setInputFormat(workingData);
            workingData = Filter.useFilter(workingData, removeFilter);

            // Re-set class index (last attribute)
            if (workingData.classIndex() == -1) {
                workingData.setClassIndex(workingData.numAttributes() - 1);
            }
        }

        // 3. Count remaining nominal attributes (excluding class)
        int nominalCount = 0;
        for (int j = 0; j < workingData.numAttributes(); j++) {
            if (workingData.attribute(j).isNominal() && j != workingData.classIndex()) {
                nominalCount++;
            }
        }

        System.out.printf("Found %d categorical attributes (excluding class) after removing high-cardinality ones.%n",
                nominalCount);

        if (nominalCount == 0) {
            System.out.println("No categorical attributes to convert.");
            return workingData;
        }

        // 4. Apply NominalToBinary filter safely
        NominalToBinary nominalToBinary = new NominalToBinary();
        nominalToBinary.setInputFormat(workingData);
        Instances transformedData = Filter.useFilter(workingData, nominalToBinary);

        System.out.printf("Attributes before conversion: %d%n", workingData.numAttributes());
        System.out.printf("Attributes after conversion: %d%n", transformedData.numAttributes());
        System.out.println("Categorical attributes have been converted to binary (0/1) format.");

        return transformedData;
    }

    // --- Final comprehensive status report ---
    private static void printFinalStatusReport(Instances data) {
        System.out.println();
        System.out.println("======================================================================");
        System.out.println("=== FINAL PREPROCESSING STATUS REPORT ===");
        System.out.println("======================================================================");

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
        System.out.println("   - All duplicate rows have been removed.");

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
        System.out.println("   - All numeric attributes normalized to [0, 1].");

        System.out.println("\n6. CATEGORICAL CONVERSION:");
        System.out.println("   - Categorical attributes converted to numerical (binary) where applicable.");

        System.out.println("\n======================================================================");
        System.out.println(" ALL PREPROCESSING STEPS COMPLETED SUCCESSFULLY");
        System.out.println("======================================================================");
    }

    // --- MAIN: handles absolute + relative paths, and auto ARFF naming ---
    public static void main(String[] args) throws Exception {

        // 1. Input CSV: from args or default
        String inputCsv;
        if (args.length > 0) {
            // User gave an absolute or relative CSV path
            inputCsv = args[0];
        } else {
            // Default dataset in project
            inputCsv = "datasets/heart_disease.csv";
        }

        // 2. Output ARFF: from args or auto-generate next to CSV
        String outputArff;
        if (args.length > 1) {
            // User specified ARFF path
            outputArff = args[1];
        } else {
            // Auto-generate ARFF in same folder as CSV
            File inputFile = new File(inputCsv);
            String parent = inputFile.getParent(); // may be null for relative
            String baseName = inputFile.getName().replaceAll("\\.csv$", ""); // remove .csv

            if (parent != null) {
                outputArff = parent + File.separator + baseName + ".arff";
            } else {
                outputArff = baseName + ".arff";
            }
        }

        // 3. Print for clarity
        System.out.println("Input CSV:  " + inputCsv);
        System.out.println("Output ARFF: " + outputArff);

        // 4. Run preprocessing
        preprocess(inputCsv, outputArff);
    }
}
