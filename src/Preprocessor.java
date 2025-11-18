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

        // print Missing and Zero values report
        printMissingAndZeroReport(data);

        // save to ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputArffPath));
        saver.writeBatch();

        System.out.println();
        System.out.println("===ARFF file saved===");
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

            System.out.printf("Attribute: %-15s | Missing: %4d | Zero values (numeric only): %4d%n",
                    attr.name(), missingCount, (attr.isNumeric() ? zeroCount : 0));
        }
        System.out.println();
        System.out.println("Note: Zero values might or might not be invalid.");
        System.out.println("      Later we will decide which attributes treat 0 as 'missing'.");
    }

    public static void main(String[] args) throws Exception {
        String inputCsv = "datasets/heart_disease.csv";
        String outputArff = "datasets/heart_disease_preprocessed.arff";

        preprocess(inputCsv, outputArff);
    }
}
