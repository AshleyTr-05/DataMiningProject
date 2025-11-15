import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;

import java.io.File;

public class Preprocessor {

    public static String preprocess(String inputCsvPath, String outputArffPath) throws Exception {

        // 1. Load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(inputCsvPath));  // <-- datasets/heart_disease.csv
        Instances data = loader.getDataSet();

        // 2. Set class attribute to last column
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // 3. Save to ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputArffPath));
        saver.writeBatch();

        // 4. Print info to prove it worked
        System.out.println("=== Preprocessing done ===");
        System.out.println("Input CSV  : " + inputCsvPath);
        System.out.println("Output ARFF: " + outputArffPath);
        System.out.println("Instances  : " + data.numInstances());
        System.out.println("Attributes : " + data.numAttributes());
        System.out.println("Class attr : " + data.classAttribute().name());
            
        System.out.println("First 3 rows:");
        for (int i = 0; i < Math.min(3, data.numInstances()); i++) {
            System.out.println("  " + data.instance(i));
        }

        return outputArffPath;
    }

    public static void main(String[] args) throws Exception {
        String inputCsv  = "datasets/heart_disease.csv";
        String outputArff = "datasets/heart_disease_preprocessed.arff";

        preprocess(inputCsv, outputArff);
    }
}
