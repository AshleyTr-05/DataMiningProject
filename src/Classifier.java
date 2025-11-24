import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

import java.util.Random;

public class Classifier {

    public static void main(String[] args) throws Exception {

        // 1. Load cleaned ARFF
        DataSource source = new DataSource("datasets/heart_disease_preprocessed.arff");
        Instances data = source.getDataSet();

        // 2. Set class attribute
        data.setClassIndex(data.numAttributes() - 1);

        // 3. Train a J48 tree
        J48 tree = new J48();
        tree.buildClassifier(data);

        // 4. Evaluate with 10-fold cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(tree, data, 10, new Random(1));

        // 5. Output results
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }
}
