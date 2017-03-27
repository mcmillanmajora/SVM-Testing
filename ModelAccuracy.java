import java.util.ArrayList;

/**
 * Created by Angie on 2/12/2017.
 */
public class ModelAccuracy {
    private int[][] trainingAcc;
    private int[][] testAcc;
    ArrayList<String> categories;

    public ModelAccuracy(ArrayList<String> categories) {
        this.categories = categories;
        int catSize = categories.size();
        trainingAcc = new int[catSize][catSize];
        testAcc = new int[catSize][catSize];
    }

    public double getTestAcc() {
        return getAcc(testAcc);
    }

    public double getTrainingAcc() {
        return getAcc(trainingAcc);
    }

    // row is the truth, column is the system output
    private void addToMatrix(int col, int row, int[][] matrix) {
        int count = matrix[col][row];
        matrix[col][row] = count + 1;
    }

    public void addToTest(int col, int row) {
        addToMatrix(col, row, testAcc);
    }

    public void addToTraining(int col, int row) {
        addToMatrix(col, row, trainingAcc);
    }

    private double getAcc(int[][] matrix) {
        double correct = 0.0;
        double total = 0.0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                total += matrix[j][i];
                if (i == j) { correct += matrix[j][i]; }
            }
        }
        return correct/total;
    }

    public String printTestMatrix() {
        StringBuilder output = new StringBuilder();
        output.append("Confusion matrix for the test data: \nrow is the truth, column is the system output\n\n");
        output.append(printMatrix(testAcc));
        output.append("\n\nTesting accuracy: ");
        output.append(getTestAcc());
        return output.toString();
    }

    public String printTrainingMatrix() {
        StringBuilder output = new StringBuilder();
        output.append("Confusion matrix for the test data: \nrow is the truth, column is the system output\n\n");
        output.append(printMatrix(trainingAcc));
        output.append("\n\nTesting accuracy: ");
        output.append(getTrainingAcc());
        return output.toString();
    }

    private String printMatrix(int[][] matrix) {
        StringBuilder graph = new StringBuilder();
        for (int i = 0; i < categories.size(); i++) {
            graph.append("\t");
            graph.append(categories.get(i));
        }
        graph.append("\n");
        for (int i = 0; i < categories.size(); i++) {
            graph.append(categories.get(i));
            graph.append("\t");
            for (int j = 0; j < categories.size(); j++) {
                graph.append(matrix[j][i]);
                graph.append("\t");
            }
            graph.append("\n");
        }
        graph.append("\n");
        return graph.toString();
    }
}
