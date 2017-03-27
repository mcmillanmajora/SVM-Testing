import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Created by Angie on 3/2/2017.
 */
public class SVM_Classify {
    private static String kernelType;
    private static int degree;
    private static double gamma;
    private static double coef0;
    private static double rho;
    private static ArrayList<Vector> supportVectors;

    public static void main(String[] args) throws IOException {
        Scanner input = new Scanner(new File("model.5")); //args[1]
        supportVectors = new ArrayList<Vector>();
        ArrayList<String> categories = new ArrayList<String>();
        boolean SV = false;

        while (input.hasNextLine()) {
            String line = input.nextLine();
            if (!line.isEmpty()) {
                String[] info = line.trim().split("\\s+");
                if (info[0].equals("kernel_type")) {
                    kernelType = info[1];
                } else if (info[0].equals("degree")) {
                    degree = Integer.parseInt(info[1]);
                } else if (info[0].equals("gamma")) {
                    gamma = Double.parseDouble(info[1]);
                } else if (info[0].equals("coef0")) {
                    coef0 = Double.parseDouble(info[1]);
                } else if (info[0].equals("rho")) {
                    rho = Double.parseDouble(info[1]);
                } else if (info[0].equals("label")) {
                    for (int i = 1; i < info.length; i++) {
                        categories.add(info[i]);
                    }
                } else if (info[0].equals("SV")){
                    SV = true;
                } else if (SV) {
                    double weight = Double.parseDouble(info[0]);
                    Vector instance = new Vector(weight);
                    for (int i = 1; i < info.length; i++) {
                        String[] feature = info[i].split(":");
                        instance.addFeature(Integer.parseInt(feature[0]), Double.parseDouble(feature[1]));
                    }
                    supportVectors.add(instance);
                }
            }
        }

        Scanner testData = new Scanner(new File("test")); //args[0]
        PrintWriter sysPW = new PrintWriter(new File("sys_output5")); //args[2]
        ModelAccuracy acc = new ModelAccuracy(categories);
        while (testData.hasNextLine()) {
            String line = testData.nextLine();
            if (!line.isEmpty()) {
                String[] vector = line.split("\\s+");
                int goldLabel = Integer.parseInt(vector[0]);
                Vector testInstance = new Vector(1.0);
                for (int i = 1; i < vector.length; i++) {
                    String[] feature = vector[i].split(":");
                    testInstance.addFeature(Integer.parseInt(feature[0]), Double.parseDouble(feature[1]));
                }
                double fOfX = calcVectorSim(testInstance);
                int sysLabel = 0;
                if (fOfX < 0) {
                    sysLabel = 1;
                }
                sysPW.println(goldLabel + " " + sysLabel + " " + fOfX);
                acc.addToTest(sysLabel, goldLabel);
            }
        }
        sysPW.close();
        System.out.println(acc.printTestMatrix());
    }

    public static double calcVectorSim(Vector testInstance) {
        double output = 0.0;
        if (kernelType.equals("linear")) {
            for (int i = 0; i < supportVectors.size(); i++) {
                double curr = linearKernel(testInstance, supportVectors.get(i));
                output += (curr * supportVectors.get(i).getWeight());
            }
        } else if (kernelType.equals("polynomial")) {
            for (int i = 0; i < supportVectors.size(); i++) {
                double curr = polyKernel(testInstance, supportVectors.get(i));
                output += (curr * supportVectors.get(i).getWeight());
            }
        } else if (kernelType.equals("rbf")) {
            for (int i = 0; i < supportVectors.size(); i++) {
                double curr = rbfKernel(testInstance, supportVectors.get(i));
                output += (curr * supportVectors.get(i).getWeight());
            }
        } else if (kernelType.equals("sigmoid")) {
            for (int i = 0; i < supportVectors.size(); i++) {
                double curr = sigmoidKernel(testInstance, supportVectors.get(i));
                output += (curr * supportVectors.get(i).getWeight());
            }
        } else {
            System.err.println("INVALID KERNEL TYPE");
            System.exit(1);
        }
        return output - rho;
    }

    // returns u'*v
    public static double linearKernel(Vector testInstance, Vector supportInstance) {
        double output = 0.0;
        ArrayList<Integer> testFeatures = testInstance.getFeatureList();
        ArrayList<Integer> supportFeatures = supportInstance.getFeatureList();

        int testPointer = 0;
        int supportPointer = 0;
        while (testPointer < testFeatures.size() && supportPointer < supportFeatures.size()) {
            int testFeat = testFeatures.get(testPointer);
            int supportFeat = supportFeatures.get(supportPointer);
            if (testFeat < supportFeat) {
                testPointer++;
            } else if (supportFeat < testFeat) {
                supportPointer++;
            } else {
                output += testInstance.getValue(testPointer) * supportInstance.getValue(supportPointer);
                testPointer++;
                supportPointer++;
            }
        }
        return output;
    }

    // returns (gamma*u'*v + coef0)^degree
    public static double polyKernel(Vector testInstance, Vector supportInstance) {
        double base = linearKernel(testInstance, supportInstance);
        base = (base * gamma) + coef0;
        return Math.pow(base, degree);
    }

    // returns exp(-gamma*(|u'-v|^2))
    public static double rbfKernel(Vector testInstance, Vector supportInstance) {
        ArrayList<Integer> testFeatures = testInstance.getFeatureList();
        ArrayList<Integer> supportFeatures = supportInstance.getFeatureList();
        ArrayList<Integer> resultFeatures = new ArrayList<Integer>();
        int testPointer = 0;
        int supportPointer = 0;
        while (testPointer < testFeatures.size() && supportPointer < supportFeatures.size()) {
            int testFeat = testFeatures.get(testPointer);
            int supportFeat = supportFeatures.get(supportPointer);
            if (testFeat < supportFeat) {
                resultFeatures.add(testFeat);
                testPointer++;
            } else if (supportFeat < testFeat) {
                resultFeatures.add(supportFeat);
                supportPointer++;
            } else {
                resultFeatures.add(testFeat);
                testPointer++;
                supportPointer++;
            }
        }
        double output = Math.pow(resultFeatures.size(), 2);
        output = output * -1 * gamma;
        return Math.exp(output);
    }

    // returns tanh(gamma*u'*v + coef0)
    public static double sigmoidKernel(Vector testInstance, Vector supportInstance) {
        double output = linearKernel(testInstance, supportInstance);
        output = output * gamma;
        output = output + coef0;
        return Math.tanh(output);
    }
}

class Vector {
    double weight;
    ArrayList<Integer> non0featureList;
    ArrayList<Double> featureValues;

    public Vector(double weight) {
        this.weight = weight;
        non0featureList = new ArrayList<Integer>();
        featureValues = new ArrayList<Double>();
    }

    public void addFeature(int index, double value) {
        non0featureList.add(index);
        featureValues.add(value);
    }

    public ArrayList<Double> getFeatureValues() {
        return featureValues;
    }

    public double getValue(int index) {
        return featureValues.get(index);
    }

    public ArrayList<Integer> getFeatureList() {
        return non0featureList;
    }

    public double getWeight() {
        return weight;
    }
}