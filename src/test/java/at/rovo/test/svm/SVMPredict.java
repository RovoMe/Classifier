package at.rovo.test.svm;

import at.rovo.classifier.svm.Model;
import at.rovo.classifier.svm.SVMType;
import at.rovo.classifier.svm.struct.Node;
import at.rovo.classifier.svm.struct.PrintInterface;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;

/**
 *
 * @author Chih-Chung Chang, Chih-Jen Lin
 */
class SVMPredict
{
    private static PrintInterface printString = System.out::print;

    private static void info(String s)
    {
        printString.print(s);
    }

    private static double atof(String s)
    {
        return Double.valueOf(s);
    }

    private static int atoi(String s)
    {
        return Integer.parseInt(s);
    }

    private static void predict(BufferedReader input, DataOutputStream output, Model model, int predict_probability)
            throws IOException
    {
        int correct = 0;
        int total = 0;
        double error = 0;
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;

        SVMType svmType = model.getSVMType();
        int nr_class = model.getNrClass();
        double[] prob_estimates = null;

        if (predict_probability == 1)
        {
            if (SVMType.EPSILON_SVR.equals(svmType) || SVMType.NU_SVR.equals(svmType))
            {
                SVMPredict
                        .info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=" +
                              model.getSVRProbability() + "\n");
            }
            else
            {
                int[] labels = new int[nr_class];
                model.getLabels(labels);
                prob_estimates = new double[nr_class];
                output.writeBytes("labels");
                for (int j = 0; j < nr_class; j++)
                {
                    output.writeBytes(" " + labels[j]);
                }
                output.writeBytes("\n");
            }
        }
        while (true)
        {
            String line = input.readLine();
            if (line == null)
            {
                break;
            }

            StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

            double target = atof(st.nextToken());
            int m = st.countTokens() / 2;
            Node[] x = new Node[m];
            for (int j = 0; j < m; j++)
            {
                x[j] = new Node();
                x[j].index = atoi(st.nextToken());
                x[j].value = atof(st.nextToken());
            }

            double v;
            if (predict_probability == 1 && (SVMType.C_SVC.equals(svmType) || SVMType.NU_SVC.equals(svmType)))
            {
                v = model.predictProbability(x, prob_estimates);
                output.writeBytes(v + " ");
                for (int j = 0; j < nr_class; j++)
                {
                    output.writeBytes(prob_estimates[j] + " ");
                }
                output.writeBytes("\n");
            }
            else
            {
                v = model.predict(x);
                output.writeBytes(v + "\n");
            }

            if (v == target)
            {
                ++correct;
            }
            error += (v - target) * (v - target);
            sumv += v;
            sumy += target;
            sumvv += v * v;
            sumyy += target * target;
            sumvy += v * target;
            ++total;
        }
        if (SVMType.EPSILON_SVR.equals(svmType) || SVMType.NU_SVR.equals(svmType))
        {
            SVMPredict.info("Mean squared error = " + error / total + " (regression)\n");
            SVMPredict.info("Squared correlation coefficient = " +
                            ((total * sumvy - sumv * sumy) * (total * sumvy - sumv * sumy)) /
                            ((total * sumvv - sumv * sumv) * (total * sumyy - sumy * sumy)) + " (regression)\n");
        }
        else
        {
            SVMPredict.info("Accuracy = " + (double) correct / total * 100 + "% (" + correct + "/" + total +
                            ") (classification)\n");
        }
    }

    private static void exitWithHelp()
    {
        System.err.print("usage: SVMPredict [options] test_file model_file output_file\n" + "options:\n" +
                         "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n" +
                         "-q : quiet mode (no outputs)\n");
        System.exit(1);
    }

    public static void main(String argv[]) throws IOException
    {
        int i, predict_probability = 0;
        printString = System.out::print;

        // parse options
        for (i = 0; i < argv.length; i++)
        {
            if (argv[i].charAt(0) != '-')
            {
                break;
            }
            ++i;
            switch (argv[i - 1].charAt(1))
            {
                case 'b':
                    predict_probability = atoi(argv[i]);
                    break;
                case 'q':
                    printString = (String s) -> {};
                    i--;
                    break;
                default:
                    System.err.print("Unknown option: " + argv[i - 1] + "\n");
                    exitWithHelp();
            }
        }
        if (i >= argv.length - 2)
        {
            exitWithHelp();
        }
        try
        {
            BufferedReader input = new BufferedReader(new FileReader(argv[i]));
            DataOutputStream output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(argv[i + 2])));
            Model model = Model.load(argv[i + 1]);
            if (predict_probability == 1)
            {
                if (null == model || model.checkProbabilityModel() == 0)
                {
                    System.err.print("Model does not support probability estimates\n");
                    System.exit(1);
                }
            }
            else
            {
                if (null != model && model.checkProbabilityModel() != 0)
                {
                    SVMPredict.info("Model supports probability estimates, but disabled in prediction.\n");
                }
            }
            predict(input, output, model, predict_probability);
            input.close();
            output.close();
        }
        catch (FileNotFoundException | ArrayIndexOutOfBoundsException e)
        {
            exitWithHelp();
        }
    }
}
