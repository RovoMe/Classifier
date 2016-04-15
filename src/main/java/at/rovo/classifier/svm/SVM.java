package at.rovo.classifier.svm;

import at.rovo.classifier.Classifier;
import at.rovo.classifier.svm.solver.instance.CSVC;
import at.rovo.classifier.svm.solver.instance.EpsilonSVR;
import at.rovo.classifier.svm.solver.instance.NuSVC;
import at.rovo.classifier.svm.solver.instance.NuSVR;
import at.rovo.classifier.svm.solver.instance.OneClass;
import at.rovo.classifier.svm.solver.instance.SolveInstance;
import at.rovo.classifier.svm.struct.DecisionFunction;
import at.rovo.classifier.svm.struct.Node;
import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.Problem;
import at.rovo.classifier.svm.struct.SolutionInfo;
import at.rovo.classifier.svm.utils.Utils;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * A Support Vector Machine (SVM) performs classification by constructing an N-dimensional hyperplane that optimally
 * separates the data into two categories. SVM models are closely related to neural networks. In fact, a SVM model using
 * a sigmoid kernel function is equivalent to a two-layer, perceptron neural network.
 * <p>
 * Support Vector Machine (SVM) models are a close cousin to classical multilayer perceptron neural networks. Using a
 * kernel function, SVM’s are an alternative training method for polynomial, radial basis function and multi-layer
 * perceptron classifiers in which the weights of the network are found by solving a quadratic programming problem with
 * linear constraints, rather than by solving a non-convex, unconstrained minimization problem as in standard neural
 * network training.
 * <p>
 * In the parlance of SVM literature, a predictor variable is called an attribute, and a transformed attribute that is
 * used to define the hyperplane is called a feature. The task of choosing the most suitable representation is known as
 * feature selection. A set of features that describes one case (i.e., a row of predictor values) is called a vector. So
 * the goal of SVM modeling is to find the optimal hyperplane that separates clusters of vector in such a way that cases
 * with one category of the target variable are on one side of the plane and cases with the other category are on the
 * other size of the plane. The vectors near the hyperplane are the support vectors.
 * <p>
 * This implementation is taken from libSVM and got modified to fit into the available classification framework. The
 * original implementation was done by Chih-Chung Chang, Chih-Jen Lin.
 *
 * @author Chih-Chung Chang, Chih-Jen Lin, Roman Vottner
 * @link http://www.csie.ntu.edu.tw/~cjlin/libsvm/
 */
public class SVM extends Classifier<Node[], Double>
{
    /** A static logger instance **/
    protected static Logger logger = LogManager.getLogger(SVM.class.getName());
    /** The version of libSVM which was used as base source **/
    public static final int LIBSVM_VERSION = 317;
    /** A random number generator **/
    public static final Random rand = new Random();
    /** The parameters passed to the application **/
    private Parameter param = null;
    /** The training data **/
    private Problem prob = null;
    /** The trained model based on the training data **/
    private Model model = null;

    /**
     * Initializes a new support vector machine.
     *
     * @param param
     *         The parameters necessary to initialize the appropriate type of SVM used to train and classify data
     */
    public SVM(Parameter param)
    {
        this.param = param;
        this.prob = new Problem();
    }

    /**
     * Platt's binary SVM Probablistic Output: an improvement from Lin et al.
     *
     * @param l
     * @param dec_values
     * @param labels
     * @param probAB
     */
    private void sigmoidTrain(int l, double[] dec_values, List<Double> labels, double[] probAB)
    {
        double A, B;
        double prior1 = 0, prior0 = 0;
        int i;

        for (i = 0; i < l; i++)
        {
            if (labels.get(i) > 0)
            {
                prior1 += 1;
            }
            else
            {
                prior0 += 1;
            }
        }

        int max_iter = 100; // Maximal number of iterations
        double min_step = 1e-10; // Minimal step taken in line search
        double sigma = 1e-12; // For numerically strict PD of Hessian
        double eps = 1e-5;
        double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
        double loTarget = 1 / (prior0 + 2.0);
        double[] t = new double[l];
        double fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize;
        double newA, newB, newf, d1, d2;
        int iter;

        // Initial Point and Initial Fun Value
        A = 0.0;
        B = Math.log((prior0 + 1.0) / (prior1 + 1.0));
        double fval = 0.0;

        for (i = 0; i < l; i++)
        {
            if (labels.get(i) > 0)
            {
                t[i] = hiTarget;
            }
            else
            {
                t[i] = loTarget;
            }
            fApB = dec_values[i] * A + B;
            if (fApB >= 0)
            {
                fval += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
            }
            else
            {
                fval += (t[i] - 1) * fApB + Math.log(1 + Math.exp(fApB));
            }
        }
        for (iter = 0; iter < max_iter; iter++)
        {
            // Update Gradient and Hessian (use H' = H + sigma I)
            h11 = sigma; // numerically ensures strict PD
            h22 = sigma;
            h21 = 0.0;
            g1 = 0.0;
            g2 = 0.0;
            for (i = 0; i < l; i++)
            {
                fApB = dec_values[i] * A + B;
                if (fApB >= 0)
                {
                    p = Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
                    q = 1.0 / (1.0 + Math.exp(-fApB));
                }
                else
                {
                    p = 1.0 / (1.0 + Math.exp(fApB));
                    q = Math.exp(fApB) / (1.0 + Math.exp(fApB));
                }
                d2 = p * q;
                h11 += dec_values[i] * dec_values[i] * d2;
                h22 += d2;
                h21 += dec_values[i] * d2;
                d1 = t[i] - p;
                g1 += dec_values[i] * d1;
                g2 += d1;
            }

            // Stopping Criteria
            if (Math.abs(g1) < eps && Math.abs(g2) < eps)
            {
                break;
            }

            // Finding Newton direction: -inv(H') * g
            det = h11 * h22 - h21 * h21;
            dA = -(h22 * g1 - h21 * g2) / det;
            dB = -(-h21 * g1 + h11 * g2) / det;
            gd = g1 * dA + g2 * dB;

            stepsize = 1; // Line Search
            while (stepsize >= min_step)
            {
                newA = A + stepsize * dA;
                newB = B + stepsize * dB;

                // New function value
                newf = 0.0;
                for (i = 0; i < l; i++)
                {
                    fApB = dec_values[i] * newA + newB;
                    if (fApB >= 0)
                    {
                        newf += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
                    }
                    else
                    {
                        newf += (t[i] - 1) * fApB + Math.log(1 + Math.exp(fApB));
                    }
                }
                // Check sufficient decrease
                if (newf < fval + 0.0001 * stepsize * gd)
                {
                    A = newA;
                    B = newB;
                    fval = newf;
                    break;
                }
                else
                {
                    stepsize = stepsize / 2.0;
                }
            }

            if (stepsize < min_step)
            {
                if (logger.isDebugEnabled())
                {
                    logger.debug("Line search fails in two-class probability estimates\n");
                }
                break;
            }
        }

        if (iter >= max_iter)
        {
            if (logger.isDebugEnabled())
            {
                logger.debug("Reaching maximal iterations in two-class probability estimates\n");
            }
        }
        probAB[0] = A;
        probAB[1] = B;
    }

    /**
     * Cross-validation decision values for probability estimates.
     *
     * @param prob
     * @param param
     * @param Cp
     * @param Cn
     * @param probAB
     */
    private void binarySVCProbability(Problem prob, Parameter param, double Cp, double Cn, double[] probAB)
    {
        int i;
        int nr_fold = 5;
        int[] perm = new int[prob.numInstances];
        double[] dec_values = new double[prob.numInstances];

        // random shuffle
        for (i = 0; i < prob.numInstances; i++)
        {
            perm[i] = i;
        }
        for (i = 0; i < prob.numInstances; i++)
        {
            int j = i + rand.nextInt(prob.numInstances - i);
            do
            {
                Utils.swap(perm, i, j);
            }
            while (false);
        }
        for (i = 0; i < nr_fold; i++)
        {
            int begin = i * prob.numInstances / nr_fold;
            int end = (i + 1) * prob.numInstances / nr_fold;
            int j, k;
            Problem subprob = new Problem();

            subprob.numInstances = prob.numInstances - (end - begin);
            subprob.x = new ArrayList<>(subprob.numInstances);
            subprob.y = new ArrayList<>(subprob.numInstances);

            k = 0;
            for (j = 0; j < begin; j++)
            {
                subprob.x.add(prob.x.get(perm[j]));
                subprob.y.add(prob.y.get(perm[j]));
                ++k;
            }
            for (j = end; j < prob.numInstances; j++)
            {
                subprob.x.add(prob.x.get(perm[j]));
                subprob.y.add(prob.y.get(perm[j]));
                ++k;
            }
            int p_count = 0, n_count = 0;
            for (j = 0; j < k; j++)
            {
                if (subprob.y.get(j) > 0)
                {
                    p_count++;
                }
                else
                {
                    n_count++;
                }
            }

            if (p_count == 0 && n_count == 0)
            {
                for (j = begin; j < end; j++)
                {
                    dec_values[perm[j]] = 0;
                }
            }
            else if (p_count > 0 && n_count == 0)
            {
                for (j = begin; j < end; j++)
                {
                    dec_values[perm[j]] = 1;
                }
            }
            else if (p_count == 0 && n_count > 0)
            {
                for (j = begin; j < end; j++)
                {
                    dec_values[perm[j]] = -1;
                }
            }
            else
            {
                Parameter subparam = (Parameter) param.clone();
                subparam.probability = 0;
                subparam.C = 1.0;
                subparam.nrWeight = 2;
                subparam.weightLabel = new int[2];
                subparam.weight = new double[2];
                subparam.weightLabel[0] = +1;
                subparam.weightLabel[1] = -1;
                subparam.weight[0] = Cp;
                subparam.weight[1] = Cn;
                Model submodel = this.train(subprob, subparam);
                for (j = begin; j < end; j++)
                {
                    double[] dec_value = new double[1];
                    submodel.predictValues(prob.x.get(perm[j]), dec_value);
                    dec_values[perm[j]] = dec_value[0];
                    // ensure +1 -1 order; reason not using CV subroutine
                    dec_values[perm[j]] *= submodel.label[0];
                }
            }
        }
        sigmoidTrain(prob.numInstances, dec_values, prob.y, probAB);
    }

    /**
     * @param prob
     * @param param
     *
     * @return The parameter of a Laplace distribution
     */
    private double svrProbability(Problem prob, Parameter param)
    {
        int i;
        int nr_fold = 5;
        double[] ymv = new double[prob.numInstances];
        double mae = 0;

        Parameter newparam = (Parameter) param.clone();
        newparam.probability = 0;
        this.crossValidation(newparam, nr_fold, ymv);
        for (i = 0; i < prob.numInstances; i++)
        {
            ymv[i] = prob.y.get(i) - ymv[i];
            mae += Math.abs(ymv[i]);
        }
        mae /= prob.numInstances;
        double std = Math.sqrt(2 * mae * mae);
        int count = 0;
        mae = 0;
        for (i = 0; i < prob.numInstances; i++)
        {
            if (Math.abs(ymv[i]) > 5 * std)
            {
                count = count + 1;
            }
            else
            {
                mae += Math.abs(ymv[i]);
            }
        }
        mae /= (prob.numInstances - count);
        if (logger.isDebugEnabled())
        {
            logger.debug("Prob. model for test data: target value = predicted value " +
                         "+ z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=" + mae + "\n");
        }
        return mae;
    }

    /**
     * label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
     * <p>
     * perm, length l, must be allocated before calling this subroutine
     *
     * @param prob
     * @param nr_class_ret
     * @param label_ret
     * @param start_ret
     * @param count_ret
     * @param perm
     */

    private void groupClasses(Problem prob, int[] nr_class_ret, int[][] label_ret, int[][] start_ret, int[][] count_ret,
                              int[] perm)
    {
        int l = prob.numInstances;
        int max_nr_class = 16;
        int nr_class = 0;
        int[] label = new int[max_nr_class];
        int[] count = new int[max_nr_class];
        int[] data_label = new int[l];
        int i;

        for (i = 0; i < l; i++)
        {
            int this_label = (prob.y.get(i)).intValue();
            int j;
            for (j = 0; j < nr_class; j++)
            {
                if (this_label == label[j])
                {
                    ++count[j];
                    break;
                }
            }
            data_label[i] = j;
            if (j == nr_class)
            {
                if (nr_class == max_nr_class)
                {
                    max_nr_class *= 2;
                    int[] new_data = new int[max_nr_class];
                    System.arraycopy(label, 0, new_data, 0, label.length);
                    label = new_data;
                    new_data = new int[max_nr_class];
                    System.arraycopy(count, 0, new_data, 0, count.length);
                    count = new_data;
                }
                label[nr_class] = this_label;
                count[nr_class] = 1;
                ++nr_class;
            }
        }

        // Labels are ordered by their first occurrence in the training set.
        // However, for two-class sets with -1/+1 labels and -1 appears first,
        // we swap labels to ensure that internally the binary SVM has positive
        // data corresponding to the +1 instances.
        if (nr_class == 2 && label[0] == -1 && label[1] == +1)
        {
            do
            {
                Utils.swap(label, 0, 1);
            }
            while (false);
            do
            {
                Utils.swap(count, 0, 1);
            }
            while (false);
            for (i = 0; i < l; i++)
            {
                if (data_label[i] == 0)
                {
                    data_label[i] = 1;
                }
                else
                {
                    data_label[i] = 0;
                }
            }
        }

        int[] start = new int[nr_class];
        start[0] = 0;
        for (i = 1; i < nr_class; i++)
        {
            start[i] = start[i - 1] + count[i - 1];
        }
        for (i = 0; i < l; i++)
        {
            perm[start[data_label[i]]] = i;
            ++start[data_label[i]];
        }
        start[0] = 0;
        for (i = 1; i < nr_class; i++)
        {
            start[i] = start[i - 1] + count[i - 1];
        }

        nr_class_ret[0] = nr_class;
        label_ret[0] = label;
        start_ret[0] = start;
        count_ret[0] = count;
    }

    /**
     * Trains a model based on the available data in the problem according to the parameters provided.
     * <p>
     * Training a support vector machine consists of finding the optimal hyperplane, that is, the one with the maximum
     * distance from the nearest training patterns. The support vectors are these nearest to the hyperplane.
     *
     * @param prob
     *         The training data
     * @param param
     *         The parameters provided to distinguish what kind of model should be build
     *
     * @return The trained model
     */
    private Model train(Problem prob, Parameter param)
    {
        Model model = new Model();
        model.param = param;

        if (SVMType.ONE_CLASS.equals(param.svmType) || SVMType.EPSILON_SVR.equals(param.svmType) ||
            SVMType.NU_SVR.equals(param.svmType))
        {
            // regression or one-class-svm
            model.nrClass = 2;
            model.label = null;
            model.nSV = null;
            model.probA = null;
            model.probB = null;
            model.svCoef = new double[1][];

            if (param.probability == 1 &&
                (SVMType.EPSILON_SVR.equals(param.svmType) || SVMType.NU_SVR.equals(param.svmType)))
            {
                model.probA = new double[1];
                model.probA[0] = svrProbability(prob, param);
            }

            DecisionFunction f = trainOne(prob, param, 0, 0);
            model.rho = new double[1];
            model.rho[0] = f.rho;

            int nSV = 0;
            int i;
            for (i = 0; i < prob.numInstances; i++)
            {
                if (Math.abs(f.alpha[i]) > 0)
                {
                    ++nSV;
                }
            }
            model.numInstances = nSV;
            model.SV = new Node[nSV][];
            model.svCoef[0] = new double[nSV];
            model.svIndices = new int[nSV];
            int j = 0;
            for (i = 0; i < prob.numInstances; i++)
            {
                if (Math.abs(f.alpha[i]) > 0)
                {
                    model.SV[j] = prob.x.get(i);
                    model.svCoef[0][j] = f.alpha[i];
                    model.svIndices[j] = i + 1;
                    ++j;
                }
            }
        }
        else
        {
            // classification
            int l = prob.numInstances;
            int[] tmp_nr_class = new int[1];
            int[][] tmp_label = new int[1][];
            int[][] tmp_start = new int[1][];
            int[][] tmp_count = new int[1][];
            int[] perm = new int[l];

            // group training data of the same class
            groupClasses(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count, perm);
            int nr_class = tmp_nr_class[0];
            int[] label = tmp_label[0];
            int[] start = tmp_start[0];
            int[] count = tmp_count[0];

            if (nr_class == 1)
            {
                if (logger.isDebugEnabled())
                {
                    logger.debug("WARNING: training data in only one class. See README for details.\n");
                }
            }

            Node[][] x = new Node[l][];
            int i;
            for (i = 0; i < l; i++)
            {
                x[i] = prob.x.get(perm[i]);
            }

            // calculate weighted C
            double[] weighted_C = new double[nr_class];
            for (i = 0; i < nr_class; i++)
            {
                weighted_C[i] = param.C;
            }
            for (i = 0; i < param.nrWeight; i++)
            {
                int j;
                for (j = 0; j < nr_class; j++)
                {
                    if (param.weightLabel[i] == label[j])
                    {
                        break;
                    }
                }
                if (j == nr_class)
                {
                    System.err.print("WARNING: class label " + param.weightLabel[i] +
                                     " specified in weight is not found\n");
                }
                else
                {
                    weighted_C[j] *= param.weight[i];
                }
            }

            // train k*(k-1)/2 models
            boolean[] nonzero = new boolean[l];
            for (i = 0; i < l; i++)
            {
                nonzero[i] = false;
            }
            DecisionFunction[] f = new DecisionFunction[nr_class * (nr_class - 1) / 2];

            double[] probA = null, probB = null;
            if (param.probability == 1)
            {
                probA = new double[nr_class * (nr_class - 1) / 2];
                probB = new double[nr_class * (nr_class - 1) / 2];
            }

            int p = 0;
            for (i = 0; i < nr_class; i++)
            {
                for (int j = i + 1; j < nr_class; j++)
                {
                    Problem sub_prob = new Problem();
                    int si = start[i], sj = start[j];
                    int ci = count[i], cj = count[j];
                    sub_prob.numInstances = ci + cj;
                    sub_prob.x = new ArrayList<>(sub_prob.numInstances);
                    sub_prob.y = new ArrayList<>(sub_prob.numInstances);
                    int k;
                    for (k = 0; k < ci; k++)
                    {
                        sub_prob.x.add(x[si + k]);
                        sub_prob.y.add(+1.);
                    }
                    for (k = 0; k < cj; k++)
                    {
                        sub_prob.x.add(x[sj + k]);
                        sub_prob.y.add(-1.);
                    }

                    if (param.probability == 1)
                    {
                        double[] probAB = new double[2];
                        binarySVCProbability(sub_prob, param, weighted_C[i], weighted_C[j], probAB);
                        probA[p] = probAB[0];
                        probB[p] = probAB[1];
                    }

                    f[p] = trainOne(sub_prob, param, weighted_C[i], weighted_C[j]);
                    for (k = 0; k < ci; k++)
                    {
                        if (!nonzero[si + k] && Math.abs(f[p].alpha[k]) > 0)
                        {
                            nonzero[si + k] = true;
                        }
                    }
                    for (k = 0; k < cj; k++)
                    {
                        if (!nonzero[sj + k] && Math.abs(f[p].alpha[ci + k]) > 0)
                        {
                            nonzero[sj + k] = true;
                        }
                    }
                    ++p;
                }
            }

            // build output
            model.nrClass = nr_class;

            model.label = new int[nr_class];
            for (i = 0; i < nr_class; i++)
            {
                model.label[i] = label[i];
            }

            model.rho = new double[nr_class * (nr_class - 1) / 2];
            for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
            {
                model.rho[i] = f[i].rho;
            }

            if (param.probability == 1)
            {
                model.probA = new double[nr_class * (nr_class - 1) / 2];
                model.probB = new double[nr_class * (nr_class - 1) / 2];
                for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
                {
                    model.probA[i] = probA[i];
                    model.probB[i] = probB[i];
                }
            }
            else
            {
                model.probA = null;
                model.probB = null;
            }

            int nnz = 0;
            int[] nz_count = new int[nr_class];
            model.nSV = new int[nr_class];
            for (i = 0; i < nr_class; i++)
            {
                int nSV = 0;
                for (int j = 0; j < count[i]; j++)
                {
                    if (nonzero[start[i] + j])
                    {
                        ++nSV;
                        ++nnz;
                    }
                }
                model.nSV[i] = nSV;
                nz_count[i] = nSV;
            }

            if (logger.isDebugEnabled())
            {
                logger.debug("Total nSV = " + nnz + "\n");
            }

            model.numInstances = nnz;
            model.SV = new Node[nnz][];
            model.svIndices = new int[nnz];
            p = 0;
            for (i = 0; i < l; i++)
            {
                if (nonzero[i])
                {
                    model.SV[p] = x[i];
                    model.svIndices[p++] = perm[i] + 1;
                }
            }

            int[] nz_start = new int[nr_class];
            nz_start[0] = 0;
            for (i = 1; i < nr_class; i++)
            {
                nz_start[i] = nz_start[i - 1] + nz_count[i - 1];
            }

            model.svCoef = new double[nr_class - 1][];
            for (i = 0; i < nr_class - 1; i++)
            {
                model.svCoef[i] = new double[nnz];
            }

            p = 0;
            for (i = 0; i < nr_class; i++)
            {
                for (int j = i + 1; j < nr_class; j++)
                {
                    // classifier (i,j): coefficients with
                    // i are in sv_coef[j-1][nz_start[i]...],
                    // j are in sv_coef[i][nz_start[j]...]

                    int si = start[i];
                    int sj = start[j];
                    int ci = count[i];
                    int cj = count[j];

                    int q = nz_start[i];
                    int k;
                    for (k = 0; k < ci; k++)
                    {
                        if (nonzero[si + k])
                        {
                            model.svCoef[j - 1][q++] = f[p].alpha[k];
                        }
                    }
                    q = nz_start[j];
                    for (k = 0; k < cj; k++)
                    {
                        if (nonzero[sj + k])
                        {
                            model.svCoef[i][q++] = f[p].alpha[ci + k];
                        }
                    }
                    ++p;
                }
            }
        }
        return model;
    }

    /**
     * Invokes the appropriate kernel function for the training set according to the provided parameter.
     *
     * @param prob
     *         The training data
     * @param param
     *         The parameters passed to the application
     * @param Cp
     * @param Cn
     *
     * @return A decision function
     */
    DecisionFunction trainOne(Problem prob, Parameter param, double Cp, double Cn)
    {
        double[] alpha = new double[prob.numInstances];
        SolutionInfo si = new SolutionInfo();
        SolveInstance instance;
        switch (param.svmType)
        {
            case C_SVC:
                instance = new CSVC(Cp, Cn);
                instance.solve(prob, param, alpha, si);
                break;
            case NU_SVC:
                instance = new NuSVC();
                instance.solve(prob, param, alpha, si);
                break;
            case ONE_CLASS:
                instance = new OneClass();
                instance.solve(prob, param, alpha, si);
                break;
            case EPSILON_SVR:
                instance = new EpsilonSVR();
                instance.solve(prob, param, alpha, si);
                break;
            case NU_SVR:
                instance = new NuSVR();
                instance.solve(prob, param, alpha, si);
                break;
        }

        if (logger.isDebugEnabled())
        {
            logger.debug("obj = " + si.obj + ", rho = " + si.rho + "\n");

            // output SVs
            int nSV = 0; // number of support vectors
            int nBSV = 0; // number of bounded support vectors
            for (int i = 0; i < prob.numInstances; i++)
            {
                if (Math.abs(alpha[i]) > 0)
                {
                    ++nSV;
                    if (prob.y.get(i) > 0)
                    {
                        if (Math.abs(alpha[i]) >= si.upper_bound_p)
                        {
                            ++nBSV;
                        }
                    }
                    else
                    {
                        if (Math.abs(alpha[i]) >= si.upper_bound_n)
                        {
                            ++nBSV;
                        }
                    }
                }
            }

            logger.debug("nSV = " + nSV + ", nBSV = " + nBSV + "\n");
        }

        DecisionFunction f = new DecisionFunction();
        f.alpha = alpha;
        f.rho = si.rho;
        return f;
    }

    /**
     * Performs a cross-validation for the current trained model.
     *
     * @param param
     *         The parameters specifying how the cross-validation should be performed
     */
    public void crossValidation(Parameter param)
    {
        int i;
        int total_correct = 0;
        double total_error = 0;
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
        double[] target = new double[prob.numInstances];

        this.crossValidation(param, param.nrFold, target);
        if (SVMType.EPSILON_SVR.equals(param.svmType) || SVMType.NU_SVR.equals(param.svmType))
        {
            for (i = 0; i < prob.numInstances; i++)
            {
                double y = prob.y.get(i);
                double v = target[i];
                total_error += (v - y) * (v - y);
                sumv += v;
                sumy += y;
                sumvv += v * v;
                sumyy += y * y;
                sumvy += v * y;
            }
            System.out.print("Cross Validation Mean squared error = " + total_error / prob.numInstances + "\n");
            System.out.print("Cross Validation Squared correlation coefficient = " +
                             ((prob.numInstances * sumvy - sumv * sumy) * (prob.numInstances * sumvy - sumv * sumy)) /
                             ((prob.numInstances * sumvv - sumv * sumv) * (prob.numInstances * sumyy - sumy * sumy)) +
                             "\n");
        }
        else
        {
            for (i = 0; i < prob.numInstances; i++)
            {
                if (target[i] == prob.y.get(i))
                {
                    ++total_correct;
                }
            }
            System.out.print("Cross Validation Accuracy = " + 100.0 * total_correct / prob.numInstances + "%\n");
        }
    }

    /**
     * Performs the actual cross validation.
     *
     * @param param
     *         The parameters specifying how the cross-validation should be performed
     * @param nr_fold
     * @param target
     */
    private void crossValidation(Parameter param, int nr_fold, double[] target)
    {
        int i;
        int[] fold_start = new int[nr_fold + 1];
        int l = this.prob.numInstances;
        int[] perm = new int[l];

        // stratified cv may not give leave-one-out rate
        // Each class to l folds -> some folds may have zero elements
        if ((SVMType.C_SVC.equals(param.svmType) || SVMType.NU_SVC.equals(param.svmType)) && nr_fold < l)
        {
            int[] tmp_nr_class = new int[1];
            int[][] tmp_label = new int[1][];
            int[][] tmp_start = new int[1][];
            int[][] tmp_count = new int[1][];

            groupClasses(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count, perm);

            int nr_class = tmp_nr_class[0];
            int[] start = tmp_start[0];
            int[] count = tmp_count[0];

            // random shuffle and then data grouped by fold using the array perm
            int[] fold_count = new int[nr_fold];
            int c;
            int[] index = new int[l];
            for (i = 0; i < l; i++)
            {
                index[i] = perm[i];
            }
            for (c = 0; c < nr_class; c++)
            {
                for (i = 0; i < count[c]; i++)
                {
                    int j = i + rand.nextInt(count[c] - i);
                    do
                    {
                        Utils.swap(index, start[c] + j, start[c] + i);
                    }
                    while (false);
                }
            }
            for (i = 0; i < nr_fold; i++)
            {
                fold_count[i] = 0;
                for (c = 0; c < nr_class; c++)
                {
                    fold_count[i] += (i + 1) * count[c] / nr_fold - i * count[c] / nr_fold;
                }
            }
            fold_start[0] = 0;
            for (i = 1; i <= nr_fold; i++)
            {
                fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
            }
            for (c = 0; c < nr_class; c++)
            {
                for (i = 0; i < nr_fold; i++)
                {
                    int begin = start[c] + i * count[c] / nr_fold;
                    int end = start[c] + (i + 1) * count[c] / nr_fold;
                    for (int j = begin; j < end; j++)
                    {
                        perm[fold_start[i]] = index[j];
                        fold_start[i]++;
                    }
                }
            }
            fold_start[0] = 0;
            for (i = 1; i <= nr_fold; i++)
            {
                fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
            }
        }
        else
        {
            for (i = 0; i < l; i++)
            {
                perm[i] = i;
            }
            for (i = 0; i < l; i++)
            {
                int j = i + rand.nextInt(l - i);
                do
                {
                    Utils.swap(perm, i, j);
                }
                while (false);
            }
            for (i = 0; i <= nr_fold; i++)
            {
                fold_start[i] = i * l / nr_fold;
            }
        }

        for (i = 0; i < nr_fold; i++)
        {
            int begin = fold_start[i];
            int end = fold_start[i + 1];
            int j;
            Problem subprob = new Problem();

            subprob.numInstances = l - (end - begin);
            subprob.x = new ArrayList<>(subprob.numInstances);
            subprob.y = new ArrayList<>(subprob.numInstances);

            for (j = 0; j < begin; j++)
            {
                subprob.x.add(prob.x.get(perm[j]));
                subprob.y.add(prob.y.get(perm[j]));
            }
            for (j = end; j < l; j++)
            {
                subprob.x.add(prob.x.get(perm[j]));
                subprob.y.add(prob.y.get(perm[j]));
            }
            Model submodel = this.train(subprob, param);
            if (param.probability == 1 && (SVMType.C_SVC.equals(param.svmType) || SVMType.NU_SVC.equals(param.svmType)))
            {
                double[] prob_estimates = new double[submodel.getNrClass()];
                for (j = begin; j < end; j++)
                {
                    target[perm[j]] = submodel.predictProbability(prob.x.get(perm[j]), prob_estimates);
                }
            }
            else
            {
                for (j = begin; j < end; j++)
                {
                    target[perm[j]] = submodel.predict(prob.x.get(perm[j]));
                }
            }
        }
    }

    /**
     * Checks the provided parameters for their feasibility and returns the appropriate error message in case a certain
     * parameter or the problem statement is not feasible.
     *
     * @return
     */
    String checkParameter()
    {
        // svm_type
        SVMType svmType = this.param.svmType;
        if (!SVMType.C_SVC.equals(svmType) && !SVMType.NU_SVC.equals(svmType) && !SVMType.ONE_CLASS.equals(svmType) &&
            !SVMType.EPSILON_SVR.equals(svmType) && !SVMType.NU_SVR.equals(svmType))
        {
            return "unknown svm type";
        }

        // kernel_type, degree
        KernelType kernelType = param.kernelType;
        if (KernelType.LINEAR.equals(kernelType) && KernelType.POLYNOMIAL.equals(kernelType) &&
            KernelType.RBF.equals(kernelType) && KernelType.SIGMOID.equals(kernelType) &&
            KernelType.PRECOMPUTED.equals(kernelType))
        {
            return "unknown kernel type";
        }

        if (this.param.gamma < 0)
        {
            return "gamma < 0";
        }

        if (this.param.degree < 0)
        {
            return "degree of polynomial kernel < 0";
        }

        // cache_size,eps,C,nu,p,shrinking
        if (this.param.cache_size <= 0)
        {
            return "cache_size <= 0";
        }

        if (this.param.eps <= 0)
        {
            return "eps <= 0";
        }

        if (SVMType.C_SVC.equals(svmType) || SVMType.EPSILON_SVR.equals(svmType) || SVMType.NU_SVR.equals(svmType))
        {
            if (this.param.C <= 0)
            {
                return "C <= 0";
            }
        }

        if (SVMType.NU_SVC.equals(svmType) || SVMType.ONE_CLASS.equals(svmType) || SVMType.NU_SVR.equals(svmType))
        {
            if (this.param.nu <= 0 || this.param.nu > 1)
            {
                return "nu <= 0 or nu > 1";
            }
        }

        if (SVMType.EPSILON_SVR.equals(svmType))
        {
            if (this.param.p < 0)
            {
                return "p < 0";
            }
        }

        if (this.param.shrinking != 0 && this.param.shrinking != 1)
        {
            return "shrinking != 0 and shrinking != 1";
        }

        if (this.param.probability != 0 && this.param.probability != 1)
        {
            return "probability != 0 and probability != 1";
        }

        if (this.param.probability == 1 && SVMType.ONE_CLASS.equals(svmType))
        {
            return "one-class SVM probability output not supported yet";
        }

        // check whether nu-svc is feasible
        if (SVMType.NU_SVC.equals(svmType))
        {
            int l = this.prob.numInstances;
            int max_nr_class = 16;
            int nr_class = 0;
            int[] label = new int[max_nr_class];
            int[] count = new int[max_nr_class];

            int i;
            for (i = 0; i < l; i++)
            {
                int this_label = this.prob.y.get(i).intValue();
                int j;
                for (j = 0; j < nr_class; j++)
                {
                    if (this_label == label[j])
                    {
                        ++count[j];
                        break;
                    }
                }

                if (j == nr_class)
                {
                    if (nr_class == max_nr_class)
                    {
                        max_nr_class *= 2;
                        int[] new_data = new int[max_nr_class];
                        System.arraycopy(label, 0, new_data, 0, label.length);
                        label = new_data;

                        new_data = new int[max_nr_class];
                        System.arraycopy(count, 0, new_data, 0, count.length);
                        count = new_data;
                    }
                    label[nr_class] = this_label;
                    count[nr_class] = 1;
                    ++nr_class;
                }
            }

            for (i = 0; i < nr_class; i++)
            {
                int n1 = count[i];
                for (int j = i + 1; j < nr_class; j++)
                {
                    int n2 = count[j];
                    if (this.param.nu * (n1 + n2) / 2 > Math.min(n1, n2))
                    {
                        return "specified nu is infeasible";
                    }
                }
            }
        }

        return null;
    }

    @Override
    public void train(Node[] item, Double category)
    {
        this.prob.add(category, item);
    }

    @Override
    public void train(Node[][] items, Double category)
    {
        for (Node[] nodes : items)
        {
            this.train(nodes, category);
        }
    }

    @Override
    public void train(List<Node[]> items, Double category)
    {
        for (Node[] nodes : items)
        {
            this.train(nodes, category);
        }
    }

    @Override
    public Double classify(Node[] item)
    {
        Model model = this.getTrainedModel();
        if (model != null)
        {
            return model.predict(item);
        }
        return 0.;
    }

    @Override
    public Double classify(Node[][] items)
    {
        List<Double> classification = new ArrayList<>();
        for (Node[] node : items)
        {
            classification.add(this.classify(node));
        }

        return null;
    }

    /**
     * Returns the trained model if it already exists. If it was not created before, the method tries to create it.
     * <p>
     * On creating the model, first the feasibility of the parameters are checked and afterwards a default value for the
     * gamma-parameter is used if none was specified and in case a precomputed kernel should be used the validity of the
     * training data is checked too.
     *
     * @return The trained model
     */
    public Model getTrainedModel()
    {
        if (this.model == null)
        {
            String error = checkParameter();
            if (error != null)
            {
                throw new IllegalArgumentException(error);
            }

            // normalizes the radius used for RBF f.e.
            if (this.param.gamma == 0 && this.prob.getMaxIndex() > 0)
            {
                this.param.gamma = 1.0 / this.prob.getMaxIndex();
            }

            // check if the format for a pre-computed kernel type is appropriate
            if (KernelType.PRECOMPUTED.equals(param.kernelType))
            {
                for (int i = 0; i < prob.numInstances; i++)
                {
                    if (this.prob.x.get(i)[0].index != 0)
                    {
                        System.err.print("Wrong kernel matrix: first column must be 0:sample_serial_number\n");
                        System.exit(1);
                    }
                    if ((int) prob.x.get(i)[0].value <= 0 || (int) prob.x.get(i)[0].value > this.prob.getMaxIndex())
                    {
                        System.err.print("Wrong input format: sample_serial_number out of range\n");
                        System.exit(1);
                    }
                }
            }

            this.model = train(this.prob, this.param);
        }
        return this.model;
    }

    /**
     * Resets a trained model.
     */
    public void resetModel()
    {
        this.model = null;
    }

    /**
     * Persists a trained model in the file specified in the <em> modelFileName</em> field of the {@link Parameter}
     * instance.
     * <p>
     * If the model was not trained before, the method will automatically train the model before persisting it.
     *
     * @throws IOException
     *         If the model could not be persisted
     */
    public void save() throws IOException
    {
        if (this.param != null && this.param.modelFileName != null)
        {
            if (this.model == null)
            {
                this.model = this.getTrainedModel();
            }

            this.model.save(this.param.modelFileName);
        }
    }

    @Override
    public boolean loadData(File serializedObject)
    {
        // TODO Auto-generated method stub
        return false;
    }
}
