package at.rovo.classifier.svm.solver.instance;

import at.rovo.classifier.svm.kernel.SVRKernel;
import at.rovo.classifier.svm.solver.Solver;
import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.Problem;
import at.rovo.classifier.svm.struct.SolutionInfo;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class EpsilonSVR extends SolveInstance
{
    protected static Logger logger = LogManager.getLogger(EpsilonSVR.class.getName());

    public void solve(Problem prob, Parameter param, double[] alpha, SolutionInfo si)
    {
        int l = prob.numInstances;
        double[] alpha2 = new double[2 * l];
        double[] linear_term = new double[2 * l];
        byte[] y = new byte[2 * l];
        int i;

        for (i = 0; i < l; i++)
        {
            alpha2[i] = 0;
            linear_term[i] = param.p - prob.y.get(i);
            y[i] = 1;

            alpha2[i + l] = 0;
            linear_term[i + l] = param.p + prob.y.get(i);
            y[i + l] = -1;
        }

        Solver s = new Solver();
        s.solve(2 * l, new SVRKernel(prob, param), linear_term, y, alpha2, param.C, param.C, param.eps, si,
                param.shrinking);

        double sum_alpha = 0;
        for (i = 0; i < l; i++)
        {
            alpha[i] = alpha2[i] - alpha2[i + l];
            sum_alpha += Math.abs(alpha[i]);
        }
        if (logger.isDebugEnabled())
        {
            logger.debug("nu = " + sum_alpha / (param.C * l) + "\n");
        }
    }
}
