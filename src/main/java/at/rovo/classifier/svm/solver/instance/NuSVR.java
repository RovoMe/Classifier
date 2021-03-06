package at.rovo.classifier.svm.solver.instance;

import at.rovo.classifier.svm.kernel.SVRKernel;
import at.rovo.classifier.svm.solver.Solver_NU;
import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.Problem;
import at.rovo.classifier.svm.struct.SolutionInfo;
import java.lang.invoke.MethodHandles;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NuSVR extends SolveInstance
{
    private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    @Override
    public void solve(Problem prob, Parameter param, double[] alpha, SolutionInfo si)
    {
        int l = prob.numInstances;
        double C = param.C;
        double[] alpha2 = new double[2 * l];
        double[] linear_term = new double[2 * l];
        byte[] y = new byte[2 * l];
        int i;

        double sum = C * param.nu * l / 2;
        for (i = 0; i < l; i++)
        {
            alpha2[i] = alpha2[i + l] = Math.min(sum, C);
            sum -= alpha2[i];

            linear_term[i] = -prob.y.get(i);
            y[i] = 1;

            linear_term[i + l] = prob.y.get(i);
            y[i + l] = -1;
        }

        Solver_NU s = new Solver_NU();
        s.solve(2 * l, new SVRKernel(prob, param), linear_term, y, alpha2, C, C, param.eps, si, param.shrinking);

        if (LOG.isDebugEnabled())
        {
            LOG.debug("epsilon = " + (-si.r) + "\n");
        }

        for (i = 0; i < l; i++)
        {
            alpha[i] = alpha2[i] - alpha2[i + l];
        }
    }
}
