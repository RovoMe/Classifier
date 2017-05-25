package at.rovo.classifier.svm.solver.instance;

import at.rovo.classifier.svm.kernel.SVCKernel;
import at.rovo.classifier.svm.solver.Solver;
import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.Problem;
import at.rovo.classifier.svm.struct.SolutionInfo;
import java.lang.invoke.MethodHandles;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CSVC extends SolveInstance
{
    private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    private double Cp;
    private double Cn;

    public CSVC(double Cp, double Cn)
    {
        this.Cp = Cp;
        this.Cn = Cn;
    }

    @Override
    public void solve(Problem prob, Parameter param, double[] alpha, SolutionInfo si)
    {
        int l = prob.numInstances;
        double[] minus_ones = new double[l];
        byte[] y = new byte[l];

        int i;

        for (i = 0; i < l; i++)
        {
            alpha[i] = 0;
            minus_ones[i] = -1;
            if (prob.y.get(i) > 0)
            {
                y[i] = +1;
            }
            else
            {
                y[i] = -1;
            }
        }

        Solver s = new Solver();
        s.solve(l, new SVCKernel(prob, param, y), minus_ones, y, alpha, Cp, Cn, param.eps, si, param.shrinking);

        double sum_alpha = 0;
        for (i = 0; i < l; i++)
        {
            sum_alpha += alpha[i];
        }

        if (Cp == Cn)
        {
            if (LOG.isDebugEnabled())
            {
                LOG.debug("nu = " + sum_alpha / (Cp * prob.numInstances) + "\n");
            }
        }

        for (i = 0; i < l; i++)
        {
            alpha[i] *= y[i];
        }
    }
}
