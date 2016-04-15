package at.rovo.classifier.svm.solver.instance;

import at.rovo.classifier.svm.kernel.OneClassKernel;
import at.rovo.classifier.svm.solver.Solver;
import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.Problem;
import at.rovo.classifier.svm.struct.SolutionInfo;

public class OneClass extends SolveInstance
{
    @Override
    public void solve(Problem prob, Parameter param, double[] alpha, SolutionInfo si)
    {
        int l = prob.numInstances;
        double[] zeros = new double[l];
        byte[] ones = new byte[l];
        int i;

        int n = (int) (param.nu * prob.numInstances); // # of alpha's at upper bound

        for (i = 0; i < n; i++)
        {
            alpha[i] = 1;
        }
        if (n < prob.numInstances)
        {
            alpha[n] = param.nu * prob.numInstances - n;
        }
        for (i = n + 1; i < l; i++)
        {
            alpha[i] = 0;
        }

        for (i = 0; i < l; i++)
        {
            zeros[i] = 0;
            ones[i] = 1;
        }

        Solver s = new Solver();
        s.solve(l, new OneClassKernel(prob, param), zeros, ones, alpha, 1.0, 1.0, param.eps, si, param.shrinking);
    }
}
