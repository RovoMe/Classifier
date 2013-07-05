package at.rovo.classifier.svm.solver.instance;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import at.rovo.classifier.svm.kernel.SVRKernel;
import at.rovo.classifier.svm.solver.Solver_NU;
import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.Problem;
import at.rovo.classifier.svm.struct.SolutionInfo;

public class NuSVR extends SolveInstance
{
	protected static Logger logger = LogManager.getLogger(NuSVR.class.getName());
	
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

		if (logger.isDebugEnabled())
			logger.debug("epsilon = " + (-si.r) + "\n");

		for (i = 0; i < l; i++)
			alpha[i] = alpha2[i] - alpha2[i + l];
	}
}
