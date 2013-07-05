package at.rovo.classifier.svm.kernel;

import at.rovo.classifier.svm.Cache;
import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.Problem;


public class OneClassKernel extends Kernel
{
	private final Cache cache;
	private final double[] QD;

	public OneClassKernel(Problem prob, Parameter param)
	{
		super(prob.numInstances, prob.x, param);
		cache = new Cache(prob.numInstances, (long) (param.cache_size * (1 << 20)));
		QD = new double[prob.numInstances];
		for (int i = 0; i < prob.numInstances; i++)
			QD[i] = function(i, i);
	}

	@Override
	public float[] get_Q(int i, int len)
	{
		float[][] data = new float[1][];
		int start, j;
		if ((start = cache.getData(i, data, len)) < len)
		{
			for (j = start; j < len; j++)
				data[0][j] = (float) function(i, j);
		}
		return data[0];
	}

	@Override
	public double[] get_QD()
	{
		return QD;
	}

	@Override
	public void swapIndex(int i, int j)
	{
		cache.swapIndex(i, j);
		super.swapIndex(i, j);
		do
		{
			double _ = QD[i];
			QD[i] = QD[j];
			QD[j] = _;
		}
		while (false);
	}
}