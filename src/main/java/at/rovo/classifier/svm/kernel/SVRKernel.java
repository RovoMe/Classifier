package at.rovo.classifier.svm.kernel;

import at.rovo.classifier.svm.Cache;
import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.Problem;

public class SVRKernel extends Kernel
{
	private final int l;
	private final Cache cache;
	private final byte[] sign;
	private final int[] index;
	private int next_buffer;
	private float[][] buffer;
	private final double[] QD;

	public SVRKernel(Problem prob, Parameter param)
	{
		super(prob.numInstances, prob.x, param);
		l = prob.numInstances;
		cache = new Cache(l, (long) (param.cache_size * (1 << 20)));
		QD = new double[2 * l];
		sign = new byte[2 * l];
		index = new int[2 * l];
		for (int k = 0; k < l; k++)
		{
			sign[k] = 1;
			sign[k + l] = -1;
			index[k] = k;
			index[k + l] = k;
			QD[k] = function(k, k);
			QD[k + l] = QD[k];
		}
		buffer = new float[2][2 * l];
		next_buffer = 0;
	}

	@Override
	public void swapIndex(int i, int j)
	{
		do
		{
			byte _ = sign[i];
			sign[i] = sign[j];
			sign[j] = _;
		}
		while (false);
		do
		{
			int _ = index[i];
			index[i] = index[j];
			index[j] = _;
		}
		while (false);
		do
		{
			double _ = QD[i];
			QD[i] = QD[j];
			QD[j] = _;
		}
		while (false);
	}

	@Override
	public float[] get_Q(int i, int len)
	{
		float[][] data = new float[1][];
		int j, real_i = index[i];
		if (cache.getData(real_i, data, l) < l)
		{
			for (j = 0; j < l; j++)
				data[0][j] = (float) function(real_i, j);
		}

		// reorder and copy
		float buf[] = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		byte si = sign[i];
		for (j = 0; j < len; j++)
			buf[j] = (float) si * sign[j] * data[0][index[j]];
		return buf;
	}

	@Override
	public double[] get_QD()
	{
		return QD;
	}
}
