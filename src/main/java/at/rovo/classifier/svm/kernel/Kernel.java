package at.rovo.classifier.svm.kernel;

import java.util.ArrayList;
import java.util.List;
import at.rovo.classifier.svm.KernelType;
import at.rovo.classifier.svm.struct.Node;
import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.QMatrix;

/**
 * <p></p>
 * 
 * @author Chih-Chung Chang, Chih-Jen Lin
 */
public abstract class Kernel extends QMatrix
{
	private List<Node[]> x;
	private final double[] x_square;

	// svm_parameter
	private final KernelType kernelType;
	private final int degree;
	private final double gamma;
	private final double coef0;

	public abstract float[] get_Q(int column, int len);

	public abstract double[] get_QD();

	public void swapIndex(int i, int j)
	{
		do
		{
			Node[] _ = x.get(i);
			x.set(i, x.get(j));
			x.set(j, _);
		}
		while (false);
		if (x_square != null)
			do
			{
				double _ = x_square[i];
				x_square[i] = x_square[j];
				x_square[j] = _;
			}
			while (false);
	}

	private static double powi(double base, int times)
	{
		double tmp = base, ret = 1.0;

		for (int t = times; t > 0; t /= 2)
		{
			if (t % 2 == 1)
				ret *= tmp;
			tmp = tmp * tmp;
		}
		return ret;
	}

	double function(int i, int j)
	{
		switch (kernelType)
		{
			case LINEAR:
				return dot(x.get(i), x.get(j));
			case POLYNOMIAL:
				return powi(gamma * dot(x.get(i), x.get(j)) + coef0, degree);
			case RBF:
				return Math.exp(-gamma * (x_square[i] + x_square[j] - 2 * dot(x.get(i), x.get(j))));
			case SIGMOID:
				return Math.tanh(gamma * dot(x.get(i), x.get(j)) + coef0);
			case PRECOMPUTED:
				return x.get(i)[(int) (x.get(j)[0].value)].value;
			default:
				return 0; // java
		}
	}

	Kernel(int l, List<Node[]> x_, Parameter param)
	{
		this.kernelType = param.kernelType;
		this.degree = param.degree;
		this.gamma = param.gamma;
		this.coef0 = param.coef0;

		x = clone(x_);

		if (KernelType.RBF.equals(kernelType))
		{
			x_square = new double[l];
			for (int i = 0; i < l; i++)
				x_square[i] = dot(x.get(i), x.get(i));
		}
		else
			x_square = null;
	}
	
	private List<Node[]> clone(List<Node[]> list)
	{
		List<Node[]> copy = new ArrayList<Node[]>(list.size());
		for (Node[] node : list)
			copy.add((Node[])node.clone());
		return copy;
	}

	static double dot(Node[] x, Node[] y)
	{
		double sum = 0;
		int xlen = x.length;
		int ylen = y.length;
		int i = 0;
		int j = 0;
		while (i < xlen && j < ylen)
		{
			if (x[i].index == y[j].index)
				sum += x[i++].value * y[j++].value;
			else
			{
				if (x[i].index > y[j].index)
					++j;
				else
					++i;
			}
		}
		return sum;
	}

	public static double function(Node[] x, Node[] y, Parameter param)
	{
		switch (param.kernelType)
		{
			case LINEAR:
				return dot(x, y);
			case POLYNOMIAL:
				return powi(param.gamma * dot(x, y) + param.coef0, param.degree);
			case RBF:
			{
				double sum = 0;
				int xlen = x.length;
				int ylen = y.length;
				int i = 0;
				int j = 0;
				while (i < xlen && j < ylen)
				{
					if (x[i].index == y[j].index)
					{
						double d = x[i++].value - y[j++].value;
						sum += d * d;
					}
					else if (x[i].index > y[j].index)
					{
						sum += y[j].value * y[j].value;
						++j;
					}
					else
					{
						sum += x[i].value * x[i].value;
						++i;
					}
				}
	
				while (i < xlen)
				{
					sum += x[i].value * x[i].value;
					++i;
				}
	
				while (j < ylen)
				{
					sum += y[j].value * y[j].value;
					++j;
				}
	
				return Math.exp(-param.gamma * sum);
			}
			case SIGMOID:
				return Math.tanh(param.gamma * dot(x, y) + param.coef0);
			case PRECOMPUTED:
				return x[(int) (y[0].value)].value;
			default:
				return 0; // java
		}
	}
}