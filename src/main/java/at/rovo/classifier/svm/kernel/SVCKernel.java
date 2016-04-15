package at.rovo.classifier.svm.kernel;

import at.rovo.classifier.svm.Cache;
import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.Problem;
import at.rovo.classifier.svm.utils.Utils;


//
// Q matrices for various formulations
//
public class SVCKernel extends Kernel
{
    private final byte[] y;
    private final Cache cache;
    private final double[] QD;

    public SVCKernel(Problem prob, Parameter param, byte[] y_)
    {
        super(prob.numInstances, prob.x, param);
        this.y = y_.clone();
        this.cache = new Cache(prob.numInstances, (long) (param.cache_size * (1 << 20)));
        this.QD = new double[prob.numInstances];
        for (int i = 0; i < prob.numInstances; i++)
        {
            this.QD[i] = this.function(i, i);
        }
    }

    @Override
    public float[] get_Q(int i, int len)
    {
        float[][] data = new float[1][];
        int start, j;
        if ((start = this.cache.getData(i, data, len)) < len)
        {
            for (j = start; j < len; j++)
            {
                data[0][j] = (float) (this.y[i] * this.y[j] * function(i, j));
            }
        }
        return data[0];
    }

    @Override
    public double[] get_QD()
    {
        return this.QD;
    }

    @Override
    public void swapIndex(int i, int j)
    {
        this.cache.swapIndex(i, j);
        super.swapIndex(i, j);
        Utils.swap(y, i, j);
        Utils.swap(QD, i, j);
    }
}