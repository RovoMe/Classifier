package at.rovo.classifier.svm;

/**
 * A support vector machine tries to find a line in a two-dimensional space, a plane in a three-dimensional space or a
 * hyperplane in a n-dimensional space. Separating points in a non-linear space is not trivial. Moreover transforming
 * the data into a higher dimensional space may simplify or even make it possible to perform the separation. This
 * transformation is done by a kernel function.
 * <p>
 * There are actually an infinite number of kernel functions, but in practice only a few are used.
 *
 * @author Roman Vottner
 */
public enum KernelType
{
    /**
     * A linear based kernel method.
     * <p>
     * Formula: <code>u'*v</code>
     */
    LINEAR(0)
            {
                public String toString()
                {
                    return "linear";
                }
            },
    /**
     * A polynomial based kernel method.
     * <p>
     * Formula: <code>(gamma*u'*v + coef0)^degree</code>
     */
    POLYNOMIAL(1)
            {
                public String toString()
                {
                    return "polynomial";
                }
            },
    /**
     * A radial basis function kernel method. This is the recommended method for a support vector machine.
     * <p>
     * Formula: <code>exp(-gamma*|u-v|^2)</code>
     */
    RBF(2)
            {
                public String toString()
                {
                    return "rbf";
                }
            },
    /**
     * A sigmoid based kernel method.
     * <p>
     * Formula: <code>tanh(gamma*u'*v + coef0)</code>
     */
    SIGMOID(3)
            {
                public String toString()
                {
                    return "sigmoid";
                }

            },
    /**
     * A predefined method for transforming the data into a higher space should be provided.
     */
    PRECOMPUTED(4)
            {
                public String toString()
                {
                    return "precomputed";
                }
            };

    /** The ordinal index of the corresponding kernel type **/
    private int value;

    /**
     * Initializes the kernel type with an ordinal index
     *
     * @param val
     *         The ordinal index of the kernel type
     */
    KernelType(int val)
    {
        this.value = val;
    }

    /**
     * Returns the ordinal index of the kernel type.
     *
     * @return The ordinal index of the kernel type.
     */
    public int valueOf()
    {
        return this.value;
    }

    /**
     * Returns the number of specified kernel types.
     *
     * @return The number of specified kernel types, which is 5
     */
    public static int length()
    {
        return 5;
    }

    /**
     * Returns the kernel type corresponding to the ordinal index.
     *
     * @param i
     *         The ordinal index of the kernel type to return
     *
     * @return The kernel type corresponding to the ordinal index
     */
    public static KernelType get(int i)
    {
        switch (i)
        {
            case 0:
                return KernelType.LINEAR;
            case 1:
                return KernelType.POLYNOMIAL;
            case 2:
                return KernelType.RBF;
            case 3:
                return KernelType.SIGMOID;
            case 4:
                return KernelType.PRECOMPUTED;
            default:
                return KernelType.RBF;
        }
    }

    /**
     * Returns the kernel type corresponding to the name of the kernel or the ordinal index as string.
     *
     * @param s
     *         Either the name of the kernel or the ordinal index as string
     *
     * @return The kernel type corresponding to the name or the ordinal index as string
     */
    public static KernelType get(String s)
    {
        switch (s)
        {
            case "linear":
            case "0":
                return KernelType.LINEAR;
            case "polynomial":
            case "1":
                return KernelType.POLYNOMIAL;
            case "rbf":
            case "2":
                return KernelType.RBF;
            case "sigmoid":
            case "3":
                return KernelType.SIGMOID;
            case "precomputed":
            case "4":
                return KernelType.PRECOMPUTED;
            default:
                return KernelType.RBF;
        }
    }
}
