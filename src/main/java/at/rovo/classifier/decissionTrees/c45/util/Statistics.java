package at.rovo.classifier.decissionTrees.c45.util;

/**
 * A toolbox of statistics, including sum, mean, std and indirect sort.
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public class Statistics
{
    /** The THRESHOLD value for the choice of quick sort **/
    private static final int THRESHOLD = 7;

    private Statistics()
    {
    }

    /**
     * Compute the sum of the specified float array.
     *
     * @param list
     *         The float array whose sum is to be computed
     *
     * @return The sum of the specified array
     */
    public static float sum(float... list)
    {
        float result = 0;
        for (float x : list)
        {
            result += x;
        }
        return result;
    }

    /**
     * Compute the mean of the specified float array.
     *
     * @param list
     *         The float array whose average is to be computed
     *
     * @return The mean value of the specified array
     */
    public static float mean(float... list)
    {
        return sum(list) / list.length;
    }

    /**
     * Compute the standard deviation of the specified float array.
     *
     * @param list
     *         The float array whose standard deviation is to be computed
     *
     * @return The standard deviation of the specified array
     */
    public static float standardDeviation(float... list)
    {
        float total = 0;
        for (float item : list)
        {
            total += item;
        }
        float average = total / list.length;

        float result = 0;
        for (float item : list)
        {
            result += (item - average) * (item - average);
        }
        result = (float) Math.sqrt(result / (list.length - 1));

        return result;
    }

    /**
     * Compute the standard deviation of the specified float array. <br>
     * An alternative method of <em>standardDeviation</em>()
     *
     * @param list
     *         The float array whose standard deviation is to be computed
     *
     * @return The standard deviation of the specified array
     */
    public static float std(float... list)
    {
        return standardDeviation(list);
    }

    /**
     * Indirect sort the specified float array in ascending order.
     *
     * @param data
     *         The float array whose value is to be indirect-sorted.
     *
     * @return The new sequence of the data in the sorted float array.
     */
    public static int[] indirectSort(float[] data)
    {
        int[] id = new int[data.length];
        for (int i = 0; i < id.length; i++)
        {
            id[i] = i;
        }
        quickSort(data, id, 0, data.length - 1);
        return id;
    }

    /**
     * Quick sort the specified values in <i>data</i> array and adjust their
     * corresponding values in <i>id</i> array.
     * <p/>
     * It is called when the number of values to be sorted is greater than the
     * THRESHOLD value.
     *
     * @param data
     *         The float array to be sorted
     * @param id
     *         The indirect-sort result of data, recording the new positions of the original values
     * @param left
     *         The begin (inclusive) index of the data to be sorted
     * @param right
     *         Then end (inclusive) index of the data to be sorted
     */
    private static void quickSort(float[] data, int[] id, int left, int right)
    {
        if (left + THRESHOLD >= right)
        {
            insertionSort(data, id, left, right);
        }
        else
        {
            double pivot = median3(data, id, left, right);

            int i = left, j = right - 1;
            for (; ; )
            {
                while (data[++i] < pivot)
                {
                    // do nothing
                }
                while (data[--j] > pivot)
                {
                    // do nothing
                }
                if (i < j)
                {
                    swap(id, i, j);
                    swap(data, i, j);
                }
                else
                {
                    break;
                }
            }
            swap(id, i, right - 1);
            swap(data, i, right - 1);

            quickSort(data, id, left, i - 1);
            quickSort(data, id, i + 1, right);
        }
    }

    /**
     * Insert sort the specified values in <i>data</i> array and adjust their
     * corresponding values in <i>id</i> array.
     * <p>
     * It is called when the number of values to be sorted is equal or less than
     * the THRESHOLD value.
     *
     * @param data
     *         The float array to be sorted
     * @param id
     *         The indirect-sort result of data, recording the new positions of the original values
     * @param left
     *         The begin index of the data to be sorted
     * @param right
     *         Then end index of the data to be sorted
     */
    private static void insertionSort(float[] data, int[] id, int left, int right)
    {
        for (int i = left + 1; i <= right; i++)
        {
            int tempI = id[i];
            float tempV = data[i];
            int j;
            for (j = i; j > left && tempV < data[j - 1]; j--)
            {
                id[j] = id[j - 1];
                data[j] = data[j - 1];
            }
            id[j] = tempI;
            data[j] = tempV;
        }
    }

    /**
     * Find the median of data[left], data[right] and data[center] three values
     * and range them in order.
     *
     * @param data
     *         The float array to be sorted
     * @param id
     *         The indirect-sort result of data, recording the new positions of the original values
     * @param left
     *         The begin index of the data to be sorted
     * @param right
     *         Then end index of the data to be sorted
     *
     * @return The median of data[left], data[right] and data[center]
     */
    private static double median3(float[] data, int[] id, int left, int right)
    {
        int center = (left + right) / 2;
        if (data[left] > data[center])
        {
            swap(id, left, center);
            swap(data, left, center);
        }
        if (data[left] > data[right])
        {
            swap(id, left, right);
            swap(data, left, right);
        }
        if (data[center] > data[right])
        {
            swap(id, center, right);
            swap(data, center, right);
        }

        swap(id, center, right - 1);
        swap(data, center, right - 1);

        return data[right - 1];
    }

    /**
     * Swap the position of two values in an <i>int</i> array.
     *
     * @param a
     *         The array containing the values to swap
     * @param i
     *         the value to swap with element <i>j</i>
     * @param j
     *         the value to swap with element <i>i</i>
     */
    private static void swap(int[] a, int i, int j)
    {
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }

    /**
     * Swap the position of two values in an <i>float</i> array.
     *
     * @param a
     *         The array containing the values to swap
     * @param i
     *         the value to swap with element <i>j</i>
     * @param j
     *         the value to swap with element <i>i</i>
     */
    private static void swap(float[] a, int i, int j)
    {
        float temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
}