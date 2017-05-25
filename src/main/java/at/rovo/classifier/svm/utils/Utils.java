package at.rovo.classifier.svm.utils;

import at.rovo.classifier.svm.struct.Node;
import java.util.List;

/**
 * Provides utility methods used by the SVM library
 */
public class Utils
{
    /**
     * Swaps two elements of a list containing {@link Node} elements
     *
     * @param list
     *         The list containing the node elements to swap
     * @param i
     *         The index of the element to swap
     * @param j
     *         The index of the element to swap
     */
    public static void swap(List<Node[]> list, int i, int j)
    {
        Node[] _n = list.get(i);
        list.set(i, list.get(j));
        list.set(j, _n);
    }

    /**
     * Swaps the position of two elements in the provided double array
     *
     * @param array
     *         The double array containing the two elements to swap
     * @param i
     *         The index of the element to swap
     * @param j
     *         The index of the element to swap
     */
    public static void swap(double[] array, int i, int j)
    {
        double _d = array[i];
        array[i] = array[j];
        array[j] = _d;
    }

    /**
     * Swaps the position of two elements in the provided integer array
     *
     * @param array
     *         The integer array containing the two elements to swap
     * @param i
     *         The index of the element to swap
     * @param j
     *         The index of the element to swap
     */
    public static void swap(int[] array, int i, int j)
    {
        int _i = array[i];
        array[i] = array[j];
        array[j] = _i;
    }

    /**
     * Swaps the position of two elements in the provided float array
     *
     * @param array
     *         The float array containing the two elements to swap
     * @param i
     *         The index of the element to swap
     * @param j
     *         The index of the element to swap
     */
    public static void swap(float[] array, int i, int j)
    {
        float _f = array[i];
        array[i] = array[j];
        array[j] = _f;
    }

    /**
     * Swaps the position of two elements in the provided byte array
     *
     * @param array
     *         The byte array containing the two elements to swap
     * @param i
     *         The index of the element to swap
     * @param j
     *         The index of the element to swap
     */
    public static void swap(byte[] array, int i, int j)
    {
        byte _b = array[i];
        array[i] = array[j];
        array[j] = _b;
    }
}
