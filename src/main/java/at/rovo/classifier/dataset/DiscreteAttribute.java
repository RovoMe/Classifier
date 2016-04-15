/**
 * @(#)DiscreteAttribute.java 1.5.2 09/03/29
 */

package at.rovo.classifier.dataset;

/**
 * A discrete attribute wrapping its discrete attribute values.
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public class DiscreteAttribute extends Attribute
{
    /** The nominal discrete attribute values of the attribute **/
    private String[] nominalValues;

    /**
     * Initialize a discrete attribute.
     *
     * @param name The name of the attribute
     * @param nominalValues The nominal values of the attribute
     * @param data The attribute values on the attribute
     */
    public DiscreteAttribute(String name, String[] nominalValues, String[] data)
    {
        super(name, data);
        this.nominalValues = nominalValues;
    }

    /**
     * Get the nominal values of the attribute
     *
     * @return The nominal values of the attribute
     */
    public String[] getNominalValues()
    {
        return this.nominalValues;
    }

    /**
     * Set the nominal values of the attribute
     *
     * @param The nominal values of the attribute
     */
    public void setNominalValues(String[] nominalValues)
    {
        this.nominalValues = nominalValues;
    }

    /**
     * Get the number of nominal values of the attribute
     *
     * @return The number of nominal values of the attribute
     */
    public int getNominalValuesCount()
    {
        return nominalValues.length;
    }

    /**
     * The String exhibition of the discrete attribute
     *
     * @return The name of the attribute followed by a colon and an enumeration
     *         of the nominal values of the attribute
     */
    public String toString()
    {
        return name + " : " + java.util.Arrays.toString(nominalValues);
    }
}