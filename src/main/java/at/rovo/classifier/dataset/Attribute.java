/**
 * @(#)Attribute.java        1.5.2 09/03/29
 */
package at.rovo.classifier.dataset;

/**
 * An attribute wrapping its attribute values.
 * 
 * @author Ping He
 * @author Xiaohua Xu
 */
public abstract class Attribute
{
	/** The name of the attribute **/
	protected String name;
	/** The attribute values **/
	protected String[] data;

	/**
	 * Initialize an attribute with the specified name and attribute values
	 */
	public Attribute(String name, String[] data)
	{
		this.name = name;
		this.data = data;
	}

	/**
	 * Get the name of the attribute
	 * 
	 * @return The name of the attribute
	 */
	public String getName()
	{
		return this.name;
	}

	/**
	 * Set the name of the attribute
	 * 
	 * @param name The name of the attribute
	 */
	public void setName(String name)
	{
		this.name = name;
	}

	/**
	 * Get the attribute values on the attribute
	 * 
	 * @return The attribute values on the attribute
	 */
	public String[] getData()
	{
		return this.data;
	}

	/**
	 * Set the attribute values on the attribute
	 * 
	 * @param data The attribute values on the attribute
	 */
	public void setData(String[] data)
	{
		this.data = data;
	}

	/**
	 * The String exhibition of the attribute.
	 * 
	 * @return The name of the attribute
	 */
	public String toString()
	{
		return name;
	}
}