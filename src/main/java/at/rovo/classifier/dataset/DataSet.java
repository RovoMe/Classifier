/**
 * @(#)DataSet.java 1.5.4 09/04/23
 */
package at.rovo.classifier.dataset;

/**
 * A data set wrapping all the data.
 *
 * @author Ping He
 * @author Xiaohua Xu
 * @see UciDataSet.dataset.UciDataSet
 */
public abstract class DataSet
{
    /** The name of the data set **/
    protected String name;
    /** The specified classes **/
    protected String[] classes;
    /** The array of attributes wrapping their attribute values. **/
    protected Attribute[] attributes;
    /** The index of the class attribute in the attribute array **/
    protected int classAttributeIndex;
    /** The data of the data set **/
    protected String[][] trainData;
    /** The meta data of the data set **/
    protected MetaData metaData;

    /**
     * Initialize a data set.
     *
     * @param baseName The base name of the input files (.names and .data)
     */
    public DataSet(String baseName)
    {
        int beginIndex = baseName.lastIndexOf("/");
        if (beginIndex < 0)
        {
            beginIndex = baseName.lastIndexOf("\\");
        }
        this.name = baseName.substring(beginIndex + 1);
        load(baseName);
    }

    /**
     * Load data set from the specified input files.
     *
     * @param baseName The base name of the input files (.names and .data)
     * @see UciDataSet.dataset.UciDataSet#load(String baseName)
     */
    public abstract void load(String baseName);

    /**
     * Add a column set view (<i>ie</i> attribute view) for the data set.
     */
    public void addColumnSetView()
    {
        int attributeCount = metaData.getAttributeCount();
        this.attributes = new Attribute[attributeCount];

        String[][] transposedTrainData = transpose(getTrainData());
        for (int i = 0; i < attributeCount; i++)
        {
            String name = metaData.getAttributeNameAt(i);
            if (metaData.isAttributeContinuousAt(i))
            {
                attributes[i] = new ContinuousAttribute(name, transposedTrainData[i]);
            }
            else
            {
                String[] nominalValues = metaData.getAttributeNominalValuesAt(i);
                attributes[i] = new DiscreteAttribute(name, nominalValues, transposedTrainData[i]);
            }
        }
    }

    /**
     * Set the name of the data set.
     *
     * @param name The name of the data set
     */
    public void setName(String name)
    {
        this.name = name;
    }

    /**
     * Get the name of the data set.
     *
     * @return the name of the data set
     */
    public String getName()
    {
        return name;
    }

    /**
     * Get the meta data of the data set.
     *
     * @return The meta data of the training data
     */
    public MetaData getMetaData()
    {
        return this.metaData;
    }

    /**
     * Set the meta data of the data set.
     *
     * @param metaData The new meta data object
     */
    public void setMetaData(MetaData metaData)
    {
        this.metaData = metaData;
    }

    /**
     * Set the data of the data set.
     *
     * @param trainData The data to use for training the classifier
     */
    public void setTrainData(String[][] trainData)
    {
        this.trainData = trainData;
    }

    /**
     * Get the data of the data set.
     *
     * @return The training data
     */
    public String[][] getTrainData()
    {
        return this.trainData;
    }

    /**
     * Set the attributes of the data set.
     *
     * @param attributes The attributes of the data set
     */
    public void setAttributes(Attribute[] attributes)
    {
        this.attributes = attributes;
    }

    /**
     * Get the attributes of the data set.
     *
     * @return The attributes of the data set
     */
    public Attribute[] getAttributes()
    {
        return attributes;
    }

    /**
     * Get the index of the class attribute.
     *
     * @return The class attribute index
     */
    public int getClassAttributeIndex()
    {
        return classAttributeIndex;
    }

    /**
     * Set the index of the class attribute.
     *
     * @param classAttributeIndex The index of the class attribute
     */
    public void setClassAttributeIndex(int classAttributeIndex)
    {
        this.classAttributeIndex = classAttributeIndex;
    }

    /**
     * Get the number of the data in the data set.
     *
     * @return The number of the data in the data set
     */
    public int getCaseCount()
    {
        return this.trainData.length;
    }

    /**
     * Get the number of the attributes in the data set.
     *
     * @return The number of the attributes in the data set
     */
    public int getAttributeCount()
    {
        return attributes.length;
    }

    /**
     * Sets the specified classes of the data set.
     *
     * @param classes The classes of the data set
     */
    public void setClassValues(String[] classes)
    {
        this.classes = classes;
    }

    /**
     * Get the nominal values of the class attribute.
     *
     * @return The nominal values of the class attribute
     */
    public String[] getClassValues()
    {
        return this.classes;
    }

    /**
     * Get the number of different class values
     */
    public int getClassCount()
    {
        return this.classes.length;
    }

    /**
     * Transpose a 2D String array.
     *
     * @param from Source array of the transposition
     * @return The transposed string array
     */
    private static String[][] transpose(String[][] from)
    {
        String[][] to = new String[from[0].length][from.length];
        for (int i = 0; i < from.length; i++)
        {
            for (int j = 0; j < from[0].length; j++)
            {
                to[j][i] = from[i][j];
            }
        }
        return to;
    }
}