/**
 * @(#)UciDataSet.java 1.5.2 09/03/29
 */
package at.rovo.classifier.dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * A data set loaded from UCI dataset-format files.
 * <p>
 * This implementation is based on the implemented by Xu and He in their C45 framework. They assumed that the class
 * attribute is at the same position in the meta-data file (.names) as in the data file (.data). However this assumption
 * is not true in most UCI data sets.
 *
 * @author Xiaohua Xu
 * @author Ping He
 * @author Roman Vottner
 * @link http://www.cs.washington.edu/dm/vfml/appendixes/c45.htm
 */
public class UciDataSet extends DataSet
{
    private final static Logger LOG = LogManager.getLogger(MethodHandles.lookup().lookupClass());

    /** Needed to extract the index of the class label **/
    private boolean foundClassLabel = false;
    /** The index of the class label in the meta file **/
    private int metaClassIndex = 0;

    /**
     * Initialize a UCI data set.
     *
     * @param baseName
     *         The base name of the input files (.names and .data)
     */
    public UciDataSet(String baseName)
    {
        super(baseName);
        this.addColumnSetView();
    }

    /**
     * <p>Loads the data contained in the UCI data file.</p>
     *
     * @param baseName
     *         The file name to load the data from
     */
    public void load(String baseName)
    {
        this.setMetaData(this.loadMetaData(baseName + ".names"));
        this.setTrainData(this.loadData(baseName + ".data"));
    }

    /**
     * Load meta data from a specified file ending with <em>.names</em>.
     *
     * @param filename
     *         The name of the input file
     *
     * @return The loaded meta data
     *
     * @link http://www.cs.washington.edu/dm/vfml/appendixes/c45.htm
     */
    public MetaData loadMetaData(String filename)
    {
        int attributeIndex = 0;

        // The 'tempAttributeNames' ArrayList dynamically reads the attribute
        // names in
        List<String> tempAttributeNames = new ArrayList<>();
        // The 'tempIsContinuous' ArrayList dynamically reads their
        // continuous properties in;
        List<Boolean> tempIsContinuous = new ArrayList<>();
        // The 'tempNominalValues' ArrayList dynamically reads the nominal
        // values of the discrete attributes in
        List<String[]> tempNominalValues = new ArrayList<>();

        try
        {
            BufferedReader bf = new BufferedReader(new FileReader(filename));
            String line;
            while ((line = bf.readLine()) != null)
            {
                if (line.trim().length() == 0)
                {
                    continue;
                }

                // The class values are often defined first, but this is no must
                // Storing the index of the "attribute" does not work in the way
                // Xu and He implemented it, as the class value in the data set
                // often comes last while it is often defined first in the meta
                // file
                if (!line.contains(":"))
                {
                    List<String> classes = new ArrayList<>();
                    StringTokenizer token = new StringTokenizer(line, "\t\n,.");
                    while (token.hasMoreTokens())
                    {
                        String s = token.nextToken().trim();
                        if (s.contains("|"))
                        {
                            break;
                        }
                        classes.add(s);
                    }
                    // store the class values
                    this.setClassValues((String[]) classes.toArray());
                    tempAttributeNames.add("class");
                    tempIsContinuous.add(false);
                    tempNominalValues.add((String[]) classes.toArray());

                    // store the index of the class label to move the class label
                    // to the appropriate position in case the indices are
                    // different
                    this.metaClassIndex = attributeIndex;
                }
                else
                {
                    StringTokenizer token = new StringTokenizer(line, ":\t\n,.");
                    String word = token.nextToken().trim();

                    // Add attribute name
                    tempAttributeNames.add(word);

                    word = token.nextToken().trim();
                    // Construct and add a ContinuousAttribute object
                    if (word.equals("continuous"))
                    {
                        tempIsContinuous.add(true);
                        tempNominalValues.add(null);
                    }
                    // Construct and add a DiscreteAttribute object
                    else
                    {
                        tempIsContinuous.add(false);
                        // Get the nominal values of discrete attributes
                        List<String> values = new ArrayList<>();
                        values.add(word);
                        while (token.hasMoreTokens())
                        {
                            String s = token.nextToken().trim();
                            // Ignore comments
                            if (s.contains("|"))
                            {
                                break;
                            }
                            values.add(s);
                        }
                        tempNominalValues.add((String[]) values.toArray());
                    }
                }

                attributeIndex++;
            }
            bf.close();
        }
        catch (IOException e)
        {
            LOG.error("Exception while loading meta data from file " + filename + "! Reason: "
                      + e.getLocalizedMessage(), e);
        }

        // Transform the ArrayList to arrays
        return new MetaData((String[]) tempAttributeNames.toArray(), (Boolean[]) tempIsContinuous.toArray(),
                            (String[][]) tempNominalValues.toArray());
    }

    /**
     * Load data from the specified file.
     *
     * @param filename
     *         The name of the input file
     *
     * @return The loaded data
     */
    public String[][] loadData(String filename)
    {
        // Read in the test data in its original arrangement and extract its
        // data fields
        List<String[]> testList = new ArrayList<>();
        try
        {
            BufferedReader bf = new BufferedReader(new FileReader(filename));
            String reader;
            while ((reader = bf.readLine()) != null)
            {
                if (!reader.trim().equals(""))
                {
                    String[] line = this.extract(reader, ",");
                    testList.add(line);
                }
            }
            bf.close();
        }
        catch (IOException e)
        {
            LOG.error("Exception while loading data from file " + filename + "! Reason: " + e.getLocalizedMessage(), e);
        }

        // Transform the dynamic ArrayList to static Array
        return testList.toArray(new String[0][]);
    }

    /**
     * Extracts the attribute values from a single String separated with specified delimiter
     *
     * @param source
     *         The String to be extracted
     * @param delimiterString
     *         The delimiter with which the attribute values are separated
     *
     * @return The extracted attribute values
     */
    private String[] extract(String source, String delimiterString)
    {
        String[] data = new String[source.length()];
        int count = 0;

        int splitPoint = 0;
        int j = 0;
        for (int length = source.length(); j < length; j++)
        {
            // Compare char by char
            if (delimiterString.indexOf(source.charAt(j)) >= 0)
            {
                if (splitPoint != j)
                {
                    data[count++] = source.substring(splitPoint, j).trim();
                    splitPoint = j;
                }
                splitPoint++;
            }
        }
        if (splitPoint != j)
        {
            data[count++] = source.substring(splitPoint, j).trim();
        }
        // Only number of "count" Strings are filled in data
        String[] result = new String[count];
        System.arraycopy(data, 0, result, 0, count);

        // 		String[] result = source.split(delimiterString);
        if (!this.foundClassLabel)
        {
            for (int i = 0; i < result.length; i++)
            {
                String[] classes = this.getClassValues();
                for (String _class : classes)
                {
                    if (result[i].equals(_class))
                    {
                        this.setClassAttributeIndex(i);
                        this.foundClassLabel = true;

                        this.swapClassLabelPosition(this.metaClassIndex, i);
                        break;
                    }
                }
                if (this.foundClassLabel)
                {
                    break;
                }
            }
        }

        return result;
    }

    /**
     * Swaps the position of the class labels in the MetaData file as there is only one swap necessary instead of n if
     * we would swap the classes within the data extraction.
     *
     * @param metaPos
     *         The position of the class label in the meta file
     * @param dataPos
     *         The position of the class label in the data file
     */
    private void swapClassLabelPosition(int metaPos, int dataPos)
    {
        if (metaPos == dataPos)
        {
            return;
        }

        MetaData metaData = this.getMetaData();
        // The stored positions
        String[] attributes = metaData.getAttributeNames();
        // keeps track if the attribute is continuous
        Boolean[] continuous = metaData.isAttributesContinuous();
        // The nominal values
        String[][] nominal = metaData.getAttributeNominalValues();


        String tmpName = null;
        boolean tmpCont = false;
        String[] tmpNominal = null;
        // The class labels need to be moved forward
        if (metaPos > dataPos)
        {
            for (int i = attributes.length - 1; i >= 0; i--)
            {
                if ((i <= 0 && i < dataPos) || i > metaPos)
                {
                    // These elements are already at the correct position
                }
                else if (i == metaPos)
                {
                    tmpName = attributes[i];
                    tmpCont = continuous[i];
                    tmpNominal = nominal[i];
                }
                else if (i == dataPos)
                {
                    metaData.setAttributeNameAt(i + 1, attributes[i]);
                    metaData.setAttributeContinuousAt(i + 1, continuous[i]);
                    metaData.setAttributeNominalValuesAt(i + 1, nominal[i]);

                    metaData.setAttributeNameAt(i, tmpName);
                    metaData.setAttributeContinuousAt(i, tmpCont);
                    metaData.setAttributeNominalValuesAt(i, tmpNominal);
                }
                else
                {
                    metaData.setAttributeNameAt(i + 1, attributes[i]);
                    metaData.setAttributeContinuousAt(i + 1, continuous[i]);
                    metaData.setAttributeNominalValuesAt(i + 1, nominal[i]);
                }
            }
        }
        // The class labels need to be moved backward
        else
        {
            for (int i = 0; i < attributes.length; i++)
            {
                if ((i <= 0 && i < metaPos) || i > dataPos)
                {
                    // These elements are already at the correct position
                }
                else if (i == metaPos)
                {
                    tmpName = attributes[i];
                    tmpCont = continuous[i];
                    tmpNominal = nominal[i];
                }
                else if (i < dataPos)
                {
                    metaData.setAttributeNameAt(i - 1, attributes[i]);
                    metaData.setAttributeContinuousAt(i - 1, continuous[i]);
                    metaData.setAttributeNominalValuesAt(i - 1, nominal[i]);
                }
                else
                {
                    metaData.setAttributeNameAt(i - 1, attributes[i]);
                    metaData.setAttributeContinuousAt(i - 1, continuous[i]);
                    metaData.setAttributeNominalValuesAt(i - 1, nominal[i]);

                    metaData.setAttributeNameAt(i, tmpName);
                    metaData.setAttributeContinuousAt(i, tmpCont);
                    metaData.setAttributeNominalValuesAt(i, tmpNominal);
                }
            }
        }
    }
}