package at.rovo.classifier.dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A data set loaded from an Attribute-Relation File Format (ARFF) data file.
 *
 * @author Roman Vottner
 */
public class ArffDataSet extends DataSet
{
	/*
	 * % 1. Title: Iris Plants Database % % 2. Sources: % (a) Creator: R.A.
	 * Fisher % (b) Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov) % (c)
	 * Date: July, 1988 %
	 * 
	 * @RELATION iris
	 * 
	 * @ATTRIBUTE sepallength NUMERIC
	 * 
	 * @ATTRIBUTE sepalwidth NUMERIC
	 * 
	 * @ATTRIBUTE petallength NUMERIC
	 * 
	 * @ATTRIBUTE petalwidth NUMERIC
	 * 
	 * @ATTRIBUTE class {Iris-setosa,Iris-versicolor,Iris-virginica}
	 * 
	 * @DATA 5.1,3.5,1.4,0.2,Iris-setosa 4.9,3.0,1.4,0.2,Iris-setosa
	 * 4.7,3.2,1.3,0.2,Iris-setosa 4.6,3.1,1.5,0.2,Iris-setosa
	 * 5.0,3.6,1.4,0.2,Iris-setosa 5.4,3.9,1.7,0.4,Iris-setosa
	 * 4.6,3.4,1.4,0.3,Iris-setosa 5.0,3.4,1.5,0.2,Iris-setosa
	 * 4.4,2.9,1.4,0.2,Iris-setosa 4.9,3.1,1.5,0.1,Iris-setosa
	 */

    public ArffDataSet(String baseName)
    {
        super(baseName);
        this.addColumnSetView();
    }

    /**
     * Loads the data contained in the UCI data file.
     *
     * @param baseName
     *         The file name to load the data from
     */
    public void load(String baseName)
    {
        // @ATTRIBUTE petalwidth NUMERIC
        // @ATTRIBUTE class {Iris-setosa,Iris-versicolor,Iris-virginica}

        try (BufferedReader bf = new BufferedReader(new FileReader(baseName)))
        {
            String line;
            boolean content = false;

            List<String> attributes = new ArrayList<>();
            List<Boolean> continuous = new ArrayList<>();
            List<String[]> nominalValues = new ArrayList<>();
            List<String> dateFormat = new ArrayList<>();
            List<String[]> data = new ArrayList<>();

            while ((line = bf.readLine()) != null)
            {
                if (line.trim().length() == 0)
                {
                    continue;
                }

                // check for comments
                if (line.trim().startsWith("%"))
                {
                    continue;
                }

                // MetaData section
                if (!content)
                {
                    if (line.trim().toLowerCase().startsWith("@relation"))
                    {
                        this.setName(line.substring(line.indexOf(" ")));
                    }
                    else if (line.trim().toLowerCase().startsWith("@attribute"))
                    {
                        this.parseAttribute(line, attributes, continuous, nominalValues, dateFormat);
                    }
                    else if (line.trim().toLowerCase().startsWith("@data"))
                    {
                        // no more meta data to expect, so create the
                        // corresponding object
                        MetaData meta =
                                new MetaData(attributes.toArray(new String[0]), continuous.toArray(new Boolean[0]),
                                             nominalValues.toArray(new String[0][]));
                        // TODO: extend meta data so it can hold ordinal data
                        // like dates too
                        this.setMetaData(meta);

                        content = true;
                    }
                }
                // Data section
                else
                {
                    // is the data provided in sparse format?
                    if (line.trim().startsWith("{"))
                    {
                        this.parseSparseData(line, attributes.size(), data);
                    }
                    else
                    {
                        this.parseData(line, attributes.size(), data);
                    }
                    this.setTrainData(data.toArray(new String[0][]));
                }
            }

            bf.close();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    /**
     * Parses a single @ATTRIBUTE indexed line and extracts parameters like the attribute name, its data type and
     * admissible values.
     *
     * @param line
     *         The line to parse
     * @param attributes
     *         A reference to the list that will store the attribute names
     * @param continuous
     *         A reference to the list that will store if the attributes are continuous or not
     * @param nominalValues
     *         A reference to the list that will store the nominal values a argument can have
     * @param dateFormat
     *         A reference to the list that will store the date format the data section will provide
     */
    private void parseAttribute(String line, List<String> attributes, List<Boolean> continuous,
                                List<String[]> nominalValues, List<String> dateFormat)
    {
        String[] tokens = line.split("[ |\\s]");
        attributes.add(tokens[1]);
        int pos = 2;
        while (tokens[pos].equals(""))
        {
            pos++;
        }

        // we found a numeric (continuous) value
        if (tokens[pos].trim().toLowerCase().equals("numeric") || tokens[pos].trim().toLowerCase().equals("real") ||
            tokens[pos].trim().toLowerCase().equals("int"))
        {
            continuous.add(true);
            nominalValues.add(null);
            dateFormat.add(null);
        }
        // we found a nominal value
        else if (tokens[pos].trim().startsWith("{"))
        {
            continuous.add(false);
            dateFormat.add(null);
            List<String> nomVals = new ArrayList<>();
            // there might be whitespaces between the nominal values
            boolean done = false;
            for (int i = pos; i < tokens.length && !done; i++)
            {
                String[] part = tokens[i].split(",");
                for (int j = 0; j < part.length; j++)
                {
                    String nominal = part[j];
                    // remove opening nominal tag
                    if (nominal.startsWith("{"))
                    {
                        nominal = nominal.substring(1);
                    }
                    // if the opening tag was separated from the first
                    // nominal value ignore it
                    if (nominal.equals(""))
                    {
                        continue;
                    }
                    // remove item separators or closing tags
                    if (nominal.trim().endsWith("}"))
                    {
                        done = true;
                        nominal = nominal.substring(0, nominal.length() - 1);
                    }

                    nomVals.add(nominal);
                    if (done)
                    {
                        break;
                    }
                }
            }
            nominalValues.add(nomVals.toArray(new String[0]));

            // set the class index and the class values
            if (tokens[1].equals("class"))
            {
                this.setClassAttributeIndex(attributes.size() - 1);
                this.setClassValues(nomVals.toArray(new String[0]));
            }
        }
        // we found a date
        else if (tokens[pos].trim().toLowerCase().startsWith("date"))
        {
            String dF = tokens[pos + 1].trim();
            dateFormat.add(dF);
            continuous.add(false);
            nominalValues.add(null);
        }
        // we found a sole string
        else
        {
            continuous.add(false);
            nominalValues.add(new String[] {tokens[pos]});
            dateFormat.add(null);
        }
    }

    /**
     * Parses a single line containing training or test data information.
     *
     * @param line
     *         The line to parse
     * @param numAttributes
     *         The number of defined Attributes
     * @param data
     *         A reference to the list that will store the parsed data elements
     */
    private void parseSparseData(String line, int numAttributes, List<String[]> data)
    {
        // @data
        // {1 X, 3 Y, 4 "class A"}
        // {2 W, 4 "class B"}

        // create as many data slots as attributes are available
        String[] arffData = new String[numAttributes];
        line = line.trim().substring(1);
        if (line.endsWith("}"))
        {
            line = line.substring(0, line.length() - 1);
        }
        String[] lineData = line.split(",");
        // variable for the current position in the
        int pos = 0;
        for (String segment : lineData)
        {
            String[] attr = segment.split(" ");
            int num = Integer.parseInt(attr[0]);
            // numbers are entered in sequential order
            if (num > pos)
            {
                for (int p = pos; p < num; p++)
                {
                    arffData[p] = "0";
                }
            }
            arffData[num] = attr[1];
            pos++;
        }
        // fill the remaining values in the array with 0 for
        // attributes that have not been declared
        for (int p = pos; p < numAttributes; p++)
        {
            arffData[p] = "0";
        }

        data.add(arffData);
    }

    /**
     * Parses the observed data for the currently processed line which corresponds to an observation of the experiment.
     *
     * @param line
     * @param numAttributes
     * @param data
     */
    private void parseData(String line, int numAttributes, List<String[]> data)
    {
        // @data
        // 0, X, 0, Y, "class A"
        // 0, 0, W, 0, "class B"

        // create as many data slots as attributes are available
        String[] arffData = new String[numAttributes];

        String[] lineData = line.split(",");
        boolean quoted = false;
        int pos = 0;
        for (int i = 0; i < lineData.length; i++)
        {
            String ele = lineData[i].trim();
            if (!quoted && ele.startsWith("\""))
            {
                quoted = true;
                arffData[pos] = ele.substring(1);
            }
            else if (quoted)
            {
                if (ele.endsWith("\""))
                {
                    quoted = false;
                    arffData[pos] += " " + ele.substring(0, lineData.length - 1);
                    pos++;
                }
                else
                {
                    arffData[pos] += " " + ele;
                }
            }
            else
            {
                arffData[pos] = ele;
                pos++;
            }
        }

        data.add(arffData);
    }
}
