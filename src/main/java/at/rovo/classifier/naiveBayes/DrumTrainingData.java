package at.rovo.classifier.naiveBayes;

import at.rovo.caching.drum.data.AppendableData;
import at.rovo.caching.drum.util.DrumUtils;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DrumTrainingData<F, C> implements AppendableData<DrumTrainingData<F, C>>
{
    private F feature;
    private Class<F> featureClass;
    private Map<C, Integer> catCount;
    private Class<C> categoryClass;

    public DrumTrainingData(F feature, Class<F> featureClass, Map<C, Integer> catCount, Class<C> categoryClass)
    {
        this.feature = feature;
        this.featureClass = featureClass;
        this.catCount = catCount;
        this.categoryClass = categoryClass;
    }

    public F getFeature()
    {
        return this.feature;
    }

    public void setFeature(F feature)
    {
        this.feature = feature;
    }

    public Integer getCategoryCount(C category)
    {
        return this.catCount.get(category);
    }

    public void setCategoryCount(C category, Integer in)
    {
        this.catCount.put(category, in);
    }

    //@Override
    public void append(DrumTrainingData<F, C> data)
    {
        for (C category : data.catCount.keySet())
        {
            int val = this.catCount.get(category) + data.catCount.get(category);
            this.catCount.put(category, val);
        }
    }

    @Override
    public byte[] toBytes()
    {
        // 4 bytes int - size of the feature
        // n bytes char[] - feature
        // 4 bytes number of categories
        // next steps loop for every category
        // 4 bytes int - size of cat-name
        // n bytes char[] - category name
        // 4 bytes int - category value
        byte[] featureBytes = convertToBytes(this.feature);
        int featureSize = featureBytes.length;

        byte[][] catBytes = new byte[this.catCount.keySet().size()][];
        List<C> cats = new ArrayList<>();
        int i = 0;
        int catSize = 0;
        for (C category : this.catCount.keySet())
        {
            catBytes[i] = this.convertToBytes(category);
            // 4 bytes for the length of the cat-name, n bytes for the cat name and 4 bytes for the cat-count
            catSize += 8 + catBytes[i].length;
            cats.add(category);
            i++;
        }

        byte[] totalBytes = new byte[featureSize + 12 + catSize];

        int pos = 0;
        // copy size of feature into the byte array
        System.arraycopy(DrumUtils.int2bytes(featureSize), 0, totalBytes, pos, pos += 4);
        // copy feature characters into the byte array
        System.arraycopy(featureBytes, 0, totalBytes, pos, pos += featureSize);
        // copy the number of categories into the byte array
        System.arraycopy(DrumUtils.int2bytes(i + 1), 0, totalBytes, pos, pos += 4);

        for (i = 0; i < cats.size(); i++)
        {
            byte[] catNameBytes = catBytes[i];
            // copy the length of the cat name into the byte array
            System.arraycopy(DrumUtils.int2bytes(catNameBytes.length), 0, totalBytes, pos, pos += 4);
            // copy the category name into the byte array
            System.arraycopy(catNameBytes, 0, totalBytes, pos, pos += catNameBytes.length);
            // copy the count of the category into the byte array
            System.arraycopy(DrumUtils.int2bytes(this.catCount.get(cats.get(i))), 0, totalBytes, pos, pos += 4);
        }
        return totalBytes;
    }

    @Override
    public DrumTrainingData<F, C> readBytes(byte[] data)
    {
        int pos = 0;
        // read feature length
        byte[] featureSizeBytes = new byte[4];
        System.arraycopy(data, pos, featureSizeBytes, 0, 4);
        pos += 4;
        int featureSize = DrumUtils.bytes2int(featureSizeBytes);
        // read the feature
        byte[] featureBytes = new byte[featureSize];
        System.arraycopy(data, pos, featureBytes, 0, featureSize);
        pos += featureSize;
        F feature = this.convertBytesToObject(featureBytes, featureClass);
        // read the number of categories
        byte[] numCatBytes = new byte[4];
        System.arraycopy(data, pos, numCatBytes, 0, 4);
        pos += 4;
        int numCat = DrumUtils.bytes2int(numCatBytes);

        Map<C, Integer> catCounts = new HashMap<>(numCat);
        for (int i = 0; i < numCat; i++)
        {
            // read the length of the category name
            byte[] catNameLengthBytes = new byte[4];
            System.arraycopy(data, pos, catNameLengthBytes, 0, 4);
            pos += 4;
            int catNameLength = DrumUtils.bytes2int(catNameLengthBytes);
            // read the category name
            byte[] catNameBytes = new byte[catNameLength];
            System.arraycopy(data, pos, catNameBytes, 0, catNameLength);
            pos += catNameLength;
            C category = this.convertBytesToObject(catNameBytes, categoryClass);
            // reat the count for the category
            byte[] catCountBytes = new byte[4];
            System.arraycopy(data, pos, catCountBytes, 0, 4);
            pos += 4;
            int catCount = DrumUtils.bytes2int(catCountBytes);
            // add the extracted values to the map
            catCounts.put(category, catCount);
        }

        // create a new object with the deserialized data
        return new DrumTrainingData<>(feature, featureClass, catCounts, categoryClass);
    }

    @Override
    public boolean equals(Object obj)
    {
        if (obj instanceof DrumTrainingData)
        {
            DrumTrainingData data = (DrumTrainingData) obj;
            if (data.feature.equals(this.feature) && data.catCount.equals(this.catCount))
            {
                return true;
            }
        }
        return false;
    }

    @Override
    public int hashCode()
    {
        int result = 17;
        result = 31 * result + feature.hashCode();
        result = 31 * result + catCount.hashCode();
        return result;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("categories: { ");
        for (C category : this.catCount.keySet())
        {
            sb.append(category.toString());
            sb.append(":");
            sb.append(this.catCount.get(category));
            sb.append(", ");
        }
        sb.delete(sb.length() - 2, sb.length());
        sb.append(" }, feature: ");
        sb.append(this.feature);
        return sb.toString();
    }

    private byte[] convertToBytes(Object obj)
    {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream())
        {
            try (ObjectOutputStream oos = new ObjectOutputStream(baos))
            {
                oos.writeObject(obj);
                return baos.toByteArray();
            }
            catch (IOException ioEx)
            {
                throw new RuntimeException("Could not serialize object", ioEx);
            }
        }
        catch (IOException ioEx)
        {
            throw new RuntimeException("Could not convert object to byte[]", ioEx);
        }
    }

    private <T> T convertBytesToObject(byte[] bytes, Class<T> typeClass)
    {
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes)))
        {
            return typeClass.cast(ois.readObject());
        }
        catch (IOException | ClassNotFoundException ex)
        {
            throw new RuntimeException("Could not serialize bytes back to object", ex);
        }
    }
}
