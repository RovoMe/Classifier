package at.rovo.classifier.dataset;

/**
 * <p>Specifies which file format to use for loading the data set.</p>
 * <p>By default the UCI data set will be loaded.</p>
 * 
 * @author Roman Vottner
 */
public enum DataSetType
{
	/**
	 * <p>
	 * Specifies to load data from a UCI file set, which contains of a names and
	 * a data file where the names file contains the meta data describing the
	 * attributes while the data file contains the observations.
	 * </p>
	 * <p>This is the default data set type.</p>
	 */
	UCI,

	/**
	 * <p>
	 * Specifies to load data from an Attribute-Relation File Format (ARFF) data
	 * file which contain meta data and observations in the same file.
	 * </p>
	 */
	ARFF
}
