/**
 * @(#)Parameter.java 1.5.2 09/03/29
 */

package at.rovo.classifier.decissionTrees.c45;

/**
 * The default parameter setting in the implementation of C4.5 algorithm
 *
 * @author Ping He
 * @author Xiaohua Xu
 */

public class Parameter
{
    /** Its minus is the default minimal valid Gain value. **/
    public static final float EPSILON = 0.001f;
    /** The default precision in comparing two float values. **/
    public static final double PRECISION = 1E-5;
    /** The default minimal weight to construct an internal tree node. **/
    public static final float MINWEIGHT = 2.0f;

    /**
     * The default confidence level in c4.5's error-based pruning. <p> CONFIDENCE must be changed together with
     * SQUAREDSTD parameter. The setConfidence(float confidence) method in ml.classifier.dt.Estimator can be used to
     * reset confidence and its corresponding SQUAREDSTD. </p>
     */
    public static final float CONFIDENCE = 0.25f;
    /**
     * The default squared standard deviation value corresponding to the default CONFIDENCE. <p> SQUAREDSTD must be
     * changed together with CONFIDENCE parameter. The setConfidence(float confidence) method in
     * ml.classifier.dt.Estimator can be used to reset confidence and its corresponding SQUAREDSTD. </p>
     */
    public static final float SQUAREDSTD = 0.47955623f;

    private Parameter()
    {
    }
}