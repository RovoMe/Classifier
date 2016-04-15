/**
 * @(#)Stopwatch.java 1.5.2 09/03/29
 */

package at.rovo.classifier.decissionTrees.c45.util;

/**
 * A stopwatch for timing
 *
 * @author Xiaohua Xu
 * @author Ping He
 */
public class Stopwatch
{
    /** The possible statuses of the stopwatch. **/
    private enum Status
    {
        start,
        run,
        pause,
        stop
    }

    /** The start time of the recorded time interval. **/
    private static long startTime;
    /** The end time of the recorded time interval. **/
    private static long endTime;
    /** The recorded time interval. **/
    private static long interval = 0;
    /** The default status of the stopwatch is stop. **/
    private static Status status = Status.stop;

    /**
     * Initialize a stopwatch.
     */
    private Stopwatch()
    {
        reset();
    }

    /**
     * Restart to time.
     */
    public static void reset()
    {
        interval = 0;
        setStatus(Status.start);
        startTime = System.currentTimeMillis();
    }

    /**
     * Start to time.
     */
    public static void start()
    {
        interval = 0;
        setStatus(Status.run);
        startTime = System.currentTimeMillis();
    }

    /**
     * Start to time. <br /> An alternative method of <em>start</em>().
     */
    public static void run()
    {
        if (status.equals(Status.pause) || status.equals(Status.start))
        {
            startTime = System.currentTimeMillis();
            setStatus(Status.run);
        }
    }

    /**
     * Pause to time.
     */
    public static void pause()
    {
        if (status.equals(Status.run))
        {
            endTime = System.currentTimeMillis();
            interval += endTime - startTime;
            setStatus(Status.pause);
        }
    }

    /**
     * Stop to time.
     */
    public static void stop()
    {
        if (status.equals(Status.run) || status.equals(Status.start))
        {
            endTime = System.currentTimeMillis();
            interval += endTime - startTime;
        }
        setStatus(Status.stop);
    }

    /**
     * Get the time interval.
     */
    public static long runtime()
    {
        if (status.equals(Status.run))
        {
            return System.currentTimeMillis() - startTime;
        }
        else
        {
            return interval;
        }
    }

    /**
     * Get the time interval. <br> An alternative method of <em>runtime</em>().
     */
    public static long interval()
    {
        if (status.equals(Status.run))
        {
            return System.currentTimeMillis() - startTime;
        }
        else
        {
            return interval;
        }
    }

    /**
     * Get the current state of the stopwatch.
     */
    public static Status getCurrentStatus()
    {
        return status;
    }

    /**
     * Set the current state of the stopwatch.
     */
    private static void setStatus(Status s)
    {
        status = s;
    }
}