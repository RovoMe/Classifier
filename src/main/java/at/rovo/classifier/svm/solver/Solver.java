package at.rovo.classifier.svm.solver;

import at.rovo.classifier.svm.struct.QMatrix;
import at.rovo.classifier.svm.struct.SolutionInfo;
import at.rovo.classifier.svm.utils.Utils;
import at.rovo.common.Pair;
import java.lang.invoke.MethodHandles;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


//An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
//Solves:
//
//min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//y^T \alpha = \delta
//y_i = +1 or -1
//0 <= alpha_i <= Cp for y_i = 1
//0 <= alpha_i <= Cn for y_i = -1
//
//Given:
//
//Q, p, y, Cp, Cn, and an initial feasible point \alpha
//l is the size of vectors and matrices
//eps is the stopping tolerance
//
//solution will be put in \alpha, objective value will be put in obj
//
public class Solver
{
    private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    private static final byte LOWER_BOUND = 0;
    private static final byte UPPER_BOUND = 1;
    private static final byte FREE = 2;
    
    protected int active_size;
    protected byte[] y;
    protected double[] G; // gradient of objective function
    protected QMatrix Q;
    protected double[] QD;
    protected double eps;
    protected boolean unshrink; // XXX
    protected int l;

    private byte[] alpha_status; // LOWER_BOUND, UPPER_BOUND, FREE
    private double[] alpha;
    private double Cp, Cn;
    private double[] p;
    private int[] active_set;
    private double[] G_bar; // gradient, if we treat free variables as 0

    static final double INF = java.lang.Double.POSITIVE_INFINITY;

    private double getC(int i)
    {
        return (y[i] > 0) ? Cp : Cn;
    }

    private void updateAlphaStatus(int i)
    {
        if (alpha[i] >= getC(i))
        {
            alpha_status[i] = UPPER_BOUND;
        }
        else if (alpha[i] <= 0)
        {
            alpha_status[i] = LOWER_BOUND;
        }
        else
        {
            alpha_status[i] = FREE;
        }
    }

    boolean isUpperBound(int i)
    {
        return alpha_status[i] == UPPER_BOUND;
    }

    boolean isLowerBound(int i)
    {
        return alpha_status[i] == LOWER_BOUND;
    }

    private boolean isFree(int i)
    {
        return alpha_status[i] == FREE;
    }

    void swapIndex(int i, int j)
    {
        Q.swapIndex(i, j);
        Utils.swap(y, i, j);
        Utils.swap(G, i, j);
        Utils.swap(alpha_status, i, j);
        Utils.swap(alpha, i, j);
        Utils.swap(p, i, j);
        Utils.swap(active_set, i, j);
        Utils.swap(G_bar, i, j);
    }

    void reconstructGradient()
    {
        // reconstruct inactive elements of G from G_bar and free variables

        if (active_size == l)
        {
            return;
        }

        int i, j;
        int nr_free = 0;

        for (j = active_size; j < l; j++)
        {
            G[j] = G_bar[j] + p[j];
        }

        for (j = 0; j < active_size; j++)
        {
            if (isFree(j))
            {
                nr_free++;
            }
        }

        if (2 * nr_free < active_size)
        {
            if (LOG.isDebugEnabled())
            {
                LOG.debug("\nWARNING: using -h 0 may be faster\n");
            }
        }

        if (nr_free * l > 2 * active_size * (l - active_size))
        {
            for (i = active_size; i < l; i++)
            {
                float[] Q_i = Q.get_Q(i, active_size);
                for (j = 0; j < active_size; j++)
                {
                    if (isFree(j))
                    {
                        G[i] += alpha[j] * Q_i[j];
                    }
                }
            }
        }
        else
        {
            for (i = 0; i < active_size; i++)
            {
                if (isFree(i))
                {
                    float[] Q_i = Q.get_Q(i, l);
                    double alpha_i = alpha[i];
                    for (j = active_size; j < l; j++)
                    {
                        G[j] += alpha_i * Q_i[j];
                    }
                }
            }
        }
    }

    public void solve(int l, QMatrix Q, double[] p_, byte[] y_, double[] alpha_, double Cp, double Cn, double eps,
                      SolutionInfo si, int shrinking)
    {
        this.l = l;
        this.Q = Q;
        QD = Q.get_QD();
        p = p_.clone();
        y = y_.clone();
        alpha = alpha_.clone();
        this.Cp = Cp;
        this.Cn = Cn;
        this.eps = eps;
        this.unshrink = false;

        // initialize alpha_status
        {
            alpha_status = new byte[l];
            for (int i = 0; i < l; i++)
            {
                updateAlphaStatus(i);
            }
        }

        // initialize active set (for shrinking)
        {
            active_set = new int[l];
            for (int i = 0; i < l; i++)
            {
                active_set[i] = i;
            }
            active_size = l;
        }

        // initialize gradient
        {
            G = new double[l];
            G_bar = new double[l];
            int i;
            for (i = 0; i < l; i++)
            {
                G[i] = p[i];
                G_bar[i] = 0;
            }
            for (i = 0; i < l; i++)
            {
                if (!isLowerBound(i))
                {
                    float[] Q_i = Q.get_Q(i, l);
                    double alpha_i = alpha[i];
                    int j;
                    for (j = 0; j < l; j++)
                    {
                        G[j] += alpha_i * Q_i[j];
                    }
                    if (isUpperBound(i))
                    {
                        for (j = 0; j < l; j++)
                        {
                            G_bar[j] += getC(i) * Q_i[j];
                        }
                    }
                }
            }
        }

        // optimization step
        int iter = 0;
        int max_iter = Math.max(10000000, l > Integer.MAX_VALUE / 100 ? Integer.MAX_VALUE : 100 * l);
        int counter = Math.min(l, 1000) + 1;
        int[] working_set = new int[2];

        while (iter < max_iter)
        {
            // show progress and do shrinking
            if (--counter == 0)
            {
                counter = Math.min(l, 1000);
                if (shrinking != 0)
                {
                    doShrinking();
                }
                if (LOG.isDebugEnabled())
                {
                    LOG.debug(".");
                }
            }

            if (selectWorkingSet(working_set) != 0)
            {
                // reconstruct the whole gradient
                reconstructGradient();
                // reset active set size and check
                active_size = l;
                if (LOG.isDebugEnabled())
                {
                    LOG.debug("*");
                }
                if (selectWorkingSet(working_set) != 0)
                {
                    break;
                }
                else
                {
                    counter = 1; // do shrinking next iteration
                }
            }

            int i = working_set[0];
            int j = working_set[1];

            ++iter;

            // update alpha[i] and alpha[j], handle bounds carefully
            float[] Q_i = Q.get_Q(i, active_size);
            float[] Q_j = Q.get_Q(j, active_size);

            double C_i = getC(i);
            double C_j = getC(j);

            double old_alpha_i = alpha[i];
            double old_alpha_j = alpha[j];

            if (y[i] != y[j])
            {
                double quad_coef = QD[i] + QD[j] + 2 * Q_i[j];
                if (quad_coef <= 0)
                {
                    quad_coef = 1e-12;
                }
                double delta = (-G[i] - G[j]) / quad_coef;
                double diff = alpha[i] - alpha[j];
                alpha[i] += delta;
                alpha[j] += delta;

                if (diff > 0)
                {
                    if (alpha[j] < 0)
                    {
                        alpha[j] = 0;
                        alpha[i] = diff;
                    }
                }
                else
                {
                    if (alpha[i] < 0)
                    {
                        alpha[i] = 0;
                        alpha[j] = -diff;
                    }
                }
                if (diff > C_i - C_j)
                {
                    if (alpha[i] > C_i)
                    {
                        alpha[i] = C_i;
                        alpha[j] = C_i - diff;
                    }
                }
                else
                {
                    if (alpha[j] > C_j)
                    {
                        alpha[j] = C_j;
                        alpha[i] = C_j + diff;
                    }
                }
            }
            else
            {
                double quad_coef = QD[i] + QD[j] - 2 * Q_i[j];
                if (quad_coef <= 0)
                {
                    quad_coef = 1e-12;
                }
                double delta = (G[i] - G[j]) / quad_coef;
                double sum = alpha[i] + alpha[j];
                alpha[i] -= delta;
                alpha[j] += delta;

                setAlpha(i, j, C_i, sum);
                setAlpha(j, i, C_j, sum);
            }

            // update G
            double delta_alpha_i = alpha[i] - old_alpha_i;
            double delta_alpha_j = alpha[j] - old_alpha_j;

            for (int k = 0; k < active_size; k++)
            {
                G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
            }

            // update alpha_status and G_bar
            {
                boolean ui = isUpperBound(i);
                boolean uj = isUpperBound(j);
                updateAlphaStatus(i);
                updateAlphaStatus(j);
                setGbarForUpperBound(l, Q, i, C_i, ui);
                setGbarForUpperBound(l, Q, j, C_j, uj);
            }

        }

        if (iter >= max_iter)
        {
            if (active_size < l)
            {
                // reconstruct the whole gradient to calculate objective value
                reconstructGradient();
                active_size = l;
                if (LOG.isDebugEnabled())
                {
                    LOG.debug("*");
                }
            }
            System.err.print("\nWARNING: reaching max number of iterations\n");
        }

        // calculate rho
        si.rho = calculateRho();

        // calculate objective value
        {
            double v = 0;
            int i;
            for (i = 0; i < l; i++)
            {
                v += alpha[i] * (G[i] + p[i]);
            }

            si.obj = v / 2;
        }

        // put back the solution
        {
            for (int i = 0; i < l; i++)
            {
                alpha_[active_set[i]] = alpha[i];
            }
        }

        si.upper_bound_p = Cp;
        si.upper_bound_n = Cn;

        if (LOG.isDebugEnabled())
        {
            LOG.debug("\noptimization finished, #iter = " + iter + "\n");
        }
    }

    private void setGbarForUpperBound(int l, QMatrix Q, int n, double c_n, boolean u_n)
    {
        float[] Q_n;
        int k;
        if (u_n != isUpperBound(n))
        {
            Q_n = Q.get_Q(n, l);
            if (u_n)
            {
                for (k = 0; k < l; k++)
                {
                    G_bar[k] -= c_n * Q_n[k];
                }
            }
            else
            {
                for (k = 0; k < l; k++)
                {
                    G_bar[k] += c_n * Q_n[k];
                }
            }
        }
    }

    private void setAlpha(int i, int j, double c_n, double sum)
    {
        if (sum > c_n)
        {
            if (alpha[i] > c_n)
            {
                alpha[i] = c_n;
                alpha[j] = sum - c_n;
            }
        }
        else
        {
            if (alpha[j] < 0)
            {
                alpha[j] = 0;
                alpha[i] = sum;
            }
        }
    }

    // return 1 if already optimal, return 0 otherwise
    int selectWorkingSet(int[] working_set)
    {
        // return i,j such that
        // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        // j: mimimizes the decrease of obj value
        // (if quadratic coefficeint <= 0, replace it with tau)
        // -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

        double Gmax = -INF;
        double Gmax2 = -INF;
        int Gmax_idx = -1;
        int Gmin_idx = -1;
        double obj_diff_min = INF;

        Pair<Double, Integer> gmax = calculateGmax(Gmax, Gmax_idx);
        Gmax = gmax.getFirst();
        Gmax_idx = gmax.getLast();

        int i = Gmax_idx;
        float[] Q_i = null;
        if (i != -1) // null Q_i not accessed: Gmax=-INF if i=-1
        {
            Q_i = Q.get_Q(i, active_size);
        }

        for (int j = 0; j < active_size; j++)
        {
            if (y[j] == +1)
            {
                if (!isLowerBound(j))
                {
                    double grad_diff = Gmax + G[j];
                    if (G[j] >= Gmax2)
                    {
                        Gmax2 = G[j];
                    }
                    if (grad_diff > 0)
                    {
                        double obj_diff;
                        double quad_coef = QD[i] + QD[j] - 2.0 * y[i] * Q_i[j];
                        if (quad_coef > 0)
                        {
                            obj_diff = -(grad_diff * grad_diff) / quad_coef;
                        }
                        else
                        {
                            obj_diff = -(grad_diff * grad_diff) / 1e-12;
                        }

                        if (obj_diff <= obj_diff_min)
                        {
                            Gmin_idx = j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            }
            else
            {
                if (!isUpperBound(j))
                {
                    double grad_diff = Gmax - G[j];
                    if (-G[j] >= Gmax2)
                    {
                        Gmax2 = -G[j];
                    }
                    if (grad_diff > 0)
                    {
                        double obj_diff;
                        double quad_coef = QD[i] + QD[j] + 2.0 * y[i] * Q_i[j];
                        if (quad_coef > 0)
                        {
                            obj_diff = -(grad_diff * grad_diff) / quad_coef;
                        }
                        else
                        {
                            obj_diff = -(grad_diff * grad_diff) / 1e-12;
                        }

                        if (obj_diff <= obj_diff_min)
                        {
                            Gmin_idx = j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            }
        }

        if (Gmax + Gmax2 < eps)
        {
            return 1;
        }

        working_set[0] = Gmax_idx;
        working_set[1] = Gmin_idx;
        return 0;
    }

    protected Pair<Double, Integer> calculateGmax(double Gmax, int Gmax_idx)
    {
        Pair<Double, Integer> ret = new Pair<>();
        for (int t = 0; t < active_size; t++)
        {
            if (y[t] == +1)
            {
                if (!isUpperBound(t))
                {
                    if (-G[t] >= Gmax)
                    {
                        Gmax = -G[t];
                        Gmax_idx = t;
                    }
                }
            }
            else
            {
                if (!isLowerBound(t))
                {
                    if (G[t] >= Gmax)
                    {
                        Gmax = G[t];
                        Gmax_idx = t;
                    }
                }
            }
        }
        ret.setFirst(Gmax);
        ret.setLast(Gmax_idx);

        return ret;
    }

    private boolean beShrunk(int i, double Gmax1, double Gmax2)
    {
        return beShrunk(i, Gmax1, Gmax2, Gmax1, Gmax2);
    }

    protected boolean beShrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
    {
        if (isUpperBound(i))
        {
            if (y[i] == +1)
            {
                return (-G[i] > Gmax1);
            }
            else
            {
                return (-G[i] > Gmax4);
            }
        }
        else if (isLowerBound(i))
        {
            if (y[i] == +1)
            {
                return (G[i] > Gmax2);
            }
            else
            {
                return (G[i] > Gmax3);
            }
        }
        else
        {
            return (false);
        }
    }

    void doShrinking()
    {
        int i;
        double Gmax1 = -INF; // max { -y_i * grad(f)_i | i in I_up(\alpha) }
        double Gmax2 = -INF; // max { y_i * grad(f)_i | i in I_low(\alpha) }

        // find maximal violating pair first
        for (i = 0; i < active_size; i++)
        {
            if (y[i] == +1)
            {
                if (!isUpperBound(i))
                {
                    if (-G[i] >= Gmax1)
                    {
                        Gmax1 = -G[i];
                    }
                }
                if (!isLowerBound(i))
                {
                    if (G[i] >= Gmax2)
                    {
                        Gmax2 = G[i];
                    }
                }
            }
            else
            {
                if (!isUpperBound(i))
                {
                    if (-G[i] >= Gmax2)
                    {
                        Gmax2 = -G[i];
                    }
                }
                if (!isLowerBound(i))
                {
                    if (G[i] >= Gmax1)
                    {
                        Gmax1 = G[i];
                    }
                }
            }
        }

        if (!unshrink && Gmax1 + Gmax2 <= eps * 10)
        {
            unshrink = true;
            reconstructGradient();
            active_size = l;
        }

        for (i = 0; i < active_size; i++)
        {
            if (beShrunk(i, Gmax1, Gmax2))
            {
                active_size--;
                while (active_size > i)
                {
                    if (!beShrunk(active_size, Gmax1, Gmax2))
                    {
                        swapIndex(i, active_size);
                        break;
                    }
                    active_size--;
                }
            }
        }
    }

    double calculateRho()
    {
        double r;
        int nr_free = 0;
        double ub = INF, lb = -INF, sum_free = 0;
        for (int i = 0; i < active_size; i++)
        {
            double yG = y[i] * G[i];

            if (isLowerBound(i))
            {
                if (y[i] > 0)
                {
                    ub = Math.min(ub, yG);
                }
                else
                {
                    lb = Math.max(lb, yG);
                }
            }
            else if (isUpperBound(i))
            {
                if (y[i] < 0)
                {
                    ub = Math.min(ub, yG);
                }
                else
                {
                    lb = Math.max(lb, yG);
                }
            }
            else
            {
                ++nr_free;
                sum_free += yG;
            }
        }

        if (nr_free > 0)
        {
            r = sum_free / nr_free;
        }
        else
        {
            r = (ub + lb) / 2;
        }

        return r;
    }
}