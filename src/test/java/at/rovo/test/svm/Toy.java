package at.rovo.test.svm;

import java.applet.Applet;
import java.awt.AWTEvent;
import java.awt.BorderLayout;
import java.awt.Button;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FileDialog;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Image;
import java.awt.Panel;
import java.awt.TextField;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.util.Vector;
import at.rovo.classifier.svm.KernelType;
import at.rovo.classifier.svm.Model;
import at.rovo.classifier.svm.SVM;
import at.rovo.classifier.svm.SVMType;
import at.rovo.classifier.svm.struct.Node;
import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.Problem;

/**
 * <p></p>
 * 
 * @author Chih-Chung Chang, Chih-Jen Lin
 */
public class Toy extends Applet
{
	/** Unique identifier necessary for serialization **/
	private static final long serialVersionUID = 3249842143206389271L;
	private Parameter param;
	static final String DEFAULT_PARAM = "-t 2 -c 100";
	int XLEN;
	int YLEN;

	// off-screen buffer

	Image buffer;
	Graphics buffer_gc;

	// pre-allocated colors

	final static Color colors[] = { new Color(0, 0, 0), new Color(0, 120, 120),
			new Color(120, 120, 0), new Color(120, 0, 120),
			new Color(0, 200, 200), new Color(200, 200, 0),
			new Color(200, 0, 200) };

	class point
	{
		point(double x, double y, byte value)
		{
			this.x = x;
			this.y = y;
			this.value = value;
		}

		double x, y;
		byte value;
	}

	Vector<point> point_list = new Vector<point>();
	byte current_value = 1;

	/**
	 * <p>Initializes the Applet.</p>
	 */
	public void init()
	{	
		// default values
		this.param = Parameter.create(new String[0]);
		this.param.cache_size = 40;
		
		setSize(getSize());

		final Button button_change = new Button("Change");
		Button button_run = new Button("Run");
		Button button_clear = new Button("Clear");
		Button button_save = new Button("Save");
		Button button_load = new Button("Load");
		final TextField input_line = new TextField(DEFAULT_PARAM);

		BorderLayout layout = new BorderLayout();
		this.setLayout(layout);

		Panel p = new Panel();
		GridBagLayout gridbag = new GridBagLayout();
		p.setLayout(gridbag);

		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.HORIZONTAL;
		c.weightx = 1;
		c.gridwidth = 1;
		gridbag.setConstraints(button_change, c);
		gridbag.setConstraints(button_run, c);
		gridbag.setConstraints(button_clear, c);
		gridbag.setConstraints(button_save, c);
		gridbag.setConstraints(button_load, c);
		c.weightx = 5;
		c.gridwidth = 5;
		gridbag.setConstraints(input_line, c);

		button_change.setBackground(colors[current_value]);

		p.add(button_change);
		p.add(button_run);
		p.add(button_clear);
		p.add(button_save);
		p.add(button_load);
		p.add(input_line);
		this.add(p, BorderLayout.SOUTH);

		button_change.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e)
			{
				button_change_clicked();
				button_change.setBackground(colors[current_value]);
			}
		});

		button_run.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e)
			{
				button_run_clicked(input_line.getText());
			}
		});

		button_clear.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e)
			{
				button_clear_clicked();
			}
		});

		button_save.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e)
			{
				SVMType svmType = SVMType.C_SVC;
				int svm_type_idx = input_line.getText().indexOf("-s ");
				if (svm_type_idx != -1)
				{
					StringTokenizer svm_str_st = new StringTokenizer(
							input_line.getText().substring(svm_type_idx + 2).trim());
					svmType = SVMType.get(svm_str_st.nextToken());
				}
				
				button_save_clicked(svmType);
			}
		});

		button_load.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e)
			{
				button_load_clicked();
			}
		});

		input_line.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e)
			{
				button_run_clicked(input_line.getText());
			}
		});

		this.enableEvents(AWTEvent.MOUSE_EVENT_MASK);
	}

	void draw_point(point p)
	{
		Color c = colors[p.value + 3];

		Graphics window_gc = getGraphics();
		buffer_gc.setColor(c);
		buffer_gc.fillRect((int) (p.x * XLEN), (int) (p.y * YLEN), 4, 4);
		window_gc.setColor(c);
		window_gc.fillRect((int) (p.x * XLEN), (int) (p.y * YLEN), 4, 4);
	}

	void clear_all()
	{
		point_list.removeAllElements();
		if (buffer != null)
		{
			buffer_gc.setColor(colors[0]);
			buffer_gc.fillRect(0, 0, XLEN, YLEN);
		}
		repaint();
	}

	void draw_all_points()
	{
		int n = point_list.size();
		for (int i = 0; i < n; i++)
			draw_point(point_list.elementAt(i));
	}

	void button_change_clicked()
	{
		++current_value;
		if (current_value > 3)
			current_value = 1;
	}

	private static double atof(String s)
	{
		return Double.valueOf(s).doubleValue();
	}

	private static int atoi(String s)
	{
		return Integer.parseInt(s);
	}
	
	/**
	 * <p>Updates the parameter object according to the input values in the
	 * corresponding TextField.</p>
	 * 
	 * @param args The text-value of the TextField containing the parameters
	 * @return The updated parameter object
	 */
	private Parameter parseParameter(String args)
	{
		StringTokenizer st = new StringTokenizer(args);
		String[] argv = new String[st.countTokens()];
		for (int i = 0; i < argv.length; i++)
			argv[i] = st.nextToken();
		
		return Parameter.update(this.param, argv);
	}

	/**
	 * <p>Trains the SVM classifier with the points and their color, where the
	 * point itself is a feature of a class according to the color of the point
	 * is has. So points with the same color are features of the same class.</p>
	 * <p>After training a model is extracted and visualized in the painting
	 * area.</p>
	 * 
	 * @param args
	 */
	void button_run_clicked(String args)
	{	
		// guard
		if (point_list.isEmpty())
			return;

		this.param = this.parseParameter(args);
		if (this.param == null)
		{
			System.err.println("Parameter is null");
			return;
		}
		
		// build problem
		Problem prob = new Problem();
		prob.numInstances = point_list.size();
		prob.y = new ArrayList<Double>(prob.numInstances);

		if (KernelType.PRECOMPUTED.equals(param.kernelType))
		{
			
		}
		// Regression
		else if (SVMType.EPSILON_SVR.equals(param.svmType)
				|| SVMType.NU_SVR.equals(param.svmType))
		{			
			if (this.param.gamma == 0)
				this.param.gamma = 1;
			
			SVM svm = new SVM(this.param);
			
			// train SVM with the points. Internally they are added to the
			// Problem instance which is a child of TrainingData
			prob.x = new ArrayList<Node[]>(prob.numInstances);
			for (int i = 0; i < prob.numInstances; i++)
			{
				point p = point_list.elementAt(i);
				prob.x.add(new Node[1]);
				prob.x.get(i)[0] = new Node();
				prob.x.get(i)[0].index = 1;
				prob.x.get(i)[0].value = p.x;
				prob.y.add(p.y);
				svm.train(prob.x.get(i), p.y);
			}
			
			// a model is build based on the available and known training data
			// (samples inside the Problem). The model is than used to classify
			// points.
			Model model = svm.getTrainedModel();
			
			// creates a single point
			Node[] x = new Node[1];
			x[0] = new Node();
			x[0].index = 1;
			// j will store the predicted class of each point in the respective
			// line
			int[] j = new int[XLEN];
			// runs through every point of the line and normalizes the value
			// of each point. Subsequently the point is classified and the result
			// is stored in j
			for (int i = 0; i < XLEN; i++)
			{
				x[0].value = (double) i / XLEN;
				j[i] = (int) (YLEN * model.predict(x));
			}

			buffer_gc.setColor(colors[0]);
			buffer_gc.drawLine(0, 0, 0, YLEN - 1);
			
			Graphics window_gc = getGraphics();
			window_gc.setColor(colors[0]);
			window_gc.drawLine(0, 0, 0, YLEN - 1);

			// scales the measured costs of errors with the height of the 
			// canvas
			int p = (int) (this.param.p * YLEN);
			for (int i = 1; i < XLEN; i++)
			{
				buffer_gc.setColor(colors[0]);
				buffer_gc.drawLine(i, 0, i, YLEN - 1);
				window_gc.setColor(colors[0]);
				window_gc.drawLine(i, 0, i, YLEN - 1);

				buffer_gc.setColor(colors[5]);
				window_gc.setColor(colors[5]);
				buffer_gc.drawLine(i - 1, j[i - 1], i, j[i]);
				window_gc.drawLine(i - 1, j[i - 1], i, j[i]);

				if (SVMType.EPSILON_SVR.equals(this.param.svmType))
				{
					buffer_gc.setColor(colors[2]);
					window_gc.setColor(colors[2]);
					buffer_gc.drawLine(i - 1, j[i - 1] + p, i, j[i] + p);
					window_gc.drawLine(i - 1, j[i - 1] + p, i, j[i] + p);

					buffer_gc.setColor(colors[2]);
					window_gc.setColor(colors[2]);
					buffer_gc.drawLine(i - 1, j[i - 1] - p, i, j[i] - p);
					window_gc.drawLine(i - 1, j[i - 1] - p, i, j[i] - p);
				}
			}
		}
		// Classification
		else
		{			
			if (this.param.gamma == 0)
				this.param.gamma = 0.5;
			
			SVM svm = new SVM(param);
			
			// train SVM with the points. Internally they are added to the
			// Problem instance which is a child of TrainingData
			prob.x = new ArrayList<Node[]>(prob.numInstances);
			for (int i = 0; i < prob.numInstances; i++)
			{
				point p = point_list.elementAt(i);
				prob.x.add(new Node[2]);
				prob.x.get(i)[0] = new Node();
				prob.x.get(i)[0].index = 1;
				prob.x.get(i)[0].value = p.x;
				prob.x.get(i)[1] = new Node();
				prob.x.get(i)[1].index = 2;
				prob.x.get(i)[1].value = p.y;
				prob.y.add((double)p.value);
				
				svm.train(prob.x.get(i), (double)p.value);
			}

			// a model is build based on the available and known training data
			// (samples inside the Problem). The model is than used to classify
			// points.
			Model model = svm.getTrainedModel();
			
			// creates a two point which represent a vector in a space
			Node[] x = new Node[2];
			x[0] = new Node();
			x[1] = new Node();
			x[0].index = 1;
			x[1].index = 2;

			Graphics window_gc = getGraphics();
			// runs through every point and classifies it according to the 
			// trained model. Every point is drawn in the color of the class
			// it belongs to.
			for (int i = 0; i < XLEN; i++)
				for (int j = 0; j < YLEN; j++)
				{
					// runs through every point of the line and normalizes the 
					// value of each point.
					x[0].value = (double) i / XLEN;
					x[1].value = (double) j / YLEN;
					// predicts the class the vector belongs to
					double d = model.predict(x);
					if (SVMType.ONE_CLASS.equals(param.svmType) && d < 0)
						d = 2;
					// sets the color to the color of the predicted class
					buffer_gc.setColor(colors[(int) d]);
					window_gc.setColor(colors[(int) d]);
					// draws the current point with the specified color above
					buffer_gc.drawLine(i, j, i, j);
					window_gc.drawLine(i, j, i, j);
				}
		}

		// repaint all colors else they won't be visible anymore
		draw_all_points();
	}

	/**
	 * <p>Clears the loaded or drawn points.</p>
	 */
	void button_clear_clicked()
	{
		clear_all();
	}

	/**
	 * <p>Saves the drawn points to disk file. </p>
	 * 
	 * @param The selected SVM type
	 */
	void button_save_clicked(SVMType svmType)
	{
		FileDialog dialog = new FileDialog(new Frame(), "Save", FileDialog.SAVE);
		dialog.setVisible(true);
		String filename = dialog.getDirectory() + dialog.getFile();
		if (filename != null)
		{
			try
			{
				DataOutputStream fp = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filename)));
		
				int n = point_list.size();
				if (SVMType.EPSILON_SVR.equals(svmType)	|| SVMType.NU_SVR.equals(svmType))
				{
					for (int i = 0; i < n; i++)
					{
						point p = point_list.elementAt(i);
						fp.writeBytes(p.y + " 1:" + p.x + "\n");
					}
				}
				else
				{
					for (int i = 0; i < n; i++)
					{
						point p = point_list.elementAt(i);
						fp.writeBytes(p.value + " 1:" + p.x + " 2:" + p.y + "\n");
					}
				}
				fp.close();
			}
			catch (IOException e)
			{
				System.err.print(e);
			}
		}
	}

	/**
	 * <p>Loads a previously saved set of points including their classes.</p>
	 */
	void button_load_clicked()
	{
		FileDialog dialog = new FileDialog(new Frame(), "Load", FileDialog.LOAD);
		dialog.setVisible(true);
		String filename = dialog.getDirectory() + dialog.getFile();
		if (filename != null)
		{
			clear_all();
			try
			{
				BufferedReader fp = new BufferedReader(new FileReader(filename));
				String line;
				while ((line = fp.readLine()) != null)
				{
					StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
					if (st.countTokens() == 5)
					{
						byte value = (byte) atoi(st.nextToken());
						st.nextToken();
						double x = atof(st.nextToken());
						st.nextToken();
						double y = atof(st.nextToken());
						point_list.addElement(new point(x, y, value));
					}
					else if (st.countTokens() == 3)
					{
						double y = atof(st.nextToken());
						st.nextToken();
						double x = atof(st.nextToken());
						point_list.addElement(new point(x, y, current_value));
					}
					else
						break;
				}
				fp.close();
			}
			catch (IOException e)
			{
				System.err.print(e);
			}
			draw_all_points();
		}
	}

	@Override
	protected void processMouseEvent(MouseEvent e)
	{
		if (e.getID() == MouseEvent.MOUSE_PRESSED)
		{
			if (e.getX() >= XLEN || e.getY() >= YLEN)
				return;
			point p = new point((double) e.getX() / XLEN, (double) e.getY()
					/ YLEN, current_value);
			point_list.addElement(p);
			draw_point(p);
		}
	}

	@Override
	public void paint(Graphics g)
	{
		// create buffer first time
		if (buffer == null)
		{
			buffer = this.createImage(XLEN, YLEN);
			buffer_gc = buffer.getGraphics();
			buffer_gc.setColor(colors[0]);
			buffer_gc.fillRect(0, 0, XLEN, YLEN);
		}
		g.drawImage(buffer, 0, 0, this);
	}

	@Override
	public Dimension getPreferredSize()
	{
		return new Dimension(XLEN, YLEN + 50);
	}

	@Override
	public void setSize(Dimension d)
	{
		setSize(d.width, d.height);
	}

	@Override
	public void setSize(int w, int h)
	{
		super.setSize(w, h);
		XLEN = w;
		YLEN = h - 50;
		clear_all();
	}

	public static void main(String[] argv)
	{
		new AppletFrame("SVMToy", new Toy(), 500, 500 + 50);
	}
}

class AppletFrame extends Frame
{
	/** Unique identifier necessary for serialization **/
	private static final long serialVersionUID = 618401824505110823L;

	AppletFrame(String title, Applet applet, int width, int height)
	{
		super(title);
		this.addWindowListener(new WindowAdapter()
		{
			public void windowClosing(WindowEvent e)
			{
				System.exit(0);
			}
		});
		applet.init();
		applet.setSize(width, height);
		applet.start();
		this.add(applet);
		this.pack();
		this.setVisible(true);
	}
}
