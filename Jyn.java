import java.io.FileInputStream;
import java.io.FileOutputStream;

import java.io.IOException;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import java.util.ArrayList;
import java.util.function.Function;


public class Jyn {
    private ArrayList<Layer> architecture;
    private ArrayList<Integer> layers;
    private ArrayList<ArrayList<JMatrix>> dataset;
    public JMatrix output;
    private ArrayList<ArrayList<JMatrix>> nn;
    private ArrayList<ArrayList<JMatrix>> gradients;
    private ArrayList<Function<JMatrix, JMatrix>> activation_functions;
    private ArrayList<Function<JMatrix, JMatrix>> activation_derv_functions;

    private static void printnn(ArrayList<ArrayList<JMatrix>> matrixList) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        
        for (int i = 0; i < matrixList.size(); i++) {
            ArrayList<JMatrix> layer = matrixList.get(i);
            sb.append("[");
            for (int j = 0; j < layer.size(); j++) {
                JMatrix matrix = layer.get(j);
                sb.append(matrix.getList().toString());
                if (j < layer.size() - 1) {
                    sb.append(", ");
                }
            }
            sb.append("]");
            if (i < matrixList.size() - 1) {
                sb.append(", ");
            }
        }
        
        sb.append("]");
        System.out.println(sb.toString());
    }

    public Jyn(ArrayList<Layer> architecture)
    {
        this.architecture = architecture;
        this.output = new JMatrix();
        this.layers = new ArrayList<Integer>();
        this.activation_functions = new ArrayList<Function<JMatrix, JMatrix>>();
        this.activation_derv_functions = new ArrayList<Function<JMatrix, JMatrix>>();
        this.nn = new ArrayList<ArrayList<JMatrix>>();
        this.gradients = new ArrayList<ArrayList<JMatrix>>();
        
        for (Layer layer : architecture)
        {
            this.layers.add(layer.size);
            this.activation_functions.add(layer.activation);
            this.activation_derv_functions.add(layer.derivative);
        }
    }

    public static JMatrix sigmoid(JMatrix matrix)
    {
        return JMatrix.divByMatrix(JMatrix.addBy(JMatrix.exp(JMatrix.mulBy(matrix, -1)), 1), 1);
    }

    
    public static JMatrix sigmoid_derivative(JMatrix matrix) {
        JMatrix sigmoidMatrix = sigmoid(matrix);

        JMatrix oneMatrix = JMatrix.ones(matrix.getHeight(), matrix.getWidth());
        return JMatrix.mul(sigmoidMatrix, JMatrix.sub(oneMatrix, sigmoidMatrix));
    }

    public void load_dataset(ArrayList<ArrayList<JMatrix>> dataset)
    {
        this.dataset = dataset;
    }

    public double cost(JMatrix output, JMatrix target)
    {
        return JMatrix.sum(JMatrix.power(JMatrix.sub(output, target),2));
    }

    public void applyGradient(double learnRate) {
        for (int i = 0; i < this.gradients.size(); i++) {

            JMatrix weightGradient = this.gradients.get(i).get(0);
            this.nn.get(i).set(0, JMatrix.sub(this.nn.get(i).get(0), JMatrix.mulBy(weightGradient, learnRate)));
    
            JMatrix biasGradient = this.gradients.get(i).get(1);
            this.nn.get(i).set(1, JMatrix.sub(this.nn.get(i).get(1), JMatrix.mulBy(biasGradient, learnRate)));
        }
    }

    private ArrayList<JMatrix> activations(JMatrix inputs)
    {
        ArrayList<JMatrix> activations = new ArrayList<JMatrix>();
        activations.add(inputs);

        JMatrix weights = new JMatrix();
        JMatrix biases = new JMatrix();

        JMatrix z = new JMatrix();

        JMatrix activation = new JMatrix();

        int count = 0;

        for(int i = 0; i < this.nn.size(); i++)
        {
            weights = this.nn.get(i).get(0);
            biases = this.nn.get(i).get(1);

            z = JMatrix.add(JMatrix.dot(activations.get(activations.size()-1), weights), biases);
            activation = this.activation_functions.get(count).apply(z);
            activations.add(activation);

            count++;
        }
        this.output = activation;
        return activations;
    }

    public void train(int epochs, double learnRate, boolean print) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalCost = 0;
    
            for (ArrayList<JMatrix> data : dataset) {
                JMatrix inputs = data.get(0);
                JMatrix target = data.get(1);
    
                ArrayList<JMatrix> activations = activations(inputs);
    
                JMatrix output = activations.get(activations.size() - 1);
                totalCost += cost(output, target);
    
                JMatrix delta = JMatrix.mul(JMatrix.sub(output, target), activation_derv_functions.get(activation_derv_functions.size()-1).apply(activations.get(activations.size()-1)));

                this.gradients.get(gradients.size()-1).set(0, JMatrix.outer(activations.get(activations.size()-2), delta));
                this.gradients.get(gradients.size()-1).set(1, delta);

                for(int i = 2; i < this.layers.size(); i++)
                {
                    JMatrix sp =  this.activation_derv_functions.get(this.activation_derv_functions.size()-i).apply(activations.get(activations.size()-i));

                    delta = JMatrix.mul(JMatrix.transpose(JMatrix.dot(this.nn.get(this.nn.size()-i+1).get(0), delta)), sp);
                    this.gradients.get(this.gradients.size()-i).set(0, JMatrix.outer(activations.get(activations.size()-i-1), delta));
                    this.gradients.get(this.gradients.size()-i).set(1, delta);
                }
    
                applyGradient(learnRate);
            }
    
            if (print == true)
            {
                System.out.println("Epoch " + (epoch + 1) + "/" + epochs + " - Cost: " + totalCost);
            }
        }
    }

    public JMatrix forward(JMatrix activations)
    {

        JMatrix weights = new JMatrix();
        JMatrix biases = new JMatrix();

        int count = 0;
        for (int i = 0; i < this.nn.size(); i++)
        {
            weights = this.nn.get(i).get(0);
            biases = this.nn.get(i).get(1);

            activations = this.activation_functions.get(count).apply(JMatrix.add(JMatrix.dot(activations, weights), biases));
            
            count++;
        }

        this.output = activations;
        return this.output;
    }

    public void init_weights()
    {
        ArrayList<JMatrix> layer = new ArrayList<JMatrix>();
        this.nn = new ArrayList<ArrayList<JMatrix>>();
        this.gradients = new ArrayList<ArrayList<JMatrix>>();

        for(int i = 0; i < this.layers.size()-1; i++)
        {
            layer = new ArrayList<JMatrix>();
            layer.add(JMatrix.random(this.layers.get(i), this.layers.get(i+1)));
            layer.add(JMatrix.random(1, this.layers.get(i+1)));
            this.nn.add(layer);
        }

        for(int i = 0; i < this.layers.size()-1; i++)
        {
            layer = new ArrayList<JMatrix>();
            layer.add(JMatrix.zeros(this.layers.get(i), this.layers.get(i+1)));
            layer.add(JMatrix.zeros(1, this.layers.get(i+1)));
            this.gradients.add(layer);
        }
    }

    public void save(String path)
    {
        try (FileOutputStream fileOut = new FileOutputStream(path);
             ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
            out.writeObject(this.nn);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void load(String path)
    {
        try (FileInputStream fileIn = new FileInputStream(path);
             ObjectInputStream in = new ObjectInputStream(fileIn)) {
            this.nn = (ArrayList) in.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

}