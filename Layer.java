import java.util.function.Function;

public class Layer {
    public int size;
    public Function<JMatrix, JMatrix> activation;
    public Function<JMatrix, JMatrix> derivative;

    public Layer(int size, Function<JMatrix, JMatrix> activation, Function<JMatrix, JMatrix> derivative) {
        this.size = size;

        this.activation = activation != null ? activation : this::none;

        this.derivative = derivative != null ? derivative : this::noneDerivative;
    }

    public JMatrix none(JMatrix x) {
        return x;
    }

    public JMatrix noneDerivative(JMatrix x) {
        return JMatrix.zeros(x.getHeight(), x.getWidth());
    }

    public Function<JMatrix, JMatrix> getActivation() {
        return activation;
    }

    public Function<JMatrix, JMatrix> getDerivative() {
        return derivative;
    }
}
