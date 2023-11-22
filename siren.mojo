from random import seed, rand, random_ui64, randn_float64
from tensor import Tensor, TensorShape
from python import Python
import math
from utils.index import Index

alias f64 = DType.float64
alias T = Tensor[f64]


struct SineActivation:
    var w0: Int
    var input_record: T

    fn __init__(inout self, w0: Int):
        self.w0 = w0
        self.input_record = T()  # have to initialize?

    fn forward(inout self, x: T) -> T:
        let batch_size = x.shape()[0]
        let dim = x.shape()[1]
        var out = T(batch_size, dim)
        self.input_record = T(batch_size, dim)
        for b in range(batch_size):
            for j in range(dim):
                self.input_record[Index(b, j)] = x[b, j]
                out[Index(b, j)] = math.sin(x[b, j])
        return out

    fn backward(self, grad: T) -> T:
        let batch_size = grad.shape()[0]
        let dim = grad.shape()[1]
        var out = T(batch_size, dim)
        for b in range(batch_size):
            for j in range(dim):
                out[Index(b, j)] = (
                    grad[b, j] * self.w0 * math.cos(self.w0 * self.input_record[b, j])
                )
        return out

    fn __call__(inout self, x: T) -> T:
        return self.forward(x)


struct SigmoidActivation:
    var output_record: T

    fn __init__(inout self, dim: Int):
        self.output_record = T()

    fn forward(inout self, x: T) -> T:
        let batch_size = x.shape()[0]
        let dim = x.shape()[1]
        var out = T(batch_size, dim)
        self.output_record = T(batch_size, dim)
        for b in range(batch_size):
            for j in range(dim):
                let sigmoid = 1 / (1 + math.exp(x[b, j]))
                self.output_record[Index(b, j)] = sigmoid
                out[Index(b, j)] = sigmoid
        return out

    fn backward(self, grad: T) -> T:
        let batch_size = grad.shape()[0]
        let dim = grad.shape()[1]
        var out = T(batch_size, dim)
        for b in range(batch_size):
            for j in range(dim):
                out[Index(b, j)] = (
                    grad[b, j]
                    * self.output_record[b, j]
                    * (1 - self.output_record[b, j])
                )
        return out

    fn __call__(inout self, x: T) -> T:
        return self.forward(x)


struct LinearLayer:
    var dim_in: Int
    var dim_out: Int
    var weight: T
    var weight_grad: T
    var bias: T
    var bias_grad: T
    var input_record: T

    fn __init__(inout self, dim_in: Int, dim_out: Int):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = T(dim_out, dim_in)
        self.bias = T(dim_out)
        self.weight_grad = T(dim_out, dim_in)
        self.bias_grad = T(dim_out)
        self.input_record = T()

    fn forward(inout self, x: T) -> T:
        let batch_size = x.shape()[0]
        var out = T(batch_size, self.dim_out)
        self.input_record = T(batch_size, self.dim_in)
        for b in range(batch_size):
            for j in range(self.dim_in):
                self.input_record[Index(b, j)] = x[b, j]
                for i in range(self.dim_out):
                    out[Index(b, i)] += self.weight[i, j] * x[b, j]
        return out

    fn backward(inout self, grad: T) -> T:
        let batch_size = grad.shape()[0]
        var out = T(batch_size, self.dim_out)
        for b in range(batch_size):
            for i in range(self.dim_out):
                # dl/db = b * grad
                self.bias_grad[i] += grad[b, i]
                for j in range(self.dim_in):
                    # dl/dx = grad * w^T
                    out[Index(b, i)] = grad[b, i] * self.weight[i, j]
                    # dl/dw = xT * grad
                    self.weight_grad[Index(i, j)] += (
                        self.input_record[b, j] * grad[b, i]
                    )
        return out

    fn zero_grad(inout self):
        for i in range(self.dim_out):
            self.bias_grad[i] = 0
            for j in range(self.dim_in):
                self.weight_grad[i] = 0

    fn initialize_weight(inout self):
        for i in range(self.dim_out):
            self.bias[i] = 0
            for j in range(self.dim_in):
                let w: Float64 = randn_float64(0, 1)
                self.weight[Index(i, j)] = w

    fn step(inout self, lr: Float64 = 0.0001):
        for i in range(self.dim_out):
            self.bias[i] += self.bias_grad[i]
            for j in range(self.dim_in):
                self.weight[Index(i, j)] += self.weight_grad[i, j] * lr

    fn __call__(inout self, x: T) -> T:
        return self.forward(x)


struct SirenLayer:
    var activation: SineActivation
    var linear: LinearLayer
    var w0: Int

    fn __init__(inout self, in_dim: Int, out_dim: Int, w0: Int = 1):
        self.w0 = w0
        self.activation = SineActivation(w0)
        self.linear = LinearLayer(in_dim, out_dim)
        self.linear.initialize_weight()

    fn forward(inout self, x: T) -> T:
        let lin_out = self.linear(x)
        if self.w0 > 0:
            return self.activation(lin_out)
        else:
            return lin_out

    fn backward(inout self, grad: T) -> T:
        if self.w0 > 0:
            let act_grad = self.activation.backward(grad)
            return self.linear.backward(grad)
        else:
            return self.linear.backward(grad)

    fn zero_grad(inout self):
        self.linear.zero_grad()

    fn step(inout self, lr: Float64 = 0.0001):
        self.linear.step(lr)

    fn __call__(inout self, x: T) -> T:
        return self.forward(x)


struct Siren:
    var layer1: SirenLayer
    var layer2: SirenLayer
    var layer3: SirenLayer
    var layer4: SirenLayer
    var layer5: SirenLayer
    var sigmoid: SigmoidActivation

    fn __init__(
        inout self,
        batch_size: Int,
        in_dim: Int = 2,
        hidden_dim: Int = 48,
        out_dim: Int = 3,
        hidden_layers: Int = 1,
    ):
        self.layer1 = SirenLayer(in_dim, hidden_dim, w0=30)
        self.layer2 = SirenLayer(hidden_dim, hidden_dim, w0=1)
        self.layer3 = SirenLayer(hidden_dim, hidden_dim, w0=1)
        self.layer4 = SirenLayer(hidden_dim, hidden_dim, w0=1)
        self.layer5 = SirenLayer(hidden_dim, out_dim, w0=0)
        self.sigmoid = SigmoidActivation(out_dim)

    fn forward(inout self, x: T) -> T:
        let out1 = self.layer1(x)
        let out2 = self.layer2(out1)
        let out3 = self.layer3(out2)
        let out4 = self.layer4(out3)
        let out5 = self.layer5(out4)
        let outs = self.sigmoid(out5)
        return outs

    fn backward(inout self, grad: T) -> T:
        let ds = self.sigmoid.backward(grad)
        let d5 = self.layer5.backward(ds)
        let d4 = self.layer4.backward(d5)
        let d3 = self.layer3.backward(d4)
        let d2 = self.layer2.backward(d3)
        let d1 = self.layer1.backward(d2)
        return d1

    fn zero_grad(inout self):
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.layer3.zero_grad()
        self.layer4.zero_grad()
        self.layer5.zero_grad()

    fn step(inout self, lr: Float64 = 0.0001):
        self.layer1.step(lr)
        self.layer2.step(lr)
        self.layer3.step(lr)
        self.layer4.step(lr)
        self.layer5.step(lr)

    fn __call__(inout self, x: T) -> T:
        return self.forward(x)


def load_logo_image() -> T:
    let imageio = Python.import_module("imageio")
    let image = imageio.imread("./mojo-logo.png") / 255

    let height: Int = image.shape[0].to_float64().to_int()
    let width: Int = image.shape[1].to_float64().to_int()
    let channels: Int = image.shape[2].to_float64().to_int()
    var tensor = T(height, width, channels)

    for i in range(height):
        for j in range(width):
            for k in range(channels):
                tensor[Index(i, j, k)] = image[i][j][k].to_float64()
    return tensor


def visualize_logo_image(tensor: T):
    plt = Python.import_module("matplotlib.pyplot")
    np = Python.import_module("numpy")
    let height = tensor.shape()[0]
    let width = tensor.shape()[1]
    let channels = tensor.shape()[2]
    let numpy_array = np.zeros((height, width, channels), np.float64)
    var num: Float64 = 0
    var denom = 0
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                num += tensor[i, j, k]
                numpy_array.itemset((i, j, k), tensor[i, j, k])
                denom += 1
    print("array stats")
    print(numpy_array.max())
    print(numpy_array.mean())
    plt.imshow(numpy_array)
    plt.savefig("visualization.jpg")
    plt.close()


alias BATCH_SIZE = 3000


def main():
    var siren = Siren(2, 96, 3)
    let image = load_logo_image()

    # train
    for epoch_idx in range(1000):
        siren.zero_grad()
        x = T(BATCH_SIZE, 2)
        y = T(BATCH_SIZE, 3)
        for i in range(BATCH_SIZE):
            let row = random_ui64(0, 224).to_int()
            let col = random_ui64(0, 224).to_int()
            x[Index(i, 0)] = row / 224
            x[Index(i, 1)] = col / 224
            for j in range(3):
                y[Index(i, j)] = image[row, col, j]

        let yhat = siren.forward(x)
        var grad = T(BATCH_SIZE, 3)
        var sum_diff: Float64 = 0
        for i in range(BATCH_SIZE):
            for j in range(3):
                grad[Index(i, j)] = y[i, j] - yhat[i, j] / BATCH_SIZE
                sum_diff += math.min((y[i, j] - yhat[i, j]) ** 2, 10)
        print("avg diff")
        print(sum_diff / BATCH_SIZE / 3)
        siren.backward(grad)
        siren.step()

    # test
    var test_input = T(224 * 224, 2)
    for i in range(224):
        for j in range(224):
            test_input[Index(j, 0)] = i / 224
            test_input[Index(j, 1)] = j / 224
    test_out = siren(test_input)
    var test_image = T(224, 224, 3)
    var maximum: Float64 = 0
    for i in range(224):
        for j in range(224):
            for k in range(3):
                out = test_out[j, k]
                test_image[Index(i, j, k)] = out
                maximum = math.max(out, maximum)

    print("max test val")
    print(maximum)
    visualize_logo_image(test_image)
