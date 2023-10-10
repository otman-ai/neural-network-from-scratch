
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, mse, mse_prime, X, Y, epochs=1000, learning_rate=0.1, verbose=True):
    for e in range(epochs):
        error = 0
        for x, y in zip(X, Y):
            output = predict(network, x)
            error += mse(y, output)

            # Backprpagation
            grad = mse_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        error /= len(X)
        if verbose :
            print(f"{e + 1}/epochs, error={error}")
