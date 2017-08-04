import model
import updater
import weight


def quad(x):
    #implementation of quadratic function.
    #global minima is zero.
    return x * x


def main():
    gaussian = model.MultiVariableGaussian(1)
    w = weight.QuantileBasedWeight()
    lr = {'mean': 1., 'var': 0.5 / (10 ** 2)}
    u = updater.NaturalGradientUpdater(gaussian, w, lr)
    
    for i in range(10000):
        sampling = gaussian.sampling(3)
        evals = quad(sampling)
        u.update(evals, sampling)
    
    print(gaussian.mean, gaussian.var)

if __name__ == '__main__':
    main()