
import os

import torch
import gpytorch

from utils import Hellinger_dist, Chernoff_dist

smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 20

# Wrap training, prediction and plotting from the ExactGP-Tutorial into a function,
# so that we do not have to repeat the code later on
def train(model, likelihood, training_iter=training_iter):
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(model.train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, model.train_y)
        loss.backward()
        optimizer.step()

        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model.covar_module.lengthscale.item(),
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()

        if loss < 0: break


def predict(model, likelihood, test_x = torch.linspace(0, 1, 51)):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Test points are regularly spaced along [0,1]
        return likelihood(model(test_x))



'''
Customized kernel of using Squared exponential with Hellinger distance 
'''
class SEHellingerKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    has_lengthscale = True

    # this is the kernel function
    def forward(self, x1, x2, **params):

        # calculate the distance between inputs
        diff = torch.zeros((len(x1), len((x2))))
        for i,x in enumerate(x1):
            for j,y in enumerate(x2):
                diff[i,j] = Hellinger_dist(x, y)

        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        return torch.exp(-0.5 * diff / (self.lengthscale **2))
        # return torch.sin(diff).div(diff)

'''
Customized kernel of using Squared exponential with Chernoff distance 
'''
class SEChernoffKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    has_lengthscale = True

    # this is the kernel function
    def forward(self, x1, x2, **params):

        # calculate the distance between inputs
        diff = torch.zeros((len(x1), len((x2))))
        for i,x in enumerate(x1):
            for j,y in enumerate(x2):
                diff[i,j] = Chernoff_dist(x, y)

        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        return torch.exp(-0.5 * diff / (self.lengthscale **2))


# Use the simplest form of GP model, exact inference
class CustomGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module
        self.train_x = train_x
        self.train_y = train_y

        self.is_sparse = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



'''
Customized kernels of using Squared exponential with L1 distance of the sufficient statistic 
'''
class L1RBFKernel(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    has_lengthscale = True

    # this is the kernel function
    def forward(self, x1, x2, **params):

        # calculate the distance between inputs
        diff = torch.zeros((len(x1), len((x2))))
        for i,x in enumerate(x1):
            for j,y in enumerate(x2):
                diff[i,j] = torch.sum(torch.abs(x-y))

        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        return torch.exp(-0.5 * diff / (self.lengthscale **2))


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, l1=True):

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if l1:
            self.covar_module = gpytorch.kernels.ScaleKernel(L1RBFKernel())
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.train_x = train_x
        self.train_y = train_y

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

