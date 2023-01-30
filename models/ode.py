"""
ODE-VAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _single
from torch.nn.parameter import Parameter

class PeriodicPad1d(nn.Module):
    def __init__(self, pad, dim=-1):
        super(PeriodicPad1d, self).__init__()
        self.pad = pad
        self.dim = dim

    def forward(self, x):
        if self.pad > 0:
            front_padding = x.narrow(self.dim, x.shape[self.dim]-self.pad, self.pad)
            back_padding = x.narrow(self.dim, 0, self.pad)
            x = torch.cat((front_padding, x, back_padding), dim=self.dim)

        return x


# layer used for propagator (decoder)
class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 dilation=1, groups=1, boundary_cond='ode'):

        super(DecoderLayer, self).__init__()

        self.stride = _single(stride) # not implemented
        self.dilation = _single(dilation) # not implemented

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.boundary_cond = boundary_cond

        self.pad = None


    # center part
    def forward(self, input, weight, bias):
        # we don't need a conv propagator any more
        
        # self.out_channels = hidden_channels for RNN
        # self.kernel_size[0] = "nonlin_kernel_size" in .json
        
        # input shape
        # batch_size, layer_input_channel
        
        # weight shape
        # batch_size, layer_input_channel * layer_output_channel

        y_tomul = input.view(-1, 1, self.in_channels)
        
        # The view function is meant to reshape the tensor,
        # and avoids explicit data copy
        # when you call transpose(), PyTorch doesn't generate new tensor with new layout, it just modifies meta information in Tensor object so offset and stride are for new shape. The transposed tensor and original tensor are indeed sharing the memory!
        # is_contiguous() tells you wether the tensor is the original one
        
        # -1 in pytorch view does behave like -1 in numpy.reshape(),
        # i.e. the actual value for this dimension will be inferred
        # so that the number of elements in the view matches the original number of elements.
        
        # batch_size, input_channels, output_channels
        weight_tomul = weight.view(-1, self.in_channels, self.out_channels)
        
        y = torch.matmul(y_tomul, weight_tomul)
        
        
        if bias is not None:
            y = y + bias.view(-1, 1, self.out_channels)

        return y.transpose(-1, -2)

# Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_channels, data_channels, stride=1,
                 dilation=1, groups=1, prop_layers=1, prop_noise=0., boundary_cond='ode'):

        self.data_channels = data_channels
        self.prop_layers = prop_layers
        self.prop_noise = prop_noise
        self.boundary_cond = boundary_cond


        super(Decoder, self).__init__()


        self.conv_in = DecoderLayer(data_channels, hidden_channels, stride,
                                    dilation, groups, boundary_cond)

        self.conv_out = DecoderLayer(hidden_channels, data_channels, stride,
                                    dilation, groups, boundary_cond)

        if prop_layers > 0:
            self.conv_prop = nn.ModuleList([DecoderLayer(hidden_channels, hidden_channels, stride,
                                            dilation, groups, boundary_cond)
                                            for i in range(prop_layers)])

        self.cutoff = Parameter(torch.Tensor([1]))

    # _xxx means private
    def _f(self, y, in_weight, in_bias, 
                    out_weight, out_bias, prop_weight, prop_bias):
        # conv_in matmul
        # relu
        # for range(prop_layers)
        #   prop matmul
        #   relu
        # conv_out
        # tanh

        y = self.conv_in(y, in_weight, in_bias)
        y = F.relu(y, inplace=True)
        for j in range(self.prop_layers):
            y = self.conv_prop[j](y, prop_weight[:,j], prop_bias[:,j])
            y = F.relu(y, inplace=True)
        y = self.conv_out(y, out_weight, out_bias)

        return y

    def forward(self, y0,in_weight, in_bias, out_weight, out_bias, prop_weight, prop_bias, predict_length):
        if  self.boundary_cond == 'ode' :
            # y0 shape(batch_size,data_channels)
            assert len(y0.shape) == 2
            assert y0.shape[1] == self.data_channels
            y = y0
        else:
            raise ValueError("Invalid boundary condition.")

        f = lambda y: self._f(y, in_weight, in_bias, 
                                        out_weight, out_bias, prop_weight, prop_bias)

        y_list = []
        for i in range(predict_length):

            ### Euler integrator
            dt = 1e-6 # NOT REAL TIME STEP, JUST HYPERPARAMETER
            noise = self.prop_noise * torch.randn_like(y) if (self.training and self.prop_noise > 0) else 0
            f_y = f(y)
            y = y + self.cutoff * torch.tanh((dt * f_y.view(-1,self.data_channels)) / self.cutoff) + noise
            # skip connection: the y on the right side

            y_list.append(y)

        output_predict = torch.stack(y_list, dim=-2)
        return output_predict.transpose(-1, -2)

# The whole encoder-decoder
# But the details of decoder are in class Decoder
class VAE(nn.Module): # inherit nn.Module. The stuff in () for classes mean inheritance. The most basic one is (Object).
    def __init__(self, param_size=1, data_channels=1, hidden_channels=16, 
                        prop_layers=1, prop_noise=0., 
                        boundary_cond='periodic', param_dropout_prob=0.1, debug=False):

        
        super(VAE, self).__init__()

        # private variable example: ('__')
        # self.__para1
        # __xxx__ type is more special, and is not private.
        
        self.param_size = param_size
        self.data_channels = data_channels
        self.hidden_channels = hidden_channels
        self.prop_layers = prop_layers
        self.boundary_cond = boundary_cond
        self.param_dropout_prob = param_dropout_prob
        self.debug = debug

        if param_size > 0:

            pad_input = [1, 2, 4, 8]
            pad_func = PeriodicPad1d
            
            dilation_set = [4, 8, 16, 32]
            kernel_size_set = [8, 8, 4, 4]

            self.encoder = nn.Sequential(   pad_func(pad_input[0]),
                                            nn.Conv1d(data_channels, 4, kernel_size=kernel_size_set[0], dilation=dilation_set[0]),
                                            nn.ReLU(inplace=True),

                                            pad_func(pad_input[1]),
                                            nn.Conv1d(4, 16, kernel_size=kernel_size_set[1], dilation=dilation_set[1]),
                                            nn.ReLU(inplace=True),

                                            pad_func(pad_input[2]),
                                            nn.Conv1d(16, 64, kernel_size=kernel_size_set[2], dilation=dilation_set[2]),
                                            nn.ReLU(inplace=True),

                                            pad_func(pad_input[3]),
                                            nn.Conv1d(64, 64, kernel_size=kernel_size_set[3], dilation=dilation_set[3]),
                                            nn.ReLU(inplace=True),
                                            )
            self.encoder_to_param = nn.Sequential(nn.Conv1d(64, param_size, kernel_size=1, stride=1))
            self.encoder_to_logvar = nn.Sequential(nn.Conv1d(64, param_size, kernel_size=1, stride=1))

            ### Parameter to weight/bias for dynamic convolutions (RNN decoder)
            
            self.param_to_in_weight = nn.Sequential( nn.Linear(param_size, 4 * data_channels * hidden_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(4 * data_channels * hidden_channels, 
                                                data_channels * hidden_channels)
                                    )
            self.param_to_in_bias = nn.Sequential( nn.Linear(param_size, 4 * hidden_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(4 * hidden_channels, hidden_channels)
                                    )

            self.param_to_out_weight = nn.Sequential( nn.Linear(param_size, 4 * data_channels * hidden_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(4 * data_channels * hidden_channels, 
                                                data_channels * hidden_channels)
                                    )
            self.param_to_out_bias = nn.Sequential( nn.Linear(param_size, 4 * data_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(4 * data_channels, data_channels)
                                    )

            if prop_layers > 0:
                self.param_to_prop_weight = nn.Sequential( nn.Linear(param_size, 4 * prop_layers * hidden_channels * hidden_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4 * prop_layers * hidden_channels * hidden_channels, 
                                                    prop_layers * hidden_channels * hidden_channels)
                                        )
                self.param_to_prop_bias = nn.Sequential( nn.Linear(param_size, 4 * prop_layers * hidden_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4 * prop_layers * hidden_channels, prop_layers * hidden_channels)
                                        )

        ### Decoder/PDE simulator
        self.decoder = Decoder(hidden_channels, data_channels,
                                        prop_layers=prop_layers, prop_noise=prop_noise, boundary_cond=boundary_cond)
        
    # the first variable of all the methods are always self.
    def forward(self, x, y0, predict_length):
        # x goes into encoder
        # y0 goes into decoder, should be the initial state for predicting

        if self.param_size > 0:
            assert len(x.shape) == 3
            assert x.shape[1] == self.data_channels

            ### 2D Convolutional Encoder
            encoder_out = self.encoder(x)
            
            logvar = self.encoder_to_logvar(encoder_out)
            logvar_size = logvar.shape
            logvar = logvar.view(logvar_size[0], logvar_size[1], -1)
            params = self.encoder_to_param(encoder_out).view(logvar_size[0], logvar_size[1], -1)

            if self.debug:
                raw_params = params

            # Parameter Spatial Averaging Dropout
            if self.training and self.param_dropout_prob > 0:
                mask = torch.bernoulli(torch.full_like(logvar, self.param_dropout_prob))
                mask[mask > 0] = float("inf")
                logvar = logvar + mask

            # Inverse variance weighted average of params
            weights = F.softmax(-logvar, dim=-1)
            params = (params * weights).sum(dim=-1)

            # Compute logvar for inverse variance weighted average with a correlation length correction
            correlation_length = 31 # estimated as receptive field of the convolutional encoder
            logvar = -torch.logsumexp(-logvar, dim=-1) \
                        + torch.log(torch.tensor(
                            max(1, (1 - self.param_dropout_prob)
                                    * min(correlation_length, logvar_size[-2])
                                    * min(correlation_length, logvar_size[-1])),
                            dtype=logvar.dtype, device=logvar.device))

            ### Variational autoencoder reparameterization trick
            if self.training:
                stdv = (0.5 * logvar).exp()

                # Sample from unit normal
                z = params + stdv * torch.randn_like(stdv)
            else:
                z = params

            ### Parameter to weight/bias for dynamic convolutions (RNN decoder)
            # weights and biases of RNN decoder are functions of params z from CNN encoder

            in_weight = self.param_to_in_weight(z)
            in_bias = self.param_to_in_bias(z)

            out_weight = self.param_to_out_weight(z)
            out_bias = self.param_to_out_bias(z)

            if self.prop_layers > 0:
                prop_weight = self.param_to_prop_weight(z).view(-1, self.prop_layers,
                                    self.hidden_channels * self.hidden_channels)
                prop_bias = self.param_to_prop_bias(z).view(-1, self.prop_layers, self.hidden_channels)
            else:
                prop_weight = None
                prop_bias = None

        else: # if no parameter used
            in_weight = None
            in_bias = None
            out_weight = None
            out_bias = None
            prop_weight = None
            prop_bias = None
            params = None
            logvar = None

        ### Decoder/PDE simulator
        y = self.decoder(y0, in_weight, in_bias, out_weight, out_bias, 
                                prop_weight, prop_bias, predict_length)

        if self.debug:
            return y, params, logvar, [in_weight, in_bias, out_weight, out_bias, prop_weight, prop_bias], \
                    weights.view(logvar_size), raw_params.view(logvar_size)

        return y, params, logvar
