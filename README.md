# Polymorphic Dense Layer
Tensorfow + Keras layer that switches between different weights depending on the input data. 

PolymorphicDense first using input to generate a key, then compares this key with it's own keys table. Result is a generated list of similarity coefficients which is then multiplied with layer's table of kernels. Then mean is taken over this weighted kernels list, essentially producing weighted average of layer's kernels.

ControlledPolymorphicDense does not calculate a key but accepts it as one of the inputs. Therefore, key-input controls the processing of main input. Hens, ControlledPolymorphicDense can be considered as a 'transistor for tensors'.


Tested with 2-rank input (ordinary feedforward network) and 3-rank input (network that contains RNN/LSTM/GRU layers which return sequences).