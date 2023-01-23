# torch_utils

### Customized PyTorch utility package of my own, written for convenient training and testing of various data.

1. __init__.py contains high-API wrapped routines.

2. auxs.py contains some auxiliary functions.

3. datatool.py contains DataStruct class(torch.DataLoader) and its subclasses, and a simple data generator.

4. example.py describes the usage of the general pipeline of this package.

5. monitor.py contains monitoring routines that show loss, accuracy, parameter gradients, etc., during training or tesing phases.

6. nnframe.py contains torch.nn class-based general frameworks for various feed-forward networks.

7. pipeline.py contains pipeline classes that unify the preprocessing, monitoring routines in this package.

7. task.py contains computing sub-routines for the monitoring and pipeline modules.
