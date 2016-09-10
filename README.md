# TheanoUnpickler
A quick tutorial on how to unpickle GPU saved variables, on a CPU machine.

The key to unpickling these variables is setting theano.config.experimental.unpickle_gpu_on_cpu = True
  + This allows the user to unpickle these variables without theano throwing a CUDA exception.
  + Because theano variables are often stored as CudaNdArrays you often need a GPU to unpickle the parameters of
  previously trained models
  
I have included a small python script which i used to convert CudaNdArrays to NumpyArrays when unpickling

If you have any questions contact me @:
  +rcbridendev@gmail.com
