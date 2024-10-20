
_NOTE: these codes are research, proof-of-principle codes only, and are not intended to demonstrate the state of the art in terms of training times for matrix product states for machine learning_

_If you are seeking fast approaches for optimizing MPS, we recommend trying newer libraries which use stochastic gradient optimization methods, such as TorchMPS_: https://github.com/jemisjoky/TorchMPS

# Tensor network machine learning

Codes based on the paper "Supervised Learning with Quantum-Inspired Tensor Networks"
by Miles Stoudenmire and David Schwab. http://arxiv.org/abs/1605.05775

Also see "Tensor Train Polynomial Models via Riemannian Optimization" by Novikov, Trofimov, and Oseledets for a similar approach: http://arxiv.org/abs/1605.03795


# Code Overview

`fixedL` -- optimize a matrix product state (MPS) with a label index on the central tensor, similar to what is described in the paper arxiv:1605.05775, but where the label index stays fixed on the central tensor and does not move around during optimization. This MPS parameterizes a model whose output is a vector of 10 numbers (for the case of MNIST). The output entry with the largest value is the predicted label.

`fulltest` -- given an MPS ("wavefunction") generated by the fixedL program, report classification error for the MNIST testing set

`single` -- optimize an MPS for a single label type, with no label index on the MPS. This MPS parameterizes a model whose output is positive for inputs of the correct type, and zero for all other inputs.

`separate_fulltest` -- report classification error for the MNIST testing set for a set of MPS created by the "single" application. IMPORTANT: this program assumes that the MPS W00, W01, W02, etc. made by running "single" reside in folders (which you have to create) named L00/, L01/, L02/ etc. So it looks for the files L00/W00, L01/W01, etc. under the folder where you run it.

# Compiling and running the programs

Dependencies:
- ITensor tensor network library: http://itensor.org  (on github at http://github.com/ITensor/ITensor)
- <del> libpng (available through most package managers): http://www.libpng.org </del>
- <del> png++: http://www.nongnu.org/pngpp/ </del>

Steps to install and run:

1. Install the above dependencies.
2. Do `cp Makefile.sample Makefile` to create a `Makefile` from the sample provided.
3. Edit the following variables at the top of your Makefile:
  - `ITENSOR_DIR`: this should be the folder where you `git clone`'d and installed ITensor (where the options.mk file is located)
  - <del> `LIBPNG_DIR`: folder where the file libpng16.so (or libpng16.dylib on mac) is located (or change the name of the library if you install a different version of libpng) </del>
  - <del> `PNGPP_DIR`: folder where the png++ header (.hpp) files are located </del>
4. Run the command `make`, which should successfully build the `fixedL` application.
5. Copy one of the sample input files from the folder `sample_inputs/` to another folder of your choosing. Run each app by doing `./appname input_file_name`.
6. Edit the input file. At a minimum, change datadir to point to the location of the mllib/MNIST folder (inside of this repo) on your computer. Play around with the other settings such as Ntrain (max number of training images per label) to check basic things about the code before trying a heavy-duty calculation.

All of the codes require you to install the ITensor tensor network library. You can obtain it from http://github.com/ITensor/ITensor . The only software dependencies for ITensor are a compiler that supports C++11 (language and standard library) and a BLAS/LAPACK distribution such as the "lapack" package on linux, the Accelerate/Veclib framework on MacOS, or the Intel MKL library.

See http://itensor.org/ for help installing ITensor and for more documentation on it.

Once ITensor is installed, modify the first line of the provided Makefile to point to the ITensor installation folder. (Note: ITensor does not put files anywhere else on your computer; it just creates libraries inside its own folder.)

To use the Makefile, either just run `make` to build the default program (which is fixedL) or do `make app=appname` to compile the program `appname` (either fixedL, single, fulltest, or separate_fulltest).

# Input files

Sample input files for `fixedL` and `single` are provided in the sample_inputs/ folder.

See below for a list of the possible input parameters to these programs and what they do.

# FixedL program input parameters and code features

`fixedL` optimizes a matrix product state (MPS) with a label index on the central tensor, similar to what is described in the paper arxiv:1605.05775. This MPS parameterizes a model whose output is a vector of 10 numbers (for the case of MNIST). The output entry with the largest value is the predicted label.

One difference from the algorithm described in the paper is that the label index always remains on the same MPS tensor and is not moved around (although it can be moved, keeping it in a fixed position turns helps with the optimization).

Warning: fixedL can use a lot of RAM. If this happens, adjust the Nbatch parameter described below to make the program read smaller amounts of data into ram at each step.

Input parameters:
- nthread (integer) [default: 1]: number of threads to use to parallelize gradient calculations. Not recommended to set this larger than number of cores on your processor.
- Npass (integer) [default: 4]: maximum number of conjugate gradient passes to do at each bond.
- Nsweep (integer) [default: 50]: total number of sweeps (left-to-right passes over the MPS) to do.
- lambda (real) [default: 0.0]: size of the L2 (ridge) regularization penalty to include in the cost function
- maxm (integer) [default: 5000]: maximum bond dimension to allow when adaptively optimizing the MPS tensors
- minm (integer) [default: max(10,maxm/2)]: minimum bond dimension to allow when adaptively optimizing the MPS tensors (sometimes it is not possible to reach the minm value if not enough non-zero singular values are available after the SVD step)
- cutoff (real) [default: 1E-10]: truncation error goal when optimizing the MPS. Smaller value means higher accuracy. The advantage of using a cutoff is that the bond dimension will automatically shrink when it does not need to be big, but can still grow where needed.
- Ntrain (integer) [default: 60000]: number of training images per label type to use when training. Useful for speeding up the code for testing purposes or to study generalization / overfitting. If Ntrain is set to a larger value than the number of training images available then the full set of training images will be used (so it is safe to do this).
- feature (string) [default: normal]: local feature map type to use. "normal" means the [cos(pi/2*x), sin(pi/2*x)] local feature map. "series" uses the feature map [1,x/4] (motivated by the Novikov et al. paper).
- ninitial (integer) [default: 100]: number of training states per label type to use to make the initial MPS by summing training MPS together
- replace (string: "yes" or "no") [default: no]: experimental feature which if set to "yes" will replace the new bond tensor with the old one if the cost function goes up when using the new bond tensor. This can happen if SVD'ing the new bond tensor causes too big of an approximation and makes the cost function rise.
- Nbatch (integer) [default: 10]: number of "batches" into which to divide the "environment" tensors (i.e. the tensors representing each image projected into the "wings" of the weight MPS). These environment tensors can take a huge amount of RAM and so fixedL stores most of them on the hard disk (in the proj\_images folder) and only reads them into memory in batches. By increasing the number of batches you can make the code read fewer environments into memory at a time.

There are other input parameters of a more experimental nature, but the ones above are the most important.

Other code features:
- If the code finds the file "W" (the weight MPS written to disk) and the file "sites" it will read in the previous saved MPS and use it as the initial value. This is extremely useful for restarting the code with different optimization parameters. For example, you could do two sweeps with maxm=10, stop the program, then do more sweeps with a larger maxm.
- The code writes out a file called "sites" shortly after it begins. This holds what ITensor calls a "SiteSet" which is a set of common reference indices to use to allow different MPS tensors created to always share the same set of site indices.
- If the code finds the file "WRITE_WF" (this file can be empty: create it with the command `touch WRITE_WF`) then after optimizing the current bond, the code will write the weight tensor MPS to the file "W" (overwriting it if already present). Once this happens, the code will delete the file "WRITE_WF"

Tips for running fixedL:
- Getting a good initial weight MPS is important before spending a lot of compute time 
  optimizing over the full training set. A simple way to get a decent initial MPS is 
  to do some sweeps with a small maxm setting (maxm = 10, say) with Ntrain set very low,
  like to 100. Then you can do some sweeps with Ntrain=1000 and finally Ntrain=10000 which
  uses the full training set (if Ntrain is larger than the number of images per label, 
  the code just includes every training image).
- Do some sweeps at a smaller maxm or larger cutoff, which keeps the typical MPS bond
  dimension low, before doing the last few sweeps with a larger maxm and smaller cutoff.
- Another really excellent trick for initialization is described in the appendix of the
  Novikov et al. paper (arxiv:1605.03795). Train a linear classifer then define an MPS
  which gives a model having the same output as the linear classifier model. This trick
  can be extended to the single-MPS multi-task case that fixedL uses; this is left
  as an exercise to the reader but I may include a code for this here eventually.


# Single program input parameters and code features

`single` optimizes an MPS for a single label type, with no label index on the MPS. This MPS parameterizes a model whose output is (ideally) positive for inputs of the correct type, and zero for all other inputs.

The input parameters accepted by `single` are mostly the same as for `fixedL` above.

One important extra parameter needed by `single is the "label" parameter, which is an integer 0,1,...,9 
telling the program which single label to "target" when optimizing the MPS.

When saving the currently optimized weight tensor MPS to disk, the `single` app appends the label number which that MPS is targeting. So if the label parameter is set to 3, the program will output the file "W03" (either when the program ends or the WRITE_WF file is found).

# Tips for Using the Codes

- Using the "normal" feature map (i.e. [cos(pi/2*x),sin(pi/2*x)]) on larger image sizes can lead to a vanishing gradient problem. It's easy to understand why: this feature map is normalized so the overlap of two local feature vectors is strictly less than or equal to 1. To the extent that one can think of the weight MPS "W" as a sum (or superposition) of various input training states (the representer theorem, which doesn't strictly apply to tensor networks by the way), then the projection of training vectors into the environment of the weight MPS can lead to very small numbers. An interesting research direction would be to remedy this scenario by coming up with better initial states for W.

- I have been finding that the `single` code can overfit the training data. This isn't too surprising since it produces a completely different MPS for each label type, thus creates a model with many more parameters than the `fixedL` code does. It would be interesting to see if this overfitting can be remedied by explicit L2 regularization, or perhaps just by restricting the bond dimensions of the MPS.


