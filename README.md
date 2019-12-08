# Tensorflow C API Examples

## Get Libraries ##

### Download, or ... ###

You need to download libraries (tested with version 1.15.0) from:

   https://www.tensorflow.org/install/lang_c

Put them in `lib` directory next to `include`.  Update `include` for different versions.

See their LICENSE for includes.

### ... Compile ###

1) Download Bazel 0.26.1 from [https://github.com/bazelbuild/bazel/releases].

   ```shell
   $ wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh
   $ bash bazel-0.26.1-installer-linux-x86_64.sh --user

   ```

2) Get TensorFlow sources version 1.15.


   ```shell
   $ git clone https://github.com/tensorflow/tensorflow.git
   $ cd tensorflow
   $ git checkout r1.15

   ```

3) Compile C API.

   ```shell
   $ ./configure
   $ bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 //tensorflow:libtensorflow.so
   ```

## Example Applications ##

1) `tfc-version`: Display version (from Tensorflow page).
2) `tfc-model-list`: List nodes in a model.
3) `tfc-model-eval`: Load model ([Bartosz's example_2](https://github.com/BartoszPiotrowski/tensorflow-save-in-python-load-in-cpp/tree/master/example_2)) and a evaluate a vector.

## Compile ##

Check [examples/make.sh](examples/make.sh) to see how to compile:

   ```shell
   $ cd examples
   $ ./make.sh
   ```

Run with `.` to export `LD_LIBRARY_PATH` to your current shell envirnonment.

   ```shell
   $ . ./make.sh
   $ ./tfc-model-eval models/example
   ```

## Credits ##

Development of this software prototype was supported by ERC Consolidator grant no. 649043 AI4REASON.

