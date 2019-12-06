# Tensorflow C API Examples

## Get Libraries ##

You need to download libraries (tested with version 1.15.0) from:

   https://www.tensorflow.org/install/lang_c

Put them in `lib` directory next to `include`.  Update `include` for different versions.

See their LICENSE for includes.

## Example applications ##

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

