
python setup.py build_ext
python setup.py install --single-version-externally-managed --record=record.txt

# Build graphblas-opt

python3 ./mlir_graphblas/src/build.py -build-clean
GRAPHBLAS_OPT_BUID_DIR=./mlir_graphblas/src/build
cp $GRAPHBLAS_OPT_BUID_DIR/bin/graphblas-opt $PREFIX/bin
