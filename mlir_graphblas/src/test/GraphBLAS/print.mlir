// RUN: graphblas-opt %s \
// RUN:   --graphblas-structuralize  \
// RUN:   --graphblas-optimize  \
// RUN:   --graphblas-lower  \
// RUN:   --sparsification  \
// RUN:   --sparse-tensor-conversion  \
// RUN:   --linalg-bufferize  \
// RUN:   --convert-scf-to-std  \
// RUN:   --func-bufferize  \
// RUN:   --tensor-constant-bufferize  \
// RUN:   --tensor-bufferize  \
// RUN:   --finalizing-bufferize  \
// RUN:   --convert-linalg-to-loops  \
// RUN:   --convert-scf-to-std  \
// RUN:   --convert-memref-to-llvm  \
// RUN:   --convert-std-to-llvm  \
// RUN:   --reconcile-unrealized-casts  | \
// RUN: mlir-cpu-runner \
// RUN:  -e print_arbitrary_content \
// RUN:  -entry-point-result=void \
// RUN:  -shared-libs=%sparse_utils_so | \
// RUN: FileCheck %s

module {
  func @print_arbitrary_content() -> () {
      %c99 = constant 99 : index
      %0 = constant 1.3 : f32
      %1 = constant 34 : i8
      // CHECK: first line : 1.3 1.3 1.3 1.3
      graphblas.print %0, %0, %0, %0 { strings = ["first line : "] } : f32, f32, f32, f32
      // CHECK: second line : 99 string_a  string_b  string_c 
      graphblas.print %c99 { strings = ["second line : ", " string_a", " string_b", " string_c"] } : index
      // CHECK: third line : 1.3 |"| 34
      graphblas.print %0, %1 { strings = ["third line : ", " |\"| "] } : f32, i8
      return
  }
}
