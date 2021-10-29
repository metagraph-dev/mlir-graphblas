// RUN: graphblas-opt %s | graphblas-exec entry | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
  func private @create_choose_uniform_context(i64) -> !llvm.ptr<i8>
  func private @destroy_choose_uniform_context(!llvm.ptr<i8>)
  func private @random_double(!llvm.ptr<i8>) -> f64

  func @entry() {
    %c7 = arith.constant 7 : index
    %cf10 = arith.constant 0.10 : f64
    %cf90 = arith.constant 0.90 : f64
    %seed = arith.constant 123456789 : i64
    %ctx = call @create_choose_uniform_context(%seed) : (i64) -> !llvm.ptr<i8>

    ///////////////
    // Test random select by assuming
    // Prob=10% will always select less than half the values
    // Prob=90% will always select more than half the values
    ///////////////

    %v = arith.constant dense<
      [ 1.0,  2.0,  0.0, -4.0, 5.0, 0.0, 7.0, 8.0, 9.0, 2.1, 2.2, 2.3, 0.0, 2.5 ]
    > : tensor<14xf64>
    %v_cv = sparse_tensor.convert %v : tensor<14xf64> to tensor<?xf64, #CV64>

    // P10
    //
    // CHECK: (10) size<=7? 1
    //
    %10 = graphblas.select %v_cv, %cf10, %ctx { selector = "probability" } : tensor<?xf64, #CV64>, f64, !llvm.ptr<i8> to tensor<?xf64, #CV64>
    %11 = graphblas.num_vals %10 : tensor<?xf64, #CV64>
    %12 = arith.cmpi "ule", %11, %c7 : index
    graphblas.print %12 { strings=["(10) size<=7? "] } : i1

    // P90
    //
    // CHECK: (20) size>=7? 1
    //
    %20 = graphblas.select %v_cv, %cf90, %ctx { selector = "probability" } : tensor<?xf64, #CV64>, f64, !llvm.ptr<i8> to tensor<?xf64, #CV64>
    %21 = graphblas.num_vals %20 : tensor<?xf64, #CV64>
    %22 = arith.cmpi "uge", %21, %c7 : index
    graphblas.print %22 { strings=["(20) size>=7? "] } : i1

    call @destroy_choose_uniform_context(%ctx) : (!llvm.ptr<i8>) -> ()

    return
  }
}