// RUN: graphblas-opt %s | graphblas-exec entry | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSC64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
  func @entry() {
    %c0 = arith.constant 0 : index

    ///////////////
    // Test Vector
    ///////////////

    %mask_dense = arith.constant sparse<[[0], [1], [2], [3]], [0.999, 1.999, 2.999, 3.999]> : tensor<10xf64>
    %mask = sparse_tensor.convert %mask_dense : tensor<10xf64> to tensor<?xf64, #CV64>
    
    %v1_dense = arith.constant sparse<[[1], [3], [4], [6]], [10., 11., 12., 13.]> : tensor<10xf64>
    %v1 = sparse_tensor.convert %v1_dense : tensor<10xf64> to tensor<?xf64, #CV64>

    %v2_dense = arith.constant sparse<[[1], [4], [7]], [-1.2, 3.4, 7.7]> : tensor<10xf64>
    %v2 = sparse_tensor.convert %v2_dense : tensor<10xf64> to tensor<?xf64, #CV64>

    %v3_dense = arith.constant sparse<[[3], [5]], [1, 9]> : tensor<10xi32>
    %v3 = sparse_tensor.convert %v3_dense : tensor<10xi32> to tensor<?xi32, #CV64>

    // input -> output { replace? }
    //
    // CHECK:      Test 10
    // CHECK-NEXT: [_, -1.2, _, _, 3.4, _, _, 7.7, _, _]
    // CHECK-NEXT: [_, -1.2, _, _, 3.4, _, _, 7.7, _, _]
    //
    graphblas.print %c0 { strings=["Test 1"] } : index
    %10 = graphblas.dup %v1 : tensor<?xf64, #CV64>
    graphblas.update %v2 -> %10 : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>
    graphblas.print %10 { strings=[] } : tensor<?xf64, #CV64>
    %11 = graphblas.dup %v1 : tensor<?xf64, #CV64>
    graphblas.update %v2 -> %11 { replace=true } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>
    graphblas.print %11 { strings=[] } : tensor<?xf64, #CV64>

    // input -> output { accumulate_operator, replace? }
    //
    // CHECK:      Test 20
    // CHECK-NEXT: [_, 8.8, _, 11, 15.4, _, 13, 7.7, _, _]
    //
    graphblas.print %c0 { strings=["Test 2"] } : index
    %20 = graphblas.dup %v1 : tensor<?xf64, #CV64>
    graphblas.update %v2 -> %20 { accumulate_operator="plus" } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>
    graphblas.print %20 { strings=[] } : tensor<?xf64, #CV64>
    %21 = graphblas.dup %v1 : tensor<?xf64, #CV64>
    graphblas.update %v2 -> %21 { accumulate_operator="plus", replace=true } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>
    graphblas.print %21 { strings=[] } : tensor<?xf64, #CV64>

    // input -> output(mask)
    //
    // CHECK:      Test 30
    // CHECK-NEXT: [_, -1.2, _, _, 12, _, 13, _, _, _] 
    //
    graphblas.print %c0 { strings=["Test 3"] } : index
    %22 = graphblas.dup %v1 : tensor<?xf64, #CV64>
    graphblas.update %v2 -> %22(%mask) : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>(tensor<?xf64, #CV64>)
    graphblas.print %22 { strings=[] } : tensor<?xf64, #CV64> 
    
    // input -> output(mask) { replace }
    //
    // CHECK:      Test 40
    // CHECK-NEXT: [_, -1.2, _, _, _, _, _, _, _, _] 
    //
    graphblas.print %c0 { strings=["Test 4"] } : index
    %23 = graphblas.dup %v1 : tensor<?xf64, #CV64>
    graphblas.update %v2 -> %23(%mask) { replace = true } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>(tensor<?xf64, #CV64>)
    graphblas.print %23 { strings=[] } : tensor<?xf64, #CV64>
    
    // input -> output(mask) { replace, mask_complement }
    //
    // CHECK:      Test 50
    // CHECK-NEXT: [_, _, _, _, 3.4, _, _, 7.7, _, _]
    //
    graphblas.print %c0 { strings=["Test 5"] } : index
    %24 = graphblas.dup %v2 : tensor<?xf64, #CV64>
    graphblas.update %v2 -> %24(%mask) { replace = true, mask_complement = true } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>(tensor<?xf64, #CV64>)
    graphblas.print %24 { strings=[] } : tensor<?xf64, #CV64> 

    // input -> output(mask) { accumulate_operator }
    //
    // CHECK:      Test 60
    // CHECK-NEXT: [_, 8.8, _, 11, 12, _, 13, _, _, _]
    //
    graphblas.print %c0 { strings=["Test 6"] } : index
    %25 = graphblas.dup %v1 : tensor<?xf64, #CV64>
    graphblas.update %v2 -> %25(%mask) { accumulate_operator = "plus" } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>(tensor<?xf64, #CV64>)
    graphblas.print %25 { strings=[] } : tensor<?xf64, #CV64>

    // COM: TODO This doesn't work yet.
    // COM: input -> output(mask) { accumulate_operator, replace }
    // COM: 
    // COM: CHECK:      Test 70
    // COM: CHECK-NEXT: [_, 8.8, _, 11, _, _, _, _, _, _]
    // COM: 
    // COM: graphblas.print %c0 { strings=["Test 7"] } : index
    // COM: %26 = graphblas.dup %v1 : tensor<?xf64, #CV64>
    // COM: graphblas.update %v2 -> %26(%mask) { accumulate_operator = "plus", replace = true } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>(tensor<?xf64, #CV64>)
    // COM: graphblas.print %26 { strings=[] } : tensor<?xf64, #CV64>

    ///////////////
    // Test Matrix
    ///////////////

    // COM: input -> output { replace? }
    // COM: input -> output { accumulate_operator, replace?}
    // COM: input -> output(mask)
    // COM: input -> output(mask) { replace }
    // COM: input -> output(mask) { accumulate_operator }
    // COM: input -> output(mask) { accumulate_operator, replace }

    return
  }
}
