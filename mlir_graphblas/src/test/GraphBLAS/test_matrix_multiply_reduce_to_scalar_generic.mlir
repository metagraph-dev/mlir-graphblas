// RUN: graphblas-opt %s | graphblas-exec main | FileCheck %s

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

func @main() -> () {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c_big_num_i64 = arith.constant 9223372036854775807 : i64
    
    %a_dense = arith.constant dense<[
        [0, 1, 2, 0],
        [0, 0, 0, 3]
      ]> : tensor<2x4xi64>
    %a_csr = sparse_tensor.convert %a_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSR64>
    %a_csc = sparse_tensor.convert %a_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSC64>
    
    %b_dense = arith.constant dense<[
        [0, 7],
        [4, 0],
        [5, 0],
        [6, 8]
      ]> : tensor<4x2xi64>
    %b_csr = sparse_tensor.convert %b_dense : tensor<4x2xi64> to tensor<?x?xi64, #CSR64>
    %b_csc = sparse_tensor.convert %b_dense : tensor<4x2xi64> to tensor<?x?xi64, #CSC64>
    
    %mask_dense = arith.constant dense<[
        [9, 0],
        [0, 8]
      ]> : tensor<2x2xi64>
    %mask_csr = sparse_tensor.convert %mask_dense : tensor<2x2xi64> to tensor<?x?xi64, #CSR64>
    %mask_csc = sparse_tensor.convert %mask_dense : tensor<2x2xi64> to tensor<?x?xi64, #CSC64>

    %answer_1 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csr, %b_csr : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSR64>) to i64
    {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %28 = arith.addi %arg0, %arg1 : i64
      graphblas.yield add %28 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %28 = arith.muli %arg0, %arg1 : i64
      graphblas.yield mult %28 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_1 56
    graphblas.print %answer_1 { strings = ["answer_1 "] } : i64

    %answer_2 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csr, %b_csc : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>) to i64
    {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %28 = arith.addi %arg0, %arg1 : i64
      graphblas.yield add %28 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %28 = arith.muli %arg0, %arg1 : i64
      graphblas.yield mult %28 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_2 56
    graphblas.print %answer_2 { strings = ["answer_2 "] } : i64

    %answer_3 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csc, %b_csr : (tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSR64>) to i64 {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield add %arg1 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: index, %arg3: index, %arg4: index):
      %28 = arith.index_cast %arg4 : index to i64
      graphblas.yield mult %28 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_3 8
    graphblas.print %answer_3 { strings = ["answer_3 "] } : i64

    %answer_4 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csc, %b_csc : (tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSC64>) to i64 {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield add %arg1 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield mult %c1_i64 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_4 3
    graphblas.print %answer_4 { strings = ["answer_4 "] } : i64

    %answer_5 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csr, %b_csr, %mask_csr : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSR64>) to i64 {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %34 = arith.addi %arg0, %arg1 : i64
      graphblas.yield add %34 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %34 = arith.muli %arg0, %arg1 : i64
      graphblas.yield mult %34 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_5 38
    graphblas.print %answer_5 { strings = ["answer_5 "] } : i64

    %answer_6 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csr, %b_csc, %mask_csr : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSR64>) to i64 {
      graphblas.yield add_identity %c_big_num_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %34 = arith.cmpi slt, %arg0, %arg1 : i64
      %35 = arith.select %34, %arg0, %arg1 : i64
      graphblas.yield add %35 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %34 = arith.addi %arg0, %arg1 : i64
      graphblas.yield mult %34 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_6 16
    graphblas.print %answer_6 { strings = ["answer_6 "] } : i64

    %answer_7 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csc, %b_csr, %mask_csr : (tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSR64>) to i64 {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield add %arg1 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: index, %arg3: index, %arg4: index):
      %34 = arith.index_cast %arg4 : index to i64
      graphblas.yield mult %34 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_7 5
    graphblas.print %answer_7 { strings = ["answer_7 "] } : i64

    %answer_8 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csc, %b_csc, %mask_csr : (tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSR64>) to i64 {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield add %arg1 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield mult %c1_i64 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_8 2
    graphblas.print %answer_8 { strings = ["answer_8 "] } : i64

    %answer_9 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csr, %b_csr, %mask_csc : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>) to i64 {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %34 = arith.addi %arg0, %arg1 : i64
      graphblas.yield add %34 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %34 = arith.muli %arg0, %arg1 : i64
      graphblas.yield mult %34 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_9 38
    graphblas.print %answer_9 { strings = ["answer_9 "] } : i64

    %answer_10 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csr, %b_csc, %mask_csc : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSC64>) to i64 {
      graphblas.yield add_identity %c_big_num_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %34 = arith.cmpi slt, %arg0, %arg1 : i64
      %35 = arith.select %34, %arg0, %arg1 : i64
      graphblas.yield add %35 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %34 = arith.addi %arg0, %arg1 : i64
      graphblas.yield mult %34 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_10 16
    graphblas.print %answer_10 { strings = ["answer_10 "] } : i64

    %answer_11 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csc, %b_csr, %mask_csc : (tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>) to i64 {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield add %arg1 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: index, %arg3: index, %arg4: index):
      %34 = arith.index_cast %arg4 : index to i64
      graphblas.yield mult %34 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_11 5
    graphblas.print %answer_11 { strings = ["answer_11 "] } : i64

    %answer_12 = graphblas.matrix_multiply_reduce_to_scalar_generic %a_csc, %b_csc, %mask_csc : (tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSC64>) to i64 {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield add %arg1 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: index, %arg3: index, %arg4: index):
      %34 = arith.index_cast %arg4 : index to i64
      graphblas.yield mult %34 : i64
    }, {
        %agg_identity = arith.constant 0 : i64
        graphblas.yield agg_identity %agg_identity : i64
    }, {
        ^bb0(%lhs: i64, %rhs: i64):
            %agg_result = arith.addi %lhs, %rhs: i64
            graphblas.yield agg %agg_result : i64
    }
    // CHECK: answer_12 5
    graphblas.print %answer_12 { strings = ["answer_12 "] } : i64

    return
}


