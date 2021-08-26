// RUN: graphblas-opt %s | graphblas-opt --graphblas-structuralize | FileCheck %s

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

// additive operations

// CHECK-LABEL:   builtin.func @matrix_multiply_plus_X(
// CHECK:           %[[VAL_2:.*]] = constant 0.000000e+00 : f64
// CHECK:             graphblas.yield add_identity %[[VAL_2]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f64, %[[VAL_6:.*]]: f64):
// CHECK:             %[[VAL_7:.*]] = addf %[[VAL_5]], %[[VAL_6]] : f64
// CHECK:             graphblas.yield add %[[VAL_7]] : f64
// CHECK:           },  {
func @matrix_multiply_plus_X(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "plus_pair" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// CHECK-LABEL:   builtin.func @matrix_multiply_min_X(
// CHECK:           %[[VAL_2:.*]] = constant 1.7976931348623157E+308 : f64
// CHECK:             graphblas.yield add_identity %[[VAL_2]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f64, %[[VAL_6:.*]]: f64):
// CHECK:             %[[VAL_7:.*]] = cmpf olt, %[[VAL_5]], %[[VAL_6]] : f64
// CHECK:             %[[VAL_8:.*]] = select %[[VAL_7]], %[[VAL_5]], %[[VAL_6]] : f64
// CHECK:             graphblas.yield add %[[VAL_8]] : f64
// CHECK:           },  {
func @matrix_multiply_min_X(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "min_pair" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// CHECK-LABEL:   builtin.func @matrix_multiply_any_X(
// CHECK:           %[[VAL_2:.*]] = constant 0.000000e+00 : f64
// CHECK:             graphblas.yield add_identity %[[VAL_2]] : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f64, %[[VAL_6:.*]]: f64):
// CHECK:             graphblas.yield add %[[VAL_6]] : f64
// CHECK:           },  {
func @matrix_multiply_any_X(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "any_pair" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// multiplicative operations

// CHECK-LABEL:   builtin.func @matrix_multiply_X_pair(
// CHECK:           %[[VAL_3:.*]] = constant 1.000000e+00 : f64
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f64, %[[VAL_8:.*]]: f64):
// CHECK:             graphblas.yield mult %[[VAL_3]] : f64
// CHECK:           }
func @matrix_multiply_X_pair(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "any_pair" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// CHECK-LABEL:   builtin.func @matrix_multiply_X_times(
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f64, %[[VAL_7:.*]]: f64):
// CHECK:             %[[VAL_8:.*]] = mulf %[[VAL_6]], %[[VAL_7]] : f64
// CHECK:             graphblas.yield mult %[[VAL_8]] : f64
// CHECK:           }
func @matrix_multiply_X_times(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "any_times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// CHECK-LABEL:   builtin.func @matrix_multiply_X_plus(
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f64, %[[VAL_7:.*]]: f64):
// CHECK:             %[[VAL_8:.*]] = addf %[[VAL_6]], %[[VAL_7]] : f64
// CHECK:             graphblas.yield mult %[[VAL_8]] : f64
// CHECK:           }
func @matrix_multiply_X_plus(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "any_plus" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// CHECK-LABEL:   builtin.func @matrix_multiply_X_first(
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f64, %[[VAL_7:.*]]: f64):
// CHECK:             graphblas.yield mult %[[VAL_6]] : f64
// CHECK:           }
func @matrix_multiply_X_first(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "any_first" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// CHECK-LABEL:   builtin.func @matrix_multiply_X_second(
// CHECK:           },  {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f64, %[[VAL_7:.*]]: f64):
// CHECK:             graphblas.yield mult %[[VAL_7]] : f64
// CHECK:           }
func @matrix_multiply_X_second(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "any_second" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}