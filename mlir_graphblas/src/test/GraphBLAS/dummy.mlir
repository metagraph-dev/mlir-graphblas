// RUN: graphblas-opt %s | graphblas-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = graphblas.foo %{{.*}} : i32
        %res = graphblas.foo %0 : i32
        return
    }
}
