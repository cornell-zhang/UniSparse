// RUN: sparlay-opt %s | sparlay-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = std.constant 1 : i32
        // CHECK: %{{.*}} = sparlay.foo %{{.*}} : i32
        %res = sparlay.foo %0 : i32
        return
    }
}
