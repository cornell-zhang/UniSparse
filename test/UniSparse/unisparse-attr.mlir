// RUN: unisparse-opt %s | FileCheck %s

// #1 = #unisparse.encoding<{
//   compressMap = #unisparse.compress<fuse(0,1), trim(0)>,
//   crdMap = #unisparse.crd<(i,j,k)[s0,s1]->(indirect (i+j minus s0)*4 mod 7, (k + (minus i)) floordiv s1)>
// }>

#2 = #unisparse.encoding<{
  compressMap = #unisparse.compress<fuse(1,1), trim(1,2)>,
  crdMap = #unisparse.crd<(i,j)->(j,i)>
}>

#3 = #unisparse.encoding<{
  compressMap = #unisparse.compress<fuse(0,1), trim(0)>,
  crdMap = #unisparse.crd<(i,j,k)[s0,s1]->(indirect(i,j), indirect (j), (k + (minus i)) floordiv s1)>,
  indirectFunc = #unisparse.indirect<{
    sumVal = #unisparse.sum<groupBy (0), with val ne 0 -> 1 | otherwise -> 0>,
    enumVal = #unisparse.enumerate<groupBy (0), traverseBy (1), with val eq 0 -> sumVal | otherwise -> 0>,
    reorderVal = #unisparse.reorder<traverseBy (0), sumVal, descend>, // map: original matrix A -> output A' [0, 1]
    schedVal = #unisparse.schedule<traverseBy (0), sumVal, 2> //list[[]] -> list[]
  }>
}>

//CHECK-LABEL: func.func private @F
// func.func private @F(%arg0: tensor<?x?xf64, #1>) -> ()

//CHECK-LABEL: func.func private @G
func.func private @G(%arg1: tensor<?x?xf64, #2>) -> ()

func.func private @H(%arg2: tensor<?x?xf64, #3>) -> ()
// #2 = #unisparse.encoding<{
//   primaryMap = affine_map<(a1,a2,a3)->(a1)>,
//   secondaryMap = #unisparse.affine<(a2,a3)->(trim a2)>
// }>

// #3 = #unisparse.encoding<{
//   primaryMap = affine_map<(i,j,k)->(j,k,i)>,
//   secondaryMap = #unisparse.affine<()->()>
// }>

// #4 = #unisparse.encoding<{
//   secondaryMap = #unisparse.affine<(a1, a2)->(fuse fuse fuse trim a2, a1)>,
//   bitWidth=3
// }>

// func private @F1(%arg0: tensor<?x?x?xf64, #2>) -> (tensor<?x?x?xf64, #1>)

// func private @F2(%arg0: tensor<?x?x?xf64, #3>) -> ()

// func private @F3(%arg0: tensor<?x?x?xf64, #4>) -> ()

// //failed
// #100 = #unisparse.encoding<{
//   secondaryMap = #unisparse.affine<()->(trim d1)>
// }>

// func private @F100(%arg0: tensor<?x?x?xf64, #100>) -> ()
