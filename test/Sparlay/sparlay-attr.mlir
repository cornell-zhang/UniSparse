#1 = #sparlay.encoding<{
  compressMap = #sparlay.compress<fuse(0,1), trim(0)>,
  crdMap = affine_map<(i,j)->((i+j) mod 8,j floordiv 8)>
}>

func.func private @F(%arg0: tensor<?x?x?xf64, #1>) -> ()

// #2 = #sparlay.encoding<{
//   primaryMap = affine_map<(a1,a2,a3)->(a1)>,
//   secondaryMap = #sparlay.affine<(a2,a3)->(trim a2)>
// }>

// #3 = #sparlay.encoding<{
//   primaryMap = affine_map<(i,j,k)->(j,k,i)>,
//   secondaryMap = #sparlay.affine<()->()>
// }>

// #4 = #sparlay.encoding<{
//   secondaryMap = #sparlay.affine<(a1, a2)->(fuse fuse fuse trim a2, a1)>,
//   bitWidth=3
// }>

// func private @F1(%arg0: tensor<?x?x?xf64, #2>) -> (tensor<?x?x?xf64, #1>)

// func private @F2(%arg0: tensor<?x?x?xf64, #3>) -> ()

// func private @F3(%arg0: tensor<?x?x?xf64, #4>) -> ()

// //failed
// #100 = #sparlay.encoding<{
//   secondaryMap = #sparlay.affine<()->(trim d1)>
// }>

// func private @F100(%arg0: tensor<?x?x?xf64, #100>) -> ()
