       £K"	  АЙ6ы’Abrain.Event:2~Шъe]     NюgV	@№∞Й6ы’A"–ґ
W
keras_learning_phasePlaceholder*
dtype0*
shape: *
_output_shapes
:
a
input_1Placeholder*
dtype0*
shape: */
_output_shapes
:€€€€€€€€€  
m
random_uniform/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
W
random_uniform/minConst*
dtype0*
valueB
 *:ЌЊ*
_output_shapes
: 
W
random_uniform/maxConst*
dtype0*
valueB
 *:Ќ>*
_output_shapes
: 
†
random_uniform/RandomUniformRandomUniformrandom_uniform/shape*
dtype0*
seed2рДД*
seed±€е)*
T0*&
_output_shapes
:
b
random_uniform/subSubrandom_uniform/maxrandom_uniform/min*
T0*
_output_shapes
: 
|
random_uniform/mulMulrandom_uniform/RandomUniformrandom_uniform/sub*
T0*&
_output_shapes
:
n
random_uniformAddrandom_uniform/mulrandom_uniform/min*
T0*&
_output_shapes
:
У
convolution2d_1_WVariable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≈
convolution2d_1_W/AssignAssignconvolution2d_1_Wrandom_uniform*
validate_shape(*$
_class
loc:@convolution2d_1_W*
use_locking(*
T0*&
_output_shapes
:
М
convolution2d_1_W/readIdentityconvolution2d_1_W*$
_class
loc:@convolution2d_1_W*
T0*&
_output_shapes
:
R
ConstConst*
dtype0*
valueB*    *
_output_shapes
:
{
convolution2d_1_bVariable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
∞
convolution2d_1_b/AssignAssignconvolution2d_1_bConst*
validate_shape(*$
_class
loc:@convolution2d_1_b*
use_locking(*
T0*
_output_shapes
:
А
convolution2d_1_b/readIdentityconvolution2d_1_b*$
_class
loc:@convolution2d_1_b*
T0*
_output_shapes
:
¬
Conv2DConv2Dinput_1convolution2d_1_W/read*/
_output_shapes
:€€€€€€€€€  *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
f
Reshape/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
j
ReshapeReshapeconvolution2d_1_b/readReshape/shape*
T0*&
_output_shapes
:
U
addAddConv2DReshape*
T0*/
_output_shapes
:€€€€€€€€€  
K
ReluReluadd*
T0*/
_output_shapes
:€€€€€€€€€  
§
MaxPoolMaxPoolRelu*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
o
random_uniform_1/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
Y
random_uniform_1/minConst*
dtype0*
valueB
 *уµљ*
_output_shapes
: 
Y
random_uniform_1/maxConst*
dtype0*
valueB
 *уµ=*
_output_shapes
: 
•
random_uniform_1/RandomUniformRandomUniformrandom_uniform_1/shape*
dtype0*
seed2ЇѓЈЕ*
seed±€е)*
T0*&
_output_shapes
:
h
random_uniform_1/subSubrandom_uniform_1/maxrandom_uniform_1/min*
T0*
_output_shapes
: 
В
random_uniform_1/mulMulrandom_uniform_1/RandomUniformrandom_uniform_1/sub*
T0*&
_output_shapes
:
t
random_uniform_1Addrandom_uniform_1/mulrandom_uniform_1/min*
T0*&
_output_shapes
:
У
convolution2d_2_WVariable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
«
convolution2d_2_W/AssignAssignconvolution2d_2_Wrandom_uniform_1*
validate_shape(*$
_class
loc:@convolution2d_2_W*
use_locking(*
T0*&
_output_shapes
:
М
convolution2d_2_W/readIdentityconvolution2d_2_W*$
_class
loc:@convolution2d_2_W*
T0*&
_output_shapes
:
T
Const_1Const*
dtype0*
valueB*    *
_output_shapes
:
{
convolution2d_2_bVariable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
≤
convolution2d_2_b/AssignAssignconvolution2d_2_bConst_1*
validate_shape(*$
_class
loc:@convolution2d_2_b*
use_locking(*
T0*
_output_shapes
:
А
convolution2d_2_b/readIdentityconvolution2d_2_b*$
_class
loc:@convolution2d_2_b*
T0*
_output_shapes
:
ƒ
Conv2D_1Conv2DMaxPoolconvolution2d_2_W/read*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
h
Reshape_1/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
n
	Reshape_1Reshapeconvolution2d_2_b/readReshape_1/shape*
T0*&
_output_shapes
:
[
add_1AddConv2D_1	Reshape_1*
T0*/
_output_shapes
:€€€€€€€€€
O
Relu_1Reluadd_1*
T0*/
_output_shapes
:€€€€€€€€€
®
	MaxPool_1MaxPoolRelu_1*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
o
random_uniform_2/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
Y
random_uniform_2/minConst*
dtype0*
valueB
 *   Њ*
_output_shapes
: 
Y
random_uniform_2/maxConst*
dtype0*
valueB
 *   >*
_output_shapes
: 
•
random_uniform_2/RandomUniformRandomUniformrandom_uniform_2/shape*
dtype0*
seed2шЗй¬*
seed±€е)*
T0*&
_output_shapes
:
h
random_uniform_2/subSubrandom_uniform_2/maxrandom_uniform_2/min*
T0*
_output_shapes
: 
В
random_uniform_2/mulMulrandom_uniform_2/RandomUniformrandom_uniform_2/sub*
T0*&
_output_shapes
:
t
random_uniform_2Addrandom_uniform_2/mulrandom_uniform_2/min*
T0*&
_output_shapes
:
У
convolution2d_3_WVariable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
«
convolution2d_3_W/AssignAssignconvolution2d_3_Wrandom_uniform_2*
validate_shape(*$
_class
loc:@convolution2d_3_W*
use_locking(*
T0*&
_output_shapes
:
М
convolution2d_3_W/readIdentityconvolution2d_3_W*$
_class
loc:@convolution2d_3_W*
T0*&
_output_shapes
:
T
Const_2Const*
dtype0*
valueB*    *
_output_shapes
:
{
convolution2d_3_bVariable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
≤
convolution2d_3_b/AssignAssignconvolution2d_3_bConst_2*
validate_shape(*$
_class
loc:@convolution2d_3_b*
use_locking(*
T0*
_output_shapes
:
А
convolution2d_3_b/readIdentityconvolution2d_3_b*$
_class
loc:@convolution2d_3_b*
T0*
_output_shapes
:
∆
Conv2D_2Conv2D	MaxPool_1convolution2d_3_W/read*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
h
Reshape_2/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
n
	Reshape_2Reshapeconvolution2d_3_b/readReshape_2/shape*
T0*&
_output_shapes
:
[
add_2AddConv2D_2	Reshape_2*
T0*/
_output_shapes
:€€€€€€€€€
O
Relu_2Reluadd_2*
T0*/
_output_shapes
:€€€€€€€€€
®
	MaxPool_2MaxPoolRelu_2*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
o
random_uniform_3/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
Y
random_uniform_3/minConst*
dtype0*
valueB
 *   Њ*
_output_shapes
: 
Y
random_uniform_3/maxConst*
dtype0*
valueB
 *   >*
_output_shapes
: 
§
random_uniform_3/RandomUniformRandomUniformrandom_uniform_3/shape*
dtype0*
seed2лЃгQ*
seed±€е)*
T0*&
_output_shapes
:
h
random_uniform_3/subSubrandom_uniform_3/maxrandom_uniform_3/min*
T0*
_output_shapes
: 
В
random_uniform_3/mulMulrandom_uniform_3/RandomUniformrandom_uniform_3/sub*
T0*&
_output_shapes
:
t
random_uniform_3Addrandom_uniform_3/mulrandom_uniform_3/min*
T0*&
_output_shapes
:
У
convolution2d_4_WVariable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
«
convolution2d_4_W/AssignAssignconvolution2d_4_Wrandom_uniform_3*
validate_shape(*$
_class
loc:@convolution2d_4_W*
use_locking(*
T0*&
_output_shapes
:
М
convolution2d_4_W/readIdentityconvolution2d_4_W*$
_class
loc:@convolution2d_4_W*
T0*&
_output_shapes
:
T
Const_3Const*
dtype0*
valueB*    *
_output_shapes
:
{
convolution2d_4_bVariable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
≤
convolution2d_4_b/AssignAssignconvolution2d_4_bConst_3*
validate_shape(*$
_class
loc:@convolution2d_4_b*
use_locking(*
T0*
_output_shapes
:
А
convolution2d_4_b/readIdentityconvolution2d_4_b*$
_class
loc:@convolution2d_4_b*
T0*
_output_shapes
:
∆
Conv2D_3Conv2D	MaxPool_2convolution2d_4_W/read*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
h
Reshape_3/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
n
	Reshape_3Reshapeconvolution2d_4_b/readReshape_3/shape*
T0*&
_output_shapes
:
[
add_3AddConv2D_3	Reshape_3*
T0*/
_output_shapes
:€€€€€€€€€
O
Relu_3Reluadd_3*
T0*/
_output_shapes
:€€€€€€€€€
;
ShapeShapeRelu_3*
T0*
_output_shapes
:
\
strided_slice/packConst*
dtype0*
valueB:*
_output_shapes
:
^
strided_slice/pack_1Const*
dtype0*
valueB:*
_output_shapes
:
^
strided_slice/pack_2Const*
dtype0*
valueB:*
_output_shapes
:
ъ
strided_sliceStridedSliceShapestrided_slice/packstrided_slice/pack_1strided_slice/pack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
X
Const_4Const*
dtype0*
valueB"      *
_output_shapes
:
G
mulMulstrided_sliceConst_4*
T0*
_output_shapes
:
К
ResizeNearestNeighborResizeNearestNeighborRelu_3mul*
align_corners( *
T0*/
_output_shapes
:€€€€€€€€€
o
random_uniform_4/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
Y
random_uniform_4/minConst*
dtype0*
valueB
 *   Њ*
_output_shapes
: 
Y
random_uniform_4/maxConst*
dtype0*
valueB
 *   >*
_output_shapes
: 
•
random_uniform_4/RandomUniformRandomUniformrandom_uniform_4/shape*
dtype0*
seed2±сюџ*
seed±€е)*
T0*&
_output_shapes
:
h
random_uniform_4/subSubrandom_uniform_4/maxrandom_uniform_4/min*
T0*
_output_shapes
: 
В
random_uniform_4/mulMulrandom_uniform_4/RandomUniformrandom_uniform_4/sub*
T0*&
_output_shapes
:
t
random_uniform_4Addrandom_uniform_4/mulrandom_uniform_4/min*
T0*&
_output_shapes
:
У
convolution2d_5_WVariable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
«
convolution2d_5_W/AssignAssignconvolution2d_5_Wrandom_uniform_4*
validate_shape(*$
_class
loc:@convolution2d_5_W*
use_locking(*
T0*&
_output_shapes
:
М
convolution2d_5_W/readIdentityconvolution2d_5_W*$
_class
loc:@convolution2d_5_W*
T0*&
_output_shapes
:
T
Const_5Const*
dtype0*
valueB*    *
_output_shapes
:
{
convolution2d_5_bVariable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
≤
convolution2d_5_b/AssignAssignconvolution2d_5_bConst_5*
validate_shape(*$
_class
loc:@convolution2d_5_b*
use_locking(*
T0*
_output_shapes
:
А
convolution2d_5_b/readIdentityconvolution2d_5_b*$
_class
loc:@convolution2d_5_b*
T0*
_output_shapes
:
“
Conv2D_4Conv2DResizeNearestNeighborconvolution2d_5_W/read*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
h
Reshape_4/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
n
	Reshape_4Reshapeconvolution2d_5_b/readReshape_4/shape*
T0*&
_output_shapes
:
[
add_4AddConv2D_4	Reshape_4*
T0*/
_output_shapes
:€€€€€€€€€
O
Relu_4Reluadd_4*
T0*/
_output_shapes
:€€€€€€€€€
=
Shape_1ShapeRelu_4*
T0*
_output_shapes
:
^
strided_slice_1/packConst*
dtype0*
valueB:*
_output_shapes
:
`
strided_slice_1/pack_1Const*
dtype0*
valueB:*
_output_shapes
:
`
strided_slice_1/pack_2Const*
dtype0*
valueB:*
_output_shapes
:
Д
strided_slice_1StridedSliceShape_1strided_slice_1/packstrided_slice_1/pack_1strided_slice_1/pack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
X
Const_6Const*
dtype0*
valueB"      *
_output_shapes
:
K
mul_1Mulstrided_slice_1Const_6*
T0*
_output_shapes
:
О
ResizeNearestNeighbor_1ResizeNearestNeighborRelu_4mul_1*
align_corners( *
T0*/
_output_shapes
:€€€€€€€€€
o
random_uniform_5/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
Y
random_uniform_5/minConst*
dtype0*
valueB
 *уµљ*
_output_shapes
: 
Y
random_uniform_5/maxConst*
dtype0*
valueB
 *уµ=*
_output_shapes
: 
•
random_uniform_5/RandomUniformRandomUniformrandom_uniform_5/shape*
dtype0*
seed2£Йћ∞*
seed±€е)*
T0*&
_output_shapes
:
h
random_uniform_5/subSubrandom_uniform_5/maxrandom_uniform_5/min*
T0*
_output_shapes
: 
В
random_uniform_5/mulMulrandom_uniform_5/RandomUniformrandom_uniform_5/sub*
T0*&
_output_shapes
:
t
random_uniform_5Addrandom_uniform_5/mulrandom_uniform_5/min*
T0*&
_output_shapes
:
У
convolution2d_6_WVariable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
«
convolution2d_6_W/AssignAssignconvolution2d_6_Wrandom_uniform_5*
validate_shape(*$
_class
loc:@convolution2d_6_W*
use_locking(*
T0*&
_output_shapes
:
М
convolution2d_6_W/readIdentityconvolution2d_6_W*$
_class
loc:@convolution2d_6_W*
T0*&
_output_shapes
:
T
Const_7Const*
dtype0*
valueB*    *
_output_shapes
:
{
convolution2d_6_bVariable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
≤
convolution2d_6_b/AssignAssignconvolution2d_6_bConst_7*
validate_shape(*$
_class
loc:@convolution2d_6_b*
use_locking(*
T0*
_output_shapes
:
А
convolution2d_6_b/readIdentityconvolution2d_6_b*$
_class
loc:@convolution2d_6_b*
T0*
_output_shapes
:
‘
Conv2D_5Conv2DResizeNearestNeighbor_1convolution2d_6_W/read*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
h
Reshape_5/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
n
	Reshape_5Reshapeconvolution2d_6_b/readReshape_5/shape*
T0*&
_output_shapes
:
[
add_5AddConv2D_5	Reshape_5*
T0*/
_output_shapes
:€€€€€€€€€
O
Relu_5Reluadd_5*
T0*/
_output_shapes
:€€€€€€€€€
=
Shape_2ShapeRelu_5*
T0*
_output_shapes
:
^
strided_slice_2/packConst*
dtype0*
valueB:*
_output_shapes
:
`
strided_slice_2/pack_1Const*
dtype0*
valueB:*
_output_shapes
:
`
strided_slice_2/pack_2Const*
dtype0*
valueB:*
_output_shapes
:
Д
strided_slice_2StridedSliceShape_2strided_slice_2/packstrided_slice_2/pack_1strided_slice_2/pack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
X
Const_8Const*
dtype0*
valueB"      *
_output_shapes
:
K
mul_2Mulstrided_slice_2Const_8*
T0*
_output_shapes
:
О
ResizeNearestNeighbor_2ResizeNearestNeighborRelu_5mul_2*
align_corners( *
T0*/
_output_shapes
:€€€€€€€€€  
o
random_uniform_6/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
Y
random_uniform_6/minConst*
dtype0*
valueB
 *:ЌЊ*
_output_shapes
: 
Y
random_uniform_6/maxConst*
dtype0*
valueB
 *:Ќ>*
_output_shapes
: 
§
random_uniform_6/RandomUniformRandomUniformrandom_uniform_6/shape*
dtype0*
seed2ѓРЪG*
seed±€е)*
T0*&
_output_shapes
:
h
random_uniform_6/subSubrandom_uniform_6/maxrandom_uniform_6/min*
T0*
_output_shapes
: 
В
random_uniform_6/mulMulrandom_uniform_6/RandomUniformrandom_uniform_6/sub*
T0*&
_output_shapes
:
t
random_uniform_6Addrandom_uniform_6/mulrandom_uniform_6/min*
T0*&
_output_shapes
:
У
convolution2d_7_WVariable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
«
convolution2d_7_W/AssignAssignconvolution2d_7_Wrandom_uniform_6*
validate_shape(*$
_class
loc:@convolution2d_7_W*
use_locking(*
T0*&
_output_shapes
:
М
convolution2d_7_W/readIdentityconvolution2d_7_W*$
_class
loc:@convolution2d_7_W*
T0*&
_output_shapes
:
T
Const_9Const*
dtype0*
valueB*    *
_output_shapes
:
{
convolution2d_7_bVariable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
≤
convolution2d_7_b/AssignAssignconvolution2d_7_bConst_9*
validate_shape(*$
_class
loc:@convolution2d_7_b*
use_locking(*
T0*
_output_shapes
:
А
convolution2d_7_b/readIdentityconvolution2d_7_b*$
_class
loc:@convolution2d_7_b*
T0*
_output_shapes
:
‘
Conv2D_6Conv2DResizeNearestNeighbor_2convolution2d_7_W/read*/
_output_shapes
:€€€€€€€€€  *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
h
Reshape_6/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
n
	Reshape_6Reshapeconvolution2d_7_b/readReshape_6/shape*
T0*&
_output_shapes
:
[
add_6AddConv2D_6	Reshape_6*
T0*/
_output_shapes
:€€€€€€€€€  
S
SigmoidSigmoidadd_6*
T0*/
_output_shapes
:€€€€€€€€€  
[
Variable/initial_valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
j
VariableVariable*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
Ґ
Variable/AssignAssignVariableVariable/initial_value*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
: 
a
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*
_output_shapes
: 
]
Variable_1/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
l

Variable_1Variable*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
™
Variable_1/AssignAssign
Variable_1Variable_1/initial_value*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
g
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes
: 
]
Variable_2/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
l

Variable_2Variable*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
™
Variable_2/AssignAssign
Variable_2Variable_2/initial_value*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
: 
g
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes
: 
l
convolution2d_7_sample_weightsPlaceholder*
dtype0*
shape: *#
_output_shapes
:€€€€€€€€€
Л
convolution2d_7_targetPlaceholder*
dtype0*
shape: *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
M
Const_10Const*
dtype0*
valueB
 *Хњ÷3*
_output_shapes
: 
J
sub/xConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
<
subSubsub/xConst_10*
T0*
_output_shapes
: 
h
clip_by_value/MinimumMinimumSigmoidsub*
T0*/
_output_shapes
:€€€€€€€€€  
s
clip_by_valueMaximumclip_by_value/MinimumConst_10*
T0*/
_output_shapes
:€€€€€€€€€  
L
sub_1/xConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
^
sub_1Subsub_1/xclip_by_value*
T0*/
_output_shapes
:€€€€€€€€€  
Z
divDivclip_by_valuesub_1*
T0*/
_output_shapes
:€€€€€€€€€  
I
LogLogdiv*
T0*/
_output_shapes
:€€€€€€€€€  
d
logistic_loss/zeros_like	ZerosLikeLog*
T0*/
_output_shapes
:€€€€€€€€€  
Г
logistic_loss/GreaterEqualGreaterEqualLoglogistic_loss/zeros_like*
T0*/
_output_shapes
:€€€€€€€€€  
У
logistic_loss/SelectSelectlogistic_loss/GreaterEqualLoglogistic_loss/zeros_like*
T0*/
_output_shapes
:€€€€€€€€€  
W
logistic_loss/NegNegLog*
T0*/
_output_shapes
:€€€€€€€€€  
О
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/NegLog*
T0*/
_output_shapes
:€€€€€€€€€  
o
logistic_loss/mulMulLogconvolution2d_7_target*
T0*/
_output_shapes
:€€€€€€€€€  
{
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0*/
_output_shapes
:€€€€€€€€€  
j
logistic_loss/ExpExplogistic_loss/Select_1*
T0*/
_output_shapes
:€€€€€€€€€  
X
logistic_loss/add/xConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
z
logistic_loss/addAddlogistic_loss/add/xlogistic_loss/Exp*
T0*/
_output_shapes
:€€€€€€€€€  
e
logistic_loss/LogLoglogistic_loss/add*
T0*/
_output_shapes
:€€€€€€€€€  
t
logistic_lossAddlogistic_loss/sublogistic_loss/Log*
T0*/
_output_shapes
:€€€€€€€€€  
X
Mean/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
z
MeanMeanlogistic_lossMean/reduction_indices*
T0*
	keep_dims( *+
_output_shapes
:€€€€€€€€€  
i
Mean_1/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:
m
Mean_1MeanMeanMean_1/reduction_indices*
T0*
	keep_dims( *#
_output_shapes
:€€€€€€€€€
b
mul_3MulMean_1convolution2d_7_sample_weights*
T0*#
_output_shapes
:€€€€€€€€€
O

NotEqual/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n
NotEqualNotEqualconvolution2d_7_sample_weights
NotEqual/y*
T0*#
_output_shapes
:€€€€€€€€€
S
CastCastNotEqual*

DstT0*

SrcT0
*#
_output_shapes
:€€€€€€€€€
R
Const_11Const*
dtype0*
valueB: *
_output_shapes
:
P
Mean_2MeanCastConst_11*
T0*
	keep_dims( *
_output_shapes
: 
I
div_1Divmul_3Mean_2*
T0*#
_output_shapes
:€€€€€€€€€
R
Const_12Const*
dtype0*
valueB: *
_output_shapes
:
Q
Mean_3Meandiv_1Const_12*
T0*
	keep_dims( *
_output_shapes
: 
L
mul_4/xConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
>
mul_4Mulmul_4/xMean_3*
T0*
_output_shapes
: 
]
Variable_3/initial_valueConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
l

Variable_3Variable*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
™
Variable_3/AssignAssign
Variable_3Variable_3/initial_value*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
: 
g
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes
: 
]
Variable_4/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
l

Variable_4Variable*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
™
Variable_4/AssignAssign
Variable_4Variable_4/initial_value*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes
: 
g
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes
: 
]
Variable_5/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
l

Variable_5Variable*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
™
Variable_5/AssignAssign
Variable_5Variable_5/initial_value*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes
: 
g
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes
: 


group_depsNoOp^mul_4
\
gradients/ShapeShapemul_4*
_class

loc:@mul_4*
T0*
_output_shapes
: 
n
gradients/ConstConst*
dtype0*
_class

loc:@mul_4*
valueB
 *  А?*
_output_shapes
: 
s
gradients/FillFillgradients/Shapegradients/Const*
_class

loc:@mul_4*
T0*
_output_shapes
: 
i
gradients/mul_4_grad/ShapeShapemul_4/x*
_class

loc:@mul_4*
T0*
_output_shapes
: 
j
gradients/mul_4_grad/Shape_1ShapeMean_3*
_class

loc:@mul_4*
T0*
_output_shapes
: 
Ћ
*gradients/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_4_grad/Shapegradients/mul_4_grad/Shape_1*
_class

loc:@mul_4*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
r
gradients/mul_4_grad/mulMulgradients/FillMean_3*
_class

loc:@mul_4*
T0*
_output_shapes
: 
≥
gradients/mul_4_grad/SumSumgradients/mul_4_grad/mul*gradients/mul_4_grad/BroadcastGradientArgs*
_class

loc:@mul_4*
T0*
	keep_dims( *
_output_shapes
:
Ш
gradients/mul_4_grad/ReshapeReshapegradients/mul_4_grad/Sumgradients/mul_4_grad/Shape*
_class

loc:@mul_4*
T0*
_output_shapes
: 
u
gradients/mul_4_grad/mul_1Mulmul_4/xgradients/Fill*
_class

loc:@mul_4*
T0*
_output_shapes
: 
є
gradients/mul_4_grad/Sum_1Sumgradients/mul_4_grad/mul_1,gradients/mul_4_grad/BroadcastGradientArgs:1*
_class

loc:@mul_4*
T0*
	keep_dims( *
_output_shapes
:
Ю
gradients/mul_4_grad/Reshape_1Reshapegradients/mul_4_grad/Sum_1gradients/mul_4_grad/Shape_1*
_class

loc:@mul_4*
T0*
_output_shapes
: 
И
#gradients/Mean_3_grad/Reshape/shapeConst*
dtype0*
_class
loc:@Mean_3*
valueB:*
_output_shapes
:
≠
gradients/Mean_3_grad/ReshapeReshapegradients/mul_4_grad/Reshape_1#gradients/Mean_3_grad/Reshape/shape*
_class
loc:@Mean_3*
T0*
_output_shapes
:
k
gradients/Mean_3_grad/ShapeShapediv_1*
_class
loc:@Mean_3*
T0*
_output_shapes
:
І
gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*
_class
loc:@Mean_3*
T0*#
_output_shapes
:€€€€€€€€€
m
gradients/Mean_3_grad/Shape_1Shapediv_1*
_class
loc:@Mean_3*
T0*
_output_shapes
:
l
gradients/Mean_3_grad/Shape_2ShapeMean_3*
_class
loc:@Mean_3*
T0*
_output_shapes
: 
А
gradients/Mean_3_grad/ConstConst*
dtype0*
_class
loc:@Mean_3*
valueB: *
_output_shapes
:
Ђ
gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*
_class
loc:@Mean_3*
T0*
	keep_dims( *
_output_shapes
: 
В
gradients/Mean_3_grad/Const_1Const*
dtype0*
_class
loc:@Mean_3*
valueB: *
_output_shapes
:
ѓ
gradients/Mean_3_grad/Prod_1Prodgradients/Mean_3_grad/Shape_2gradients/Mean_3_grad/Const_1*
_class
loc:@Mean_3*
T0*
	keep_dims( *
_output_shapes
: 
|
gradients/Mean_3_grad/Maximum/yConst*
dtype0*
_class
loc:@Mean_3*
value	B :*
_output_shapes
: 
£
gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
_class
loc:@Mean_3*
T0*
_output_shapes
: 
Ь
gradients/Mean_3_grad/floordivDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
_class
loc:@Mean_3*
T0*
_output_shapes
: 
Н
gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*

DstT0*
_class
loc:@Mean_3*

SrcT0*
_output_shapes
: 
•
gradients/Mean_3_grad/truedivDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*
_class
loc:@Mean_3*
T0*#
_output_shapes
:€€€€€€€€€
i
gradients/div_1_grad/ShapeShapemul_3*
_class

loc:@div_1*
T0*
_output_shapes
:
j
gradients/div_1_grad/Shape_1ShapeMean_2*
_class

loc:@div_1*
T0*
_output_shapes
: 
Ћ
*gradients/div_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_1_grad/Shapegradients/div_1_grad/Shape_1*
_class

loc:@div_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Т
gradients/div_1_grad/truedivDivgradients/Mean_3_grad/truedivMean_2*
_class

loc:@div_1*
T0*#
_output_shapes
:€€€€€€€€€
Ј
gradients/div_1_grad/SumSumgradients/div_1_grad/truediv*gradients/div_1_grad/BroadcastGradientArgs*
_class

loc:@div_1*
T0*
	keep_dims( *
_output_shapes
:
•
gradients/div_1_grad/ReshapeReshapegradients/div_1_grad/Sumgradients/div_1_grad/Shape*
_class

loc:@div_1*
T0*#
_output_shapes
:€€€€€€€€€
n
gradients/div_1_grad/NegNegmul_3*
_class

loc:@div_1*
T0*#
_output_shapes
:€€€€€€€€€
h
gradients/div_1_grad/SquareSquareMean_2*
_class

loc:@div_1*
T0*
_output_shapes
: 
§
gradients/div_1_grad/truediv_1Divgradients/div_1_grad/Neggradients/div_1_grad/Square*
_class

loc:@div_1*
T0*#
_output_shapes
:€€€€€€€€€
¶
gradients/div_1_grad/mulMulgradients/Mean_3_grad/truedivgradients/div_1_grad/truediv_1*
_class

loc:@div_1*
T0*#
_output_shapes
:€€€€€€€€€
Ј
gradients/div_1_grad/Sum_1Sumgradients/div_1_grad/mul,gradients/div_1_grad/BroadcastGradientArgs:1*
_class

loc:@div_1*
T0*
	keep_dims( *
_output_shapes
:
Ю
gradients/div_1_grad/Reshape_1Reshapegradients/div_1_grad/Sum_1gradients/div_1_grad/Shape_1*
_class

loc:@div_1*
T0*
_output_shapes
: 
j
gradients/mul_3_grad/ShapeShapeMean_1*
_class

loc:@mul_3*
T0*
_output_shapes
:
Д
gradients/mul_3_grad/Shape_1Shapeconvolution2d_7_sample_weights*
_class

loc:@mul_3*
T0*
_output_shapes
:
Ћ
*gradients/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_3_grad/Shapegradients/mul_3_grad/Shape_1*
_class

loc:@mul_3*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
•
gradients/mul_3_grad/mulMulgradients/div_1_grad/Reshapeconvolution2d_7_sample_weights*
_class

loc:@mul_3*
T0*#
_output_shapes
:€€€€€€€€€
≥
gradients/mul_3_grad/SumSumgradients/mul_3_grad/mul*gradients/mul_3_grad/BroadcastGradientArgs*
_class

loc:@mul_3*
T0*
	keep_dims( *
_output_shapes
:
•
gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*
_class

loc:@mul_3*
T0*#
_output_shapes
:€€€€€€€€€
П
gradients/mul_3_grad/mul_1MulMean_1gradients/div_1_grad/Reshape*
_class

loc:@mul_3*
T0*#
_output_shapes
:€€€€€€€€€
є
gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*
_class

loc:@mul_3*
T0*
	keep_dims( *
_output_shapes
:
Ђ
gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*
_class

loc:@mul_3*
T0*#
_output_shapes
:€€€€€€€€€
j
gradients/Mean_1_grad/ShapeShapeMean*
_class
loc:@Mean_1*
T0*
_output_shapes
:
{
gradients/Mean_1_grad/SizeSizegradients/Mean_1_grad/Shape*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
Ц
gradients/Mean_1_grad/addAddMean_1/reduction_indicesgradients/Mean_1_grad/Size*
_class
loc:@Mean_1*
T0*
_output_shapes
:
Ч
gradients/Mean_1_grad/modModgradients/Mean_1_grad/addgradients/Mean_1_grad/Size*
_class
loc:@Mean_1*
T0*
_output_shapes
:
Б
gradients/Mean_1_grad/Shape_1Shapegradients/Mean_1_grad/mod*
_class
loc:@Mean_1*
T0*
_output_shapes
:
~
!gradients/Mean_1_grad/range/startConst*
dtype0*
_class
loc:@Mean_1*
value	B : *
_output_shapes
: 
~
!gradients/Mean_1_grad/range/deltaConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
љ
gradients/Mean_1_grad/rangeRange!gradients/Mean_1_grad/range/startgradients/Mean_1_grad/Size!gradients/Mean_1_grad/range/delta*
_class
loc:@Mean_1*
_output_shapes
:
}
 gradients/Mean_1_grad/Fill/valueConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
£
gradients/Mean_1_grad/FillFillgradients/Mean_1_grad/Shape_1 gradients/Mean_1_grad/Fill/value*
_class
loc:@Mean_1*
T0*
_output_shapes
:
ч
#gradients/Mean_1_grad/DynamicStitchDynamicStitchgradients/Mean_1_grad/rangegradients/Mean_1_grad/modgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Fill*
_class
loc:@Mean_1*
T0*#
_output_shapes
:€€€€€€€€€*
N
|
gradients/Mean_1_grad/Maximum/yConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
Ј
gradients/Mean_1_grad/MaximumMaximum#gradients/Mean_1_grad/DynamicStitchgradients/Mean_1_grad/Maximum/y*
_class
loc:@Mean_1*
T0*#
_output_shapes
:€€€€€€€€€
°
gradients/Mean_1_grad/floordivDivgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Maximum*
_class
loc:@Mean_1*
T0*
_output_shapes
:
©
gradients/Mean_1_grad/ReshapeReshapegradients/mul_3_grad/Reshape#gradients/Mean_1_grad/DynamicStitch*
_class
loc:@Mean_1*
T0*
_output_shapes
:
ƒ
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/floordiv*
_class
loc:@Mean_1*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
l
gradients/Mean_1_grad/Shape_2ShapeMean*
_class
loc:@Mean_1*
T0*
_output_shapes
:
n
gradients/Mean_1_grad/Shape_3ShapeMean_1*
_class
loc:@Mean_1*
T0*
_output_shapes
:
А
gradients/Mean_1_grad/ConstConst*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
Ђ
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const*
_class
loc:@Mean_1*
T0*
	keep_dims( *
_output_shapes
: 
В
gradients/Mean_1_grad/Const_1Const*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
ѓ
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_3gradients/Mean_1_grad/Const_1*
_class
loc:@Mean_1*
T0*
	keep_dims( *
_output_shapes
: 
~
!gradients/Mean_1_grad/Maximum_1/yConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
І
gradients/Mean_1_grad/Maximum_1Maximumgradients/Mean_1_grad/Prod_1!gradients/Mean_1_grad/Maximum_1/y*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
†
 gradients/Mean_1_grad/floordiv_1Divgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum_1*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
П
gradients/Mean_1_grad/CastCast gradients/Mean_1_grad/floordiv_1*

DstT0*
_class
loc:@Mean_1*

SrcT0*
_output_shapes
: 
≠
gradients/Mean_1_grad/truedivDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
_class
loc:@Mean_1*
T0*+
_output_shapes
:€€€€€€€€€  
o
gradients/Mean_grad/ShapeShapelogistic_loss*
_class
	loc:@Mean*
T0*
_output_shapes
:
u
gradients/Mean_grad/SizeSizegradients/Mean_grad/Shape*
_class
	loc:@Mean*
T0*
_output_shapes
: 
К
gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
_class
	loc:@Mean*
T0*
_output_shapes
: 
Л
gradients/Mean_grad/modModgradients/Mean_grad/addgradients/Mean_grad/Size*
_class
	loc:@Mean*
T0*
_output_shapes
: 
y
gradients/Mean_grad/Shape_1Shapegradients/Mean_grad/mod*
_class
	loc:@Mean*
T0*
_output_shapes
: 
z
gradients/Mean_grad/range/startConst*
dtype0*
_class
	loc:@Mean*
value	B : *
_output_shapes
: 
z
gradients/Mean_grad/range/deltaConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 
≥
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*
_class
	loc:@Mean*
_output_shapes
:
y
gradients/Mean_grad/Fill/valueConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 
Ч
gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
_class
	loc:@Mean*
T0*
_output_shapes
: 
л
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
_class
	loc:@Mean*
T0*#
_output_shapes
:€€€€€€€€€*
N
x
gradients/Mean_grad/Maximum/yConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 
ѓ
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
_class
	loc:@Mean*
T0*#
_output_shapes
:€€€€€€€€€
Щ
gradients/Mean_grad/floordivDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
_class
	loc:@Mean*
T0*
_output_shapes
:
§
gradients/Mean_grad/ReshapeReshapegradients/Mean_1_grad/truediv!gradients/Mean_grad/DynamicStitch*
_class
	loc:@Mean*
T0*
_output_shapes
:
…
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*
_class
	loc:@Mean*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
q
gradients/Mean_grad/Shape_2Shapelogistic_loss*
_class
	loc:@Mean*
T0*
_output_shapes
:
h
gradients/Mean_grad/Shape_3ShapeMean*
_class
	loc:@Mean*
T0*
_output_shapes
:
|
gradients/Mean_grad/ConstConst*
dtype0*
_class
	loc:@Mean*
valueB: *
_output_shapes
:
£
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
_class
	loc:@Mean*
T0*
	keep_dims( *
_output_shapes
: 
~
gradients/Mean_grad/Const_1Const*
dtype0*
_class
	loc:@Mean*
valueB: *
_output_shapes
:
І
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
_class
	loc:@Mean*
T0*
	keep_dims( *
_output_shapes
: 
z
gradients/Mean_grad/Maximum_1/yConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 
Я
gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
_class
	loc:@Mean*
T0*
_output_shapes
: 
Ш
gradients/Mean_grad/floordiv_1Divgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
_class
	loc:@Mean*
T0*
_output_shapes
: 
Й
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

DstT0*
_class
	loc:@Mean*

SrcT0*
_output_shapes
: 
©
gradients/Mean_grad/truedivDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_class
	loc:@Mean*
T0*/
_output_shapes
:€€€€€€€€€  
Е
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub* 
_class
loc:@logistic_loss*
T0*
_output_shapes
:
З
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log* 
_class
loc:@logistic_loss*
T0*
_output_shapes
:
л
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1* 
_class
loc:@logistic_loss*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ќ
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs* 
_class
loc:@logistic_loss*
T0*
	keep_dims( *
_output_shapes
:
—
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape* 
_class
loc:@logistic_loss*
T0*/
_output_shapes
:€€€€€€€€€  
“
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1* 
_class
loc:@logistic_loss*
T0*
	keep_dims( *
_output_shapes
:
„
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1* 
_class
loc:@logistic_loss*
T0*/
_output_shapes
:€€€€€€€€€  
Р
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*$
_class
loc:@logistic_loss/sub*
T0*
_output_shapes
:
П
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*$
_class
loc:@logistic_loss/sub*
T0*
_output_shapes
:
ы
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*$
_class
loc:@logistic_loss/sub*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
г
$gradients/logistic_loss/sub_grad/SumSum$gradients/logistic_loss_grad/Reshape6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*$
_class
loc:@logistic_loss/sub*
T0*
	keep_dims( *
_output_shapes
:
б
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*$
_class
loc:@logistic_loss/sub*
T0*/
_output_shapes
:€€€€€€€€€  
з
&gradients/logistic_loss/sub_grad/Sum_1Sum$gradients/logistic_loss_grad/Reshape8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*$
_class
loc:@logistic_loss/sub*
T0*
	keep_dims( *
_output_shapes
:
Ь
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*$
_class
loc:@logistic_loss/sub*
T0*
_output_shapes
:
е
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*$
_class
loc:@logistic_loss/sub*
T0*/
_output_shapes
:€€€€€€€€€  
«
$gradients/logistic_loss/Log_grad/InvInvlogistic_loss/add'^gradients/logistic_loss_grad/Reshape_1*$
_class
loc:@logistic_loss/Log*
T0*/
_output_shapes
:€€€€€€€€€  
ў
$gradients/logistic_loss/Log_grad/mulMul&gradients/logistic_loss_grad/Reshape_1$gradients/logistic_loss/Log_grad/Inv*$
_class
loc:@logistic_loss/Log*
T0*/
_output_shapes
:€€€€€€€€€  
£
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikeLog*'
_class
loc:@logistic_loss/Select*
T0*/
_output_shapes
:€€€€€€€€€  
Н
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual(gradients/logistic_loss/sub_grad/Reshape.gradients/logistic_loss/Select_grad/zeros_like*'
_class
loc:@logistic_loss/Select*
T0*/
_output_shapes
:€€€€€€€€€  
П
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like(gradients/logistic_loss/sub_grad/Reshape*'
_class
loc:@logistic_loss/Select*
T0*/
_output_shapes
:€€€€€€€€€  

&gradients/logistic_loss/mul_grad/ShapeShapeLog*$
_class
loc:@logistic_loss/mul*
T0*
_output_shapes
:
Ф
(gradients/logistic_loss/mul_grad/Shape_1Shapeconvolution2d_7_target*$
_class
loc:@logistic_loss/mul*
T0*
_output_shapes
:
ы
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*$
_class
loc:@logistic_loss/mul*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ѕ
$gradients/logistic_loss/mul_grad/mulMul*gradients/logistic_loss/sub_grad/Reshape_1convolution2d_7_target*$
_class
loc:@logistic_loss/mul*
T0*/
_output_shapes
:€€€€€€€€€  
г
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*$
_class
loc:@logistic_loss/mul*
T0*
	keep_dims( *
_output_shapes
:
б
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*$
_class
loc:@logistic_loss/mul*
T0*/
_output_shapes
:€€€€€€€€€  
Њ
&gradients/logistic_loss/mul_grad/mul_1MulLog*gradients/logistic_loss/sub_grad/Reshape_1*$
_class
loc:@logistic_loss/mul*
T0*/
_output_shapes
:€€€€€€€€€  
й
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*$
_class
loc:@logistic_loss/mul*
T0*
	keep_dims( *
_output_shapes
:
В
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*$
_class
loc:@logistic_loss/mul*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Н
&gradients/logistic_loss/add_grad/ShapeShapelogistic_loss/add/x*$
_class
loc:@logistic_loss/add*
T0*
_output_shapes
: 
П
(gradients/logistic_loss/add_grad/Shape_1Shapelogistic_loss/Exp*$
_class
loc:@logistic_loss/add*
T0*
_output_shapes
:
ы
6gradients/logistic_loss/add_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/add_grad/Shape(gradients/logistic_loss/add_grad/Shape_1*$
_class
loc:@logistic_loss/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
г
$gradients/logistic_loss/add_grad/SumSum$gradients/logistic_loss/Log_grad/mul6gradients/logistic_loss/add_grad/BroadcastGradientArgs*$
_class
loc:@logistic_loss/add*
T0*
	keep_dims( *
_output_shapes
:
»
(gradients/logistic_loss/add_grad/ReshapeReshape$gradients/logistic_loss/add_grad/Sum&gradients/logistic_loss/add_grad/Shape*$
_class
loc:@logistic_loss/add*
T0*
_output_shapes
: 
з
&gradients/logistic_loss/add_grad/Sum_1Sum$gradients/logistic_loss/Log_grad/mul8gradients/logistic_loss/add_grad/BroadcastGradientArgs:1*$
_class
loc:@logistic_loss/add*
T0*
	keep_dims( *
_output_shapes
:
з
*gradients/logistic_loss/add_grad/Reshape_1Reshape&gradients/logistic_loss/add_grad/Sum_1(gradients/logistic_loss/add_grad/Shape_1*$
_class
loc:@logistic_loss/add*
T0*/
_output_shapes
:€€€€€€€€€  
 
$gradients/logistic_loss/Exp_grad/mulMul*gradients/logistic_loss/add_grad/Reshape_1logistic_loss/Exp*$
_class
loc:@logistic_loss/Exp*
T0*/
_output_shapes
:€€€€€€€€€  
µ
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*)
_class
loc:@logistic_loss/Select_1*
T0*/
_output_shapes
:€€€€€€€€€  
П
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*)
_class
loc:@logistic_loss/Select_1*
T0*/
_output_shapes
:€€€€€€€€€  
С
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*)
_class
loc:@logistic_loss/Select_1*
T0*/
_output_shapes
:€€€€€€€€€  
є
$gradients/logistic_loss/Neg_grad/NegNeg,gradients/logistic_loss/Select_1_grad/Select*$
_class
loc:@logistic_loss/Neg*
T0*/
_output_shapes
:€€€€€€€€€  
Ѓ
gradients/AddNAddN*gradients/logistic_loss/Select_grad/Select(gradients/logistic_loss/mul_grad/Reshape.gradients/logistic_loss/Select_1_grad/Select_1$gradients/logistic_loss/Neg_grad/Neg*'
_class
loc:@logistic_loss/Select*
T0*/
_output_shapes
:€€€€€€€€€  *
N
Е
gradients/Log_grad/InvInvdiv^gradients/AddN*
_class

loc:@Log*
T0*/
_output_shapes
:€€€€€€€€€  
Ч
gradients/Log_grad/mulMulgradients/AddNgradients/Log_grad/Inv*
_class

loc:@Log*
T0*/
_output_shapes
:€€€€€€€€€  
m
gradients/div_grad/ShapeShapeclip_by_value*
_class

loc:@div*
T0*
_output_shapes
:
g
gradients/div_grad/Shape_1Shapesub_1*
_class

loc:@div*
T0*
_output_shapes
:
√
(gradients/div_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_grad/Shapegradients/div_grad/Shape_1*
_class

loc:@div*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Т
gradients/div_grad/truedivDivgradients/Log_grad/mulsub_1*
_class

loc:@div*
T0*/
_output_shapes
:€€€€€€€€€  
ѓ
gradients/div_grad/SumSumgradients/div_grad/truediv(gradients/div_grad/BroadcastGradientArgs*
_class

loc:@div*
T0*
	keep_dims( *
_output_shapes
:
©
gradients/div_grad/ReshapeReshapegradients/div_grad/Sumgradients/div_grad/Shape*
_class

loc:@div*
T0*/
_output_shapes
:€€€€€€€€€  
~
gradients/div_grad/NegNegclip_by_value*
_class

loc:@div*
T0*/
_output_shapes
:€€€€€€€€€  
|
gradients/div_grad/SquareSquaresub_1*
_class

loc:@div*
T0*/
_output_shapes
:€€€€€€€€€  
®
gradients/div_grad/truediv_1Divgradients/div_grad/Neggradients/div_grad/Square*
_class

loc:@div*
T0*/
_output_shapes
:€€€€€€€€€  
•
gradients/div_grad/mulMulgradients/Log_grad/mulgradients/div_grad/truediv_1*
_class

loc:@div*
T0*/
_output_shapes
:€€€€€€€€€  
ѓ
gradients/div_grad/Sum_1Sumgradients/div_grad/mul*gradients/div_grad/BroadcastGradientArgs:1*
_class

loc:@div*
T0*
	keep_dims( *
_output_shapes
:
ѓ
gradients/div_grad/Reshape_1Reshapegradients/div_grad/Sum_1gradients/div_grad/Shape_1*
_class

loc:@div*
T0*/
_output_shapes
:€€€€€€€€€  
i
gradients/sub_1_grad/ShapeShapesub_1/x*
_class

loc:@sub_1*
T0*
_output_shapes
: 
s
gradients/sub_1_grad/Shape_1Shapeclip_by_value*
_class

loc:@sub_1*
T0*
_output_shapes
:
Ћ
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*
_class

loc:@sub_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ј
gradients/sub_1_grad/SumSumgradients/div_grad/Reshape_1*gradients/sub_1_grad/BroadcastGradientArgs*
_class

loc:@sub_1*
T0*
	keep_dims( *
_output_shapes
:
Ш
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
_class

loc:@sub_1*
T0*
_output_shapes
: 
ї
gradients/sub_1_grad/Sum_1Sumgradients/div_grad/Reshape_1,gradients/sub_1_grad/BroadcastGradientArgs:1*
_class

loc:@sub_1*
T0*
	keep_dims( *
_output_shapes
:
x
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
_class

loc:@sub_1*
T0*
_output_shapes
:
µ
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*
_class

loc:@sub_1*
T0*/
_output_shapes
:€€€€€€€€€  
ѓ
gradients/AddN_1AddNgradients/div_grad/Reshapegradients/sub_1_grad/Reshape_1*
_class

loc:@div*
T0*/
_output_shapes
:€€€€€€€€€  *
N
Й
"gradients/clip_by_value_grad/ShapeShapeclip_by_value/Minimum* 
_class
loc:@clip_by_value*
T0*
_output_shapes
:
|
$gradients/clip_by_value_grad/Shape_1ShapeConst_10* 
_class
loc:@clip_by_value*
T0*
_output_shapes
: 
Ж
$gradients/clip_by_value_grad/Shape_2Shapegradients/AddN_1* 
_class
loc:@clip_by_value*
T0*
_output_shapes
:
П
(gradients/clip_by_value_grad/zeros/ConstConst*
dtype0* 
_class
loc:@clip_by_value*
valueB
 *    *
_output_shapes
: 
÷
"gradients/clip_by_value_grad/zerosFill$gradients/clip_by_value_grad/Shape_2(gradients/clip_by_value_grad/zeros/Const* 
_class
loc:@clip_by_value*
T0*/
_output_shapes
:€€€€€€€€€  
ґ
)gradients/clip_by_value_grad/GreaterEqualGreaterEqualclip_by_value/MinimumConst_10* 
_class
loc:@clip_by_value*
T0*/
_output_shapes
:€€€€€€€€€  
л
2gradients/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/clip_by_value_grad/Shape$gradients/clip_by_value_grad/Shape_1* 
_class
loc:@clip_by_value*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
к
#gradients/clip_by_value_grad/SelectSelect)gradients/clip_by_value_grad/GreaterEqualgradients/AddN_1"gradients/clip_by_value_grad/zeros* 
_class
loc:@clip_by_value*
T0*/
_output_shapes
:€€€€€€€€€  
≥
'gradients/clip_by_value_grad/LogicalNot
LogicalNot)gradients/clip_by_value_grad/GreaterEqual* 
_class
loc:@clip_by_value*/
_output_shapes
:€€€€€€€€€  
к
%gradients/clip_by_value_grad/Select_1Select'gradients/clip_by_value_grad/LogicalNotgradients/AddN_1"gradients/clip_by_value_grad/zeros* 
_class
loc:@clip_by_value*
T0*/
_output_shapes
:€€€€€€€€€  
÷
 gradients/clip_by_value_grad/SumSum#gradients/clip_by_value_grad/Select2gradients/clip_by_value_grad/BroadcastGradientArgs* 
_class
loc:@clip_by_value*
T0*
	keep_dims( *
_output_shapes
:
—
$gradients/clip_by_value_grad/ReshapeReshape gradients/clip_by_value_grad/Sum"gradients/clip_by_value_grad/Shape* 
_class
loc:@clip_by_value*
T0*/
_output_shapes
:€€€€€€€€€  
№
"gradients/clip_by_value_grad/Sum_1Sum%gradients/clip_by_value_grad/Select_14gradients/clip_by_value_grad/BroadcastGradientArgs:1* 
_class
loc:@clip_by_value*
T0*
	keep_dims( *
_output_shapes
:
Њ
&gradients/clip_by_value_grad/Reshape_1Reshape"gradients/clip_by_value_grad/Sum_1$gradients/clip_by_value_grad/Shape_1* 
_class
loc:@clip_by_value*
T0*
_output_shapes
: 
Л
*gradients/clip_by_value/Minimum_grad/ShapeShapeSigmoid*(
_class
loc:@clip_by_value/Minimum*
T0*
_output_shapes
:
З
,gradients/clip_by_value/Minimum_grad/Shape_1Shapesub*(
_class
loc:@clip_by_value/Minimum*
T0*
_output_shapes
: 
™
,gradients/clip_by_value/Minimum_grad/Shape_2Shape$gradients/clip_by_value_grad/Reshape*(
_class
loc:@clip_by_value/Minimum*
T0*
_output_shapes
:
Я
0gradients/clip_by_value/Minimum_grad/zeros/ConstConst*
dtype0*(
_class
loc:@clip_by_value/Minimum*
valueB
 *    *
_output_shapes
: 
ц
*gradients/clip_by_value/Minimum_grad/zerosFill,gradients/clip_by_value/Minimum_grad/Shape_20gradients/clip_by_value/Minimum_grad/zeros/Const*(
_class
loc:@clip_by_value/Minimum*
T0*/
_output_shapes
:€€€€€€€€€  
≠
.gradients/clip_by_value/Minimum_grad/LessEqual	LessEqualSigmoidsub*(
_class
loc:@clip_by_value/Minimum*
T0*/
_output_shapes
:€€€€€€€€€  
Л
:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/clip_by_value/Minimum_grad/Shape,gradients/clip_by_value/Minimum_grad/Shape_1*(
_class
loc:@clip_by_value/Minimum*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ы
+gradients/clip_by_value/Minimum_grad/SelectSelect.gradients/clip_by_value/Minimum_grad/LessEqual$gradients/clip_by_value_grad/Reshape*gradients/clip_by_value/Minimum_grad/zeros*(
_class
loc:@clip_by_value/Minimum*
T0*/
_output_shapes
:€€€€€€€€€  
»
/gradients/clip_by_value/Minimum_grad/LogicalNot
LogicalNot.gradients/clip_by_value/Minimum_grad/LessEqual*(
_class
loc:@clip_by_value/Minimum*/
_output_shapes
:€€€€€€€€€  
Ю
-gradients/clip_by_value/Minimum_grad/Select_1Select/gradients/clip_by_value/Minimum_grad/LogicalNot$gradients/clip_by_value_grad/Reshape*gradients/clip_by_value/Minimum_grad/zeros*(
_class
loc:@clip_by_value/Minimum*
T0*/
_output_shapes
:€€€€€€€€€  
ц
(gradients/clip_by_value/Minimum_grad/SumSum+gradients/clip_by_value/Minimum_grad/Select:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs*(
_class
loc:@clip_by_value/Minimum*
T0*
	keep_dims( *
_output_shapes
:
с
,gradients/clip_by_value/Minimum_grad/ReshapeReshape(gradients/clip_by_value/Minimum_grad/Sum*gradients/clip_by_value/Minimum_grad/Shape*(
_class
loc:@clip_by_value/Minimum*
T0*/
_output_shapes
:€€€€€€€€€  
ь
*gradients/clip_by_value/Minimum_grad/Sum_1Sum-gradients/clip_by_value/Minimum_grad/Select_1<gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*(
_class
loc:@clip_by_value/Minimum*
T0*
	keep_dims( *
_output_shapes
:
ё
.gradients/clip_by_value/Minimum_grad/Reshape_1Reshape*gradients/clip_by_value/Minimum_grad/Sum_1,gradients/clip_by_value/Minimum_grad/Shape_1*(
_class
loc:@clip_by_value/Minimum*
T0*
_output_shapes
: 
Њ
"gradients/Sigmoid_grad/SigmoidGradSigmoidGradSigmoid,gradients/clip_by_value/Minimum_grad/Reshape*
_class
loc:@Sigmoid*
T0*/
_output_shapes
:€€€€€€€€€  
l
gradients/add_6_grad/ShapeShapeConv2D_6*
_class

loc:@add_6*
T0*
_output_shapes
:
o
gradients/add_6_grad/Shape_1Shape	Reshape_6*
_class

loc:@add_6*
T0*
_output_shapes
:
Ћ
*gradients/add_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_6_grad/Shapegradients/add_6_grad/Shape_1*
_class

loc:@add_6*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
љ
gradients/add_6_grad/SumSum"gradients/Sigmoid_grad/SigmoidGrad*gradients/add_6_grad/BroadcastGradientArgs*
_class

loc:@add_6*
T0*
	keep_dims( *
_output_shapes
:
±
gradients/add_6_grad/ReshapeReshapegradients/add_6_grad/Sumgradients/add_6_grad/Shape*
_class

loc:@add_6*
T0*/
_output_shapes
:€€€€€€€€€  
Ѕ
gradients/add_6_grad/Sum_1Sum"gradients/Sigmoid_grad/SigmoidGrad,gradients/add_6_grad/BroadcastGradientArgs:1*
_class

loc:@add_6*
T0*
	keep_dims( *
_output_shapes
:
Ѓ
gradients/add_6_grad/Reshape_1Reshapegradients/add_6_grad/Sum_1gradients/add_6_grad/Shape_1*
_class

loc:@add_6*
T0*&
_output_shapes
:
Б
gradients/Conv2D_6_grad/ShapeShapeResizeNearestNeighbor_2*
_class
loc:@Conv2D_6*
T0*
_output_shapes
:
≈
+gradients/Conv2D_6_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_6_grad/Shapeconvolution2d_7_W/readgradients/add_6_grad/Reshape*/
_output_shapes
:€€€€€€€€€  *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_6
В
gradients/Conv2D_6_grad/Shape_1Shapeconvolution2d_7_W/read*
_class
loc:@Conv2D_6*
T0*
_output_shapes
:
Ѕ
,gradients/Conv2D_6_grad/Conv2DBackpropFilterConv2DBackpropFilterResizeNearestNeighbor_2gradients/Conv2D_6_grad/Shape_1gradients/add_6_grad/Reshape*&
_output_shapes
:*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_6
В
gradients/Reshape_6_grad/ShapeShapeconvolution2d_7_b/read*
_class
loc:@Reshape_6*
T0*
_output_shapes
:
Ѓ
 gradients/Reshape_6_grad/ReshapeReshapegradients/add_6_grad/Reshape_1gradients/Reshape_6_grad/Shape*
_class
loc:@Reshape_6*
T0*
_output_shapes
:
¬
Egradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad/sizeConst*
dtype0**
_class 
loc:@ResizeNearestNeighbor_2*
valueB"      *
_output_shapes
:
ћ
@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGradResizeNearestNeighborGrad+gradients/Conv2D_6_grad/Conv2DBackpropInputEgradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad/size*
align_corners( **
_class 
loc:@ResizeNearestNeighbor_2*
T0*/
_output_shapes
:€€€€€€€€€
…
gradients/Relu_5_grad/ReluGradReluGrad@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGradRelu_5*
_class
loc:@Relu_5*
T0*/
_output_shapes
:€€€€€€€€€
l
gradients/add_5_grad/ShapeShapeConv2D_5*
_class

loc:@add_5*
T0*
_output_shapes
:
o
gradients/add_5_grad/Shape_1Shape	Reshape_5*
_class

loc:@add_5*
T0*
_output_shapes
:
Ћ
*gradients/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_5_grad/Shapegradients/add_5_grad/Shape_1*
_class

loc:@add_5*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
є
gradients/add_5_grad/SumSumgradients/Relu_5_grad/ReluGrad*gradients/add_5_grad/BroadcastGradientArgs*
_class

loc:@add_5*
T0*
	keep_dims( *
_output_shapes
:
±
gradients/add_5_grad/ReshapeReshapegradients/add_5_grad/Sumgradients/add_5_grad/Shape*
_class

loc:@add_5*
T0*/
_output_shapes
:€€€€€€€€€
љ
gradients/add_5_grad/Sum_1Sumgradients/Relu_5_grad/ReluGrad,gradients/add_5_grad/BroadcastGradientArgs:1*
_class

loc:@add_5*
T0*
	keep_dims( *
_output_shapes
:
Ѓ
gradients/add_5_grad/Reshape_1Reshapegradients/add_5_grad/Sum_1gradients/add_5_grad/Shape_1*
_class

loc:@add_5*
T0*&
_output_shapes
:
Б
gradients/Conv2D_5_grad/ShapeShapeResizeNearestNeighbor_1*
_class
loc:@Conv2D_5*
T0*
_output_shapes
:
≈
+gradients/Conv2D_5_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_5_grad/Shapeconvolution2d_6_W/readgradients/add_5_grad/Reshape*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_5
В
gradients/Conv2D_5_grad/Shape_1Shapeconvolution2d_6_W/read*
_class
loc:@Conv2D_5*
T0*
_output_shapes
:
Ѕ
,gradients/Conv2D_5_grad/Conv2DBackpropFilterConv2DBackpropFilterResizeNearestNeighbor_1gradients/Conv2D_5_grad/Shape_1gradients/add_5_grad/Reshape*&
_output_shapes
:*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_5
В
gradients/Reshape_5_grad/ShapeShapeconvolution2d_6_b/read*
_class
loc:@Reshape_5*
T0*
_output_shapes
:
Ѓ
 gradients/Reshape_5_grad/ReshapeReshapegradients/add_5_grad/Reshape_1gradients/Reshape_5_grad/Shape*
_class
loc:@Reshape_5*
T0*
_output_shapes
:
¬
Egradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad/sizeConst*
dtype0**
_class 
loc:@ResizeNearestNeighbor_1*
valueB"      *
_output_shapes
:
ћ
@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGradResizeNearestNeighborGrad+gradients/Conv2D_5_grad/Conv2DBackpropInputEgradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad/size*
align_corners( **
_class 
loc:@ResizeNearestNeighbor_1*
T0*/
_output_shapes
:€€€€€€€€€
…
gradients/Relu_4_grad/ReluGradReluGrad@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGradRelu_4*
_class
loc:@Relu_4*
T0*/
_output_shapes
:€€€€€€€€€
l
gradients/add_4_grad/ShapeShapeConv2D_4*
_class

loc:@add_4*
T0*
_output_shapes
:
o
gradients/add_4_grad/Shape_1Shape	Reshape_4*
_class

loc:@add_4*
T0*
_output_shapes
:
Ћ
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*
_class

loc:@add_4*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
є
gradients/add_4_grad/SumSumgradients/Relu_4_grad/ReluGrad*gradients/add_4_grad/BroadcastGradientArgs*
_class

loc:@add_4*
T0*
	keep_dims( *
_output_shapes
:
±
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
_class

loc:@add_4*
T0*/
_output_shapes
:€€€€€€€€€
љ
gradients/add_4_grad/Sum_1Sumgradients/Relu_4_grad/ReluGrad,gradients/add_4_grad/BroadcastGradientArgs:1*
_class

loc:@add_4*
T0*
	keep_dims( *
_output_shapes
:
Ѓ
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
_class

loc:@add_4*
T0*&
_output_shapes
:

gradients/Conv2D_4_grad/ShapeShapeResizeNearestNeighbor*
_class
loc:@Conv2D_4*
T0*
_output_shapes
:
≈
+gradients/Conv2D_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_4_grad/Shapeconvolution2d_5_W/readgradients/add_4_grad/Reshape*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_4
В
gradients/Conv2D_4_grad/Shape_1Shapeconvolution2d_5_W/read*
_class
loc:@Conv2D_4*
T0*
_output_shapes
:
њ
,gradients/Conv2D_4_grad/Conv2DBackpropFilterConv2DBackpropFilterResizeNearestNeighborgradients/Conv2D_4_grad/Shape_1gradients/add_4_grad/Reshape*&
_output_shapes
:*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_4
В
gradients/Reshape_4_grad/ShapeShapeconvolution2d_5_b/read*
_class
loc:@Reshape_4*
T0*
_output_shapes
:
Ѓ
 gradients/Reshape_4_grad/ReshapeReshapegradients/add_4_grad/Reshape_1gradients/Reshape_4_grad/Shape*
_class
loc:@Reshape_4*
T0*
_output_shapes
:
Њ
Cgradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
dtype0*(
_class
loc:@ResizeNearestNeighbor*
valueB"      *
_output_shapes
:
∆
>gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGrad+gradients/Conv2D_4_grad/Conv2DBackpropInputCgradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *(
_class
loc:@ResizeNearestNeighbor*
T0*/
_output_shapes
:€€€€€€€€€
«
gradients/Relu_3_grad/ReluGradReluGrad>gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradRelu_3*
_class
loc:@Relu_3*
T0*/
_output_shapes
:€€€€€€€€€
l
gradients/add_3_grad/ShapeShapeConv2D_3*
_class

loc:@add_3*
T0*
_output_shapes
:
o
gradients/add_3_grad/Shape_1Shape	Reshape_3*
_class

loc:@add_3*
T0*
_output_shapes
:
Ћ
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
_class

loc:@add_3*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
є
gradients/add_3_grad/SumSumgradients/Relu_3_grad/ReluGrad*gradients/add_3_grad/BroadcastGradientArgs*
_class

loc:@add_3*
T0*
	keep_dims( *
_output_shapes
:
±
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
_class

loc:@add_3*
T0*/
_output_shapes
:€€€€€€€€€
љ
gradients/add_3_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad,gradients/add_3_grad/BroadcastGradientArgs:1*
_class

loc:@add_3*
T0*
	keep_dims( *
_output_shapes
:
Ѓ
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_class

loc:@add_3*
T0*&
_output_shapes
:
s
gradients/Conv2D_3_grad/ShapeShape	MaxPool_2*
_class
loc:@Conv2D_3*
T0*
_output_shapes
:
≈
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/Shapeconvolution2d_4_W/readgradients/add_3_grad/Reshape*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_3
В
gradients/Conv2D_3_grad/Shape_1Shapeconvolution2d_4_W/read*
_class
loc:@Conv2D_3*
T0*
_output_shapes
:
≥
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilter	MaxPool_2gradients/Conv2D_3_grad/Shape_1gradients/add_3_grad/Reshape*&
_output_shapes
:*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_3
В
gradients/Reshape_3_grad/ShapeShapeconvolution2d_4_b/read*
_class
loc:@Reshape_3*
T0*
_output_shapes
:
Ѓ
 gradients/Reshape_3_grad/ReshapeReshapegradients/add_3_grad/Reshape_1gradients/Reshape_3_grad/Shape*
_class
loc:@Reshape_3*
T0*
_output_shapes
:
Э
$gradients/MaxPool_2_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_2+gradients/Conv2D_3_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
_class
loc:@MaxPool_2*
T0
≠
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_2_grad/MaxPoolGradRelu_2*
_class
loc:@Relu_2*
T0*/
_output_shapes
:€€€€€€€€€
l
gradients/add_2_grad/ShapeShapeConv2D_2*
_class

loc:@add_2*
T0*
_output_shapes
:
o
gradients/add_2_grad/Shape_1Shape	Reshape_2*
_class

loc:@add_2*
T0*
_output_shapes
:
Ћ
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
_class

loc:@add_2*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
є
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
_class

loc:@add_2*
T0*
	keep_dims( *
_output_shapes
:
±
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
_class

loc:@add_2*
T0*/
_output_shapes
:€€€€€€€€€
љ
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
_class

loc:@add_2*
T0*
	keep_dims( *
_output_shapes
:
Ѓ
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_class

loc:@add_2*
T0*&
_output_shapes
:
s
gradients/Conv2D_2_grad/ShapeShape	MaxPool_1*
_class
loc:@Conv2D_2*
T0*
_output_shapes
:
≈
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/Shapeconvolution2d_3_W/readgradients/add_2_grad/Reshape*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_2
В
gradients/Conv2D_2_grad/Shape_1Shapeconvolution2d_3_W/read*
_class
loc:@Conv2D_2*
T0*
_output_shapes
:
≥
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilter	MaxPool_1gradients/Conv2D_2_grad/Shape_1gradients/add_2_grad/Reshape*&
_output_shapes
:*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_2
В
gradients/Reshape_2_grad/ShapeShapeconvolution2d_3_b/read*
_class
loc:@Reshape_2*
T0*
_output_shapes
:
Ѓ
 gradients/Reshape_2_grad/ReshapeReshapegradients/add_2_grad/Reshape_1gradients/Reshape_2_grad/Shape*
_class
loc:@Reshape_2*
T0*
_output_shapes
:
Э
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1+gradients/Conv2D_2_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
_class
loc:@MaxPool_1*
T0
≠
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
_class
loc:@Relu_1*
T0*/
_output_shapes
:€€€€€€€€€
l
gradients/add_1_grad/ShapeShapeConv2D_1*
_class

loc:@add_1*
T0*
_output_shapes
:
o
gradients/add_1_grad/Shape_1Shape	Reshape_1*
_class

loc:@add_1*
T0*
_output_shapes
:
Ћ
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
_class

loc:@add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
є
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
_class

loc:@add_1*
T0*
	keep_dims( *
_output_shapes
:
±
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
_class

loc:@add_1*
T0*/
_output_shapes
:€€€€€€€€€
љ
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
_class

loc:@add_1*
T0*
	keep_dims( *
_output_shapes
:
Ѓ
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_class

loc:@add_1*
T0*&
_output_shapes
:
q
gradients/Conv2D_1_grad/ShapeShapeMaxPool*
_class
loc:@Conv2D_1*
T0*
_output_shapes
:
≈
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/Shapeconvolution2d_2_W/readgradients/add_1_grad/Reshape*/
_output_shapes
:€€€€€€€€€*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_1
В
gradients/Conv2D_1_grad/Shape_1Shapeconvolution2d_2_W/read*
_class
loc:@Conv2D_1*
T0*
_output_shapes
:
±
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPoolgradients/Conv2D_1_grad/Shape_1gradients/add_1_grad/Reshape*&
_output_shapes
:*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D_1
В
gradients/Reshape_1_grad/ShapeShapeconvolution2d_2_b/read*
_class
loc:@Reshape_1*
T0*
_output_shapes
:
Ѓ
 gradients/Reshape_1_grad/ReshapeReshapegradients/add_1_grad/Reshape_1gradients/Reshape_1_grad/Shape*
_class
loc:@Reshape_1*
T0*
_output_shapes
:
Х
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool+gradients/Conv2D_1_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€  *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
_class
loc:@MaxPool*
T0
•
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
_class
	loc:@Relu*
T0*/
_output_shapes
:€€€€€€€€€  
f
gradients/add_grad/ShapeShapeConv2D*
_class

loc:@add*
T0*
_output_shapes
:
i
gradients/add_grad/Shape_1ShapeReshape*
_class

loc:@add*
T0*
_output_shapes
:
√
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
_class

loc:@add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
_class

loc:@add*
T0*
	keep_dims( *
_output_shapes
:
©
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
_class

loc:@add*
T0*/
_output_shapes
:€€€€€€€€€  
µ
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
_class

loc:@add*
T0*
	keep_dims( *
_output_shapes
:
¶
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_class

loc:@add*
T0*&
_output_shapes
:
m
gradients/Conv2D_grad/ShapeShapeinput_1*
_class
loc:@Conv2D*
T0*
_output_shapes
:
љ
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/Shapeconvolution2d_1_W/readgradients/add_grad/Reshape*/
_output_shapes
:€€€€€€€€€  *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D
~
gradients/Conv2D_grad/Shape_1Shapeconvolution2d_1_W/read*
_class
loc:@Conv2D*
T0*
_output_shapes
:
©
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput_1gradients/Conv2D_grad/Shape_1gradients/add_grad/Reshape*&
_output_shapes
:*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
_class
loc:@Conv2D
~
gradients/Reshape_grad/ShapeShapeconvolution2d_1_b/read*
_class
loc:@Reshape*
T0*
_output_shapes
:
¶
gradients/Reshape_grad/ReshapeReshapegradients/add_grad/Reshape_1gradients/Reshape_grad/Shape*
_class
loc:@Reshape*
T0*
_output_shapes
:
m
Const_13Const*
dtype0*%
valueB*    *&
_output_shapes
:
М

Variable_6Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
™
Variable_6/AssignAssign
Variable_6Const_13*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*&
_output_shapes
:
w
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*&
_output_shapes
:
U
Const_14Const*
dtype0*
valueB*    *
_output_shapes
:
t

Variable_7Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
Ю
Variable_7/AssignAssign
Variable_7Const_14*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:
m
Const_15Const*
dtype0*%
valueB*    *&
_output_shapes
:
М

Variable_8Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
™
Variable_8/AssignAssign
Variable_8Const_15*
validate_shape(*
_class
loc:@Variable_8*
use_locking(*
T0*&
_output_shapes
:
w
Variable_8/readIdentity
Variable_8*
_class
loc:@Variable_8*
T0*&
_output_shapes
:
U
Const_16Const*
dtype0*
valueB*    *
_output_shapes
:
t

Variable_9Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
Ю
Variable_9/AssignAssign
Variable_9Const_16*
validate_shape(*
_class
loc:@Variable_9*
use_locking(*
T0*
_output_shapes
:
k
Variable_9/readIdentity
Variable_9*
_class
loc:@Variable_9*
T0*
_output_shapes
:
m
Const_17Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_10Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_10/AssignAssignVariable_10Const_17*
validate_shape(*
_class
loc:@Variable_10*
use_locking(*
T0*&
_output_shapes
:
z
Variable_10/readIdentityVariable_10*
_class
loc:@Variable_10*
T0*&
_output_shapes
:
U
Const_18Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_11Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_11/AssignAssignVariable_11Const_18*
validate_shape(*
_class
loc:@Variable_11*
use_locking(*
T0*
_output_shapes
:
n
Variable_11/readIdentityVariable_11*
_class
loc:@Variable_11*
T0*
_output_shapes
:
m
Const_19Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_12Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_12/AssignAssignVariable_12Const_19*
validate_shape(*
_class
loc:@Variable_12*
use_locking(*
T0*&
_output_shapes
:
z
Variable_12/readIdentityVariable_12*
_class
loc:@Variable_12*
T0*&
_output_shapes
:
U
Const_20Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_13Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_13/AssignAssignVariable_13Const_20*
validate_shape(*
_class
loc:@Variable_13*
use_locking(*
T0*
_output_shapes
:
n
Variable_13/readIdentityVariable_13*
_class
loc:@Variable_13*
T0*
_output_shapes
:
m
Const_21Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_14Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_14/AssignAssignVariable_14Const_21*
validate_shape(*
_class
loc:@Variable_14*
use_locking(*
T0*&
_output_shapes
:
z
Variable_14/readIdentityVariable_14*
_class
loc:@Variable_14*
T0*&
_output_shapes
:
U
Const_22Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_15Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_15/AssignAssignVariable_15Const_22*
validate_shape(*
_class
loc:@Variable_15*
use_locking(*
T0*
_output_shapes
:
n
Variable_15/readIdentityVariable_15*
_class
loc:@Variable_15*
T0*
_output_shapes
:
m
Const_23Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_16Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_16/AssignAssignVariable_16Const_23*
validate_shape(*
_class
loc:@Variable_16*
use_locking(*
T0*&
_output_shapes
:
z
Variable_16/readIdentityVariable_16*
_class
loc:@Variable_16*
T0*&
_output_shapes
:
U
Const_24Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_17Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_17/AssignAssignVariable_17Const_24*
validate_shape(*
_class
loc:@Variable_17*
use_locking(*
T0*
_output_shapes
:
n
Variable_17/readIdentityVariable_17*
_class
loc:@Variable_17*
T0*
_output_shapes
:
m
Const_25Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_18Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_18/AssignAssignVariable_18Const_25*
validate_shape(*
_class
loc:@Variable_18*
use_locking(*
T0*&
_output_shapes
:
z
Variable_18/readIdentityVariable_18*
_class
loc:@Variable_18*
T0*&
_output_shapes
:
U
Const_26Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_19Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_19/AssignAssignVariable_19Const_26*
validate_shape(*
_class
loc:@Variable_19*
use_locking(*
T0*
_output_shapes
:
n
Variable_19/readIdentityVariable_19*
_class
loc:@Variable_19*
T0*
_output_shapes
:
m
Const_27Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_20Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_20/AssignAssignVariable_20Const_27*
validate_shape(*
_class
loc:@Variable_20*
use_locking(*
T0*&
_output_shapes
:
z
Variable_20/readIdentityVariable_20*
_class
loc:@Variable_20*
T0*&
_output_shapes
:
U
Const_28Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_21Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_21/AssignAssignVariable_21Const_28*
validate_shape(*
_class
loc:@Variable_21*
use_locking(*
T0*
_output_shapes
:
n
Variable_21/readIdentityVariable_21*
_class
loc:@Variable_21*
T0*
_output_shapes
:
m
Const_29Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_22Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_22/AssignAssignVariable_22Const_29*
validate_shape(*
_class
loc:@Variable_22*
use_locking(*
T0*&
_output_shapes
:
z
Variable_22/readIdentityVariable_22*
_class
loc:@Variable_22*
T0*&
_output_shapes
:
U
Const_30Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_23Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_23/AssignAssignVariable_23Const_30*
validate_shape(*
_class
loc:@Variable_23*
use_locking(*
T0*
_output_shapes
:
n
Variable_23/readIdentityVariable_23*
_class
loc:@Variable_23*
T0*
_output_shapes
:
m
Const_31Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_24Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_24/AssignAssignVariable_24Const_31*
validate_shape(*
_class
loc:@Variable_24*
use_locking(*
T0*&
_output_shapes
:
z
Variable_24/readIdentityVariable_24*
_class
loc:@Variable_24*
T0*&
_output_shapes
:
U
Const_32Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_25Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_25/AssignAssignVariable_25Const_32*
validate_shape(*
_class
loc:@Variable_25*
use_locking(*
T0*
_output_shapes
:
n
Variable_25/readIdentityVariable_25*
_class
loc:@Variable_25*
T0*
_output_shapes
:
m
Const_33Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_26Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_26/AssignAssignVariable_26Const_33*
validate_shape(*
_class
loc:@Variable_26*
use_locking(*
T0*&
_output_shapes
:
z
Variable_26/readIdentityVariable_26*
_class
loc:@Variable_26*
T0*&
_output_shapes
:
U
Const_34Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_27Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_27/AssignAssignVariable_27Const_34*
validate_shape(*
_class
loc:@Variable_27*
use_locking(*
T0*
_output_shapes
:
n
Variable_27/readIdentityVariable_27*
_class
loc:@Variable_27*
T0*
_output_shapes
:
m
Const_35Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_28Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_28/AssignAssignVariable_28Const_35*
validate_shape(*
_class
loc:@Variable_28*
use_locking(*
T0*&
_output_shapes
:
z
Variable_28/readIdentityVariable_28*
_class
loc:@Variable_28*
T0*&
_output_shapes
:
U
Const_36Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_29Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_29/AssignAssignVariable_29Const_36*
validate_shape(*
_class
loc:@Variable_29*
use_locking(*
T0*
_output_shapes
:
n
Variable_29/readIdentityVariable_29*
_class
loc:@Variable_29*
T0*
_output_shapes
:
m
Const_37Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_30Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_30/AssignAssignVariable_30Const_37*
validate_shape(*
_class
loc:@Variable_30*
use_locking(*
T0*&
_output_shapes
:
z
Variable_30/readIdentityVariable_30*
_class
loc:@Variable_30*
T0*&
_output_shapes
:
U
Const_38Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_31Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_31/AssignAssignVariable_31Const_38*
validate_shape(*
_class
loc:@Variable_31*
use_locking(*
T0*
_output_shapes
:
n
Variable_31/readIdentityVariable_31*
_class
loc:@Variable_31*
T0*
_output_shapes
:
m
Const_39Const*
dtype0*%
valueB*    *&
_output_shapes
:
Н
Variable_32Variable*
dtype0*
shape:*
	container *
shared_name *&
_output_shapes
:
≠
Variable_32/AssignAssignVariable_32Const_39*
validate_shape(*
_class
loc:@Variable_32*
use_locking(*
T0*&
_output_shapes
:
z
Variable_32/readIdentityVariable_32*
_class
loc:@Variable_32*
T0*&
_output_shapes
:
U
Const_40Const*
dtype0*
valueB*    *
_output_shapes
:
u
Variable_33Variable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
°
Variable_33/AssignAssignVariable_33Const_40*
validate_shape(*
_class
loc:@Variable_33*
use_locking(*
T0*
_output_shapes
:
n
Variable_33/readIdentityVariable_33*
_class
loc:@Variable_33*
T0*
_output_shapes
:
L
mul_5/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
W
mul_5Mulmul_5/xVariable_6/read*
T0*&
_output_shapes
:
m
SquareSquare*gradients/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
L
mul_6/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
N
mul_6Mulmul_6/xSquare*
T0*&
_output_shapes
:
K
add_7Addmul_5mul_6*
T0*&
_output_shapes
:
Ь
AssignAssign
Variable_6add_7*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*&
_output_shapes
:
L
add_8/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
X
add_8AddVariable_20/readadd_8/y*
T0*&
_output_shapes
:
M
Const_41Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_42Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
d
clip_by_value_1/MinimumMinimumadd_8Const_42*
T0*&
_output_shapes
:
n
clip_by_value_1Maximumclip_by_value_1/MinimumConst_41*
T0*&
_output_shapes
:
N
SqrtSqrtclip_by_value_1*
T0*&
_output_shapes
:
o
mul_7Mul*gradients/Conv2D_grad/Conv2DBackpropFilterSqrt*
T0*&
_output_shapes
:
L
add_9/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
M
add_9Addadd_7add_9/y*
T0*&
_output_shapes
:
M
Const_43Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_44Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
d
clip_by_value_2/MinimumMinimumadd_9Const_44*
T0*&
_output_shapes
:
n
clip_by_value_2Maximumclip_by_value_2/MinimumConst_43*
T0*&
_output_shapes
:
P
Sqrt_1Sqrtclip_by_value_2*
T0*&
_output_shapes
:
L
div_2Divmul_7Sqrt_1*
T0*&
_output_shapes
:
U
mul_8MulVariable_3/readdiv_2*
T0*&
_output_shapes
:
\
sub_2Subconvolution2d_1_W/readmul_8*
T0*&
_output_shapes
:
ђ
Assign_1Assignconvolution2d_1_Wsub_2*
validate_shape(*$
_class
loc:@convolution2d_1_W*
use_locking(*
T0*&
_output_shapes
:
L
mul_9/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
X
mul_9Mulmul_9/xVariable_20/read*
T0*&
_output_shapes
:
J
Square_1Squarediv_2*
T0*&
_output_shapes
:
M
mul_10/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
R
mul_10Mulmul_10/xSquare_1*
T0*&
_output_shapes
:
M
add_10Addmul_9mul_10*
T0*&
_output_shapes
:
°
Assign_2AssignVariable_20add_10*
validate_shape(*
_class
loc:@Variable_20*
use_locking(*
T0*&
_output_shapes
:
M
mul_11/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
M
mul_11Mulmul_11/xVariable_7/read*
T0*
_output_shapes
:
W
Square_2Squaregradients/Reshape_grad/Reshape*
T0*
_output_shapes
:
M
mul_12/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
F
mul_12Mulmul_12/xSquare_2*
T0*
_output_shapes
:
B
add_11Addmul_11mul_12*
T0*
_output_shapes
:
У
Assign_3Assign
Variable_7add_11*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
M
add_12/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
N
add_12AddVariable_21/readadd_12/y*
T0*
_output_shapes
:
M
Const_45Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_46Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Y
clip_by_value_3/MinimumMinimumadd_12Const_46*
T0*
_output_shapes
:
b
clip_by_value_3Maximumclip_by_value_3/MinimumConst_45*
T0*
_output_shapes
:
D
Sqrt_2Sqrtclip_by_value_3*
T0*
_output_shapes
:
Z
mul_13Mulgradients/Reshape_grad/ReshapeSqrt_2*
T0*
_output_shapes
:
M
add_13/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
D
add_13Addadd_11add_13/y*
T0*
_output_shapes
:
M
Const_47Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_48Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Y
clip_by_value_4/MinimumMinimumadd_13Const_48*
T0*
_output_shapes
:
b
clip_by_value_4Maximumclip_by_value_4/MinimumConst_47*
T0*
_output_shapes
:
D
Sqrt_3Sqrtclip_by_value_4*
T0*
_output_shapes
:
A
div_3Divmul_13Sqrt_3*
T0*
_output_shapes
:
J
mul_14MulVariable_3/readdiv_3*
T0*
_output_shapes
:
Q
sub_3Subconvolution2d_1_b/readmul_14*
T0*
_output_shapes
:
†
Assign_4Assignconvolution2d_1_bsub_3*
validate_shape(*$
_class
loc:@convolution2d_1_b*
use_locking(*
T0*
_output_shapes
:
M
mul_15/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_15Mulmul_15/xVariable_21/read*
T0*
_output_shapes
:
>
Square_3Squarediv_3*
T0*
_output_shapes
:
M
mul_16/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
F
mul_16Mulmul_16/xSquare_3*
T0*
_output_shapes
:
B
add_14Addmul_15mul_16*
T0*
_output_shapes
:
Х
Assign_5AssignVariable_21add_14*
validate_shape(*
_class
loc:@Variable_21*
use_locking(*
T0*
_output_shapes
:
M
mul_17/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Y
mul_17Mulmul_17/xVariable_8/read*
T0*&
_output_shapes
:
q
Square_4Square,gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
M
mul_18/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
R
mul_18Mulmul_18/xSquare_4*
T0*&
_output_shapes
:
N
add_15Addmul_17mul_18*
T0*&
_output_shapes
:
Я
Assign_6Assign
Variable_8add_15*
validate_shape(*
_class
loc:@Variable_8*
use_locking(*
T0*&
_output_shapes
:
M
add_16/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
Z
add_16AddVariable_22/readadd_16/y*
T0*&
_output_shapes
:
M
Const_49Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_50Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
e
clip_by_value_5/MinimumMinimumadd_16Const_50*
T0*&
_output_shapes
:
n
clip_by_value_5Maximumclip_by_value_5/MinimumConst_49*
T0*&
_output_shapes
:
P
Sqrt_4Sqrtclip_by_value_5*
T0*&
_output_shapes
:
t
mul_19Mul,gradients/Conv2D_1_grad/Conv2DBackpropFilterSqrt_4*
T0*&
_output_shapes
:
M
add_17/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
P
add_17Addadd_15add_17/y*
T0*&
_output_shapes
:
M
Const_51Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_52Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
e
clip_by_value_6/MinimumMinimumadd_17Const_52*
T0*&
_output_shapes
:
n
clip_by_value_6Maximumclip_by_value_6/MinimumConst_51*
T0*&
_output_shapes
:
P
Sqrt_5Sqrtclip_by_value_6*
T0*&
_output_shapes
:
M
div_4Divmul_19Sqrt_5*
T0*&
_output_shapes
:
V
mul_20MulVariable_3/readdiv_4*
T0*&
_output_shapes
:
]
sub_4Subconvolution2d_2_W/readmul_20*
T0*&
_output_shapes
:
ђ
Assign_7Assignconvolution2d_2_Wsub_4*
validate_shape(*$
_class
loc:@convolution2d_2_W*
use_locking(*
T0*&
_output_shapes
:
M
mul_21/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Z
mul_21Mulmul_21/xVariable_22/read*
T0*&
_output_shapes
:
J
Square_5Squarediv_4*
T0*&
_output_shapes
:
M
mul_22/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
R
mul_22Mulmul_22/xSquare_5*
T0*&
_output_shapes
:
N
add_18Addmul_21mul_22*
T0*&
_output_shapes
:
°
Assign_8AssignVariable_22add_18*
validate_shape(*
_class
loc:@Variable_22*
use_locking(*
T0*&
_output_shapes
:
M
mul_23/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
M
mul_23Mulmul_23/xVariable_9/read*
T0*
_output_shapes
:
Y
Square_6Square gradients/Reshape_1_grad/Reshape*
T0*
_output_shapes
:
M
mul_24/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
F
mul_24Mulmul_24/xSquare_6*
T0*
_output_shapes
:
B
add_19Addmul_23mul_24*
T0*
_output_shapes
:
У
Assign_9Assign
Variable_9add_19*
validate_shape(*
_class
loc:@Variable_9*
use_locking(*
T0*
_output_shapes
:
M
add_20/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
N
add_20AddVariable_23/readadd_20/y*
T0*
_output_shapes
:
M
Const_53Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_54Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Y
clip_by_value_7/MinimumMinimumadd_20Const_54*
T0*
_output_shapes
:
b
clip_by_value_7Maximumclip_by_value_7/MinimumConst_53*
T0*
_output_shapes
:
D
Sqrt_6Sqrtclip_by_value_7*
T0*
_output_shapes
:
\
mul_25Mul gradients/Reshape_1_grad/ReshapeSqrt_6*
T0*
_output_shapes
:
M
add_21/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
D
add_21Addadd_19add_21/y*
T0*
_output_shapes
:
M
Const_55Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_56Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Y
clip_by_value_8/MinimumMinimumadd_21Const_56*
T0*
_output_shapes
:
b
clip_by_value_8Maximumclip_by_value_8/MinimumConst_55*
T0*
_output_shapes
:
D
Sqrt_7Sqrtclip_by_value_8*
T0*
_output_shapes
:
A
div_5Divmul_25Sqrt_7*
T0*
_output_shapes
:
J
mul_26MulVariable_3/readdiv_5*
T0*
_output_shapes
:
Q
sub_5Subconvolution2d_2_b/readmul_26*
T0*
_output_shapes
:
°
	Assign_10Assignconvolution2d_2_bsub_5*
validate_shape(*$
_class
loc:@convolution2d_2_b*
use_locking(*
T0*
_output_shapes
:
M
mul_27/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_27Mulmul_27/xVariable_23/read*
T0*
_output_shapes
:
>
Square_7Squarediv_5*
T0*
_output_shapes
:
M
mul_28/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
F
mul_28Mulmul_28/xSquare_7*
T0*
_output_shapes
:
B
add_22Addmul_27mul_28*
T0*
_output_shapes
:
Ц
	Assign_11AssignVariable_23add_22*
validate_shape(*
_class
loc:@Variable_23*
use_locking(*
T0*
_output_shapes
:
M
mul_29/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Z
mul_29Mulmul_29/xVariable_10/read*
T0*&
_output_shapes
:
q
Square_8Square,gradients/Conv2D_2_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
M
mul_30/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
R
mul_30Mulmul_30/xSquare_8*
T0*&
_output_shapes
:
N
add_23Addmul_29mul_30*
T0*&
_output_shapes
:
Ґ
	Assign_12AssignVariable_10add_23*
validate_shape(*
_class
loc:@Variable_10*
use_locking(*
T0*&
_output_shapes
:
M
add_24/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
Z
add_24AddVariable_24/readadd_24/y*
T0*&
_output_shapes
:
M
Const_57Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_58Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
e
clip_by_value_9/MinimumMinimumadd_24Const_58*
T0*&
_output_shapes
:
n
clip_by_value_9Maximumclip_by_value_9/MinimumConst_57*
T0*&
_output_shapes
:
P
Sqrt_8Sqrtclip_by_value_9*
T0*&
_output_shapes
:
t
mul_31Mul,gradients/Conv2D_2_grad/Conv2DBackpropFilterSqrt_8*
T0*&
_output_shapes
:
M
add_25/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
P
add_25Addadd_23add_25/y*
T0*&
_output_shapes
:
M
Const_59Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_60Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
f
clip_by_value_10/MinimumMinimumadd_25Const_60*
T0*&
_output_shapes
:
p
clip_by_value_10Maximumclip_by_value_10/MinimumConst_59*
T0*&
_output_shapes
:
Q
Sqrt_9Sqrtclip_by_value_10*
T0*&
_output_shapes
:
M
div_6Divmul_31Sqrt_9*
T0*&
_output_shapes
:
V
mul_32MulVariable_3/readdiv_6*
T0*&
_output_shapes
:
]
sub_6Subconvolution2d_3_W/readmul_32*
T0*&
_output_shapes
:
≠
	Assign_13Assignconvolution2d_3_Wsub_6*
validate_shape(*$
_class
loc:@convolution2d_3_W*
use_locking(*
T0*&
_output_shapes
:
M
mul_33/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Z
mul_33Mulmul_33/xVariable_24/read*
T0*&
_output_shapes
:
J
Square_9Squarediv_6*
T0*&
_output_shapes
:
M
mul_34/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
R
mul_34Mulmul_34/xSquare_9*
T0*&
_output_shapes
:
N
add_26Addmul_33mul_34*
T0*&
_output_shapes
:
Ґ
	Assign_14AssignVariable_24add_26*
validate_shape(*
_class
loc:@Variable_24*
use_locking(*
T0*&
_output_shapes
:
M
mul_35/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_35Mulmul_35/xVariable_11/read*
T0*
_output_shapes
:
Z
	Square_10Square gradients/Reshape_2_grad/Reshape*
T0*
_output_shapes
:
M
mul_36/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
G
mul_36Mulmul_36/x	Square_10*
T0*
_output_shapes
:
B
add_27Addmul_35mul_36*
T0*
_output_shapes
:
Ц
	Assign_15AssignVariable_11add_27*
validate_shape(*
_class
loc:@Variable_11*
use_locking(*
T0*
_output_shapes
:
M
add_28/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
N
add_28AddVariable_25/readadd_28/y*
T0*
_output_shapes
:
M
Const_61Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_62Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Z
clip_by_value_11/MinimumMinimumadd_28Const_62*
T0*
_output_shapes
:
d
clip_by_value_11Maximumclip_by_value_11/MinimumConst_61*
T0*
_output_shapes
:
F
Sqrt_10Sqrtclip_by_value_11*
T0*
_output_shapes
:
]
mul_37Mul gradients/Reshape_2_grad/ReshapeSqrt_10*
T0*
_output_shapes
:
M
add_29/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
D
add_29Addadd_27add_29/y*
T0*
_output_shapes
:
M
Const_63Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_64Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Z
clip_by_value_12/MinimumMinimumadd_29Const_64*
T0*
_output_shapes
:
d
clip_by_value_12Maximumclip_by_value_12/MinimumConst_63*
T0*
_output_shapes
:
F
Sqrt_11Sqrtclip_by_value_12*
T0*
_output_shapes
:
B
div_7Divmul_37Sqrt_11*
T0*
_output_shapes
:
J
mul_38MulVariable_3/readdiv_7*
T0*
_output_shapes
:
Q
sub_7Subconvolution2d_3_b/readmul_38*
T0*
_output_shapes
:
°
	Assign_16Assignconvolution2d_3_bsub_7*
validate_shape(*$
_class
loc:@convolution2d_3_b*
use_locking(*
T0*
_output_shapes
:
M
mul_39/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_39Mulmul_39/xVariable_25/read*
T0*
_output_shapes
:
?
	Square_11Squarediv_7*
T0*
_output_shapes
:
M
mul_40/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
G
mul_40Mulmul_40/x	Square_11*
T0*
_output_shapes
:
B
add_30Addmul_39mul_40*
T0*
_output_shapes
:
Ц
	Assign_17AssignVariable_25add_30*
validate_shape(*
_class
loc:@Variable_25*
use_locking(*
T0*
_output_shapes
:
M
mul_41/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Z
mul_41Mulmul_41/xVariable_12/read*
T0*&
_output_shapes
:
r
	Square_12Square,gradients/Conv2D_3_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
M
mul_42/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
S
mul_42Mulmul_42/x	Square_12*
T0*&
_output_shapes
:
N
add_31Addmul_41mul_42*
T0*&
_output_shapes
:
Ґ
	Assign_18AssignVariable_12add_31*
validate_shape(*
_class
loc:@Variable_12*
use_locking(*
T0*&
_output_shapes
:
M
add_32/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
Z
add_32AddVariable_26/readadd_32/y*
T0*&
_output_shapes
:
M
Const_65Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_66Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
f
clip_by_value_13/MinimumMinimumadd_32Const_66*
T0*&
_output_shapes
:
p
clip_by_value_13Maximumclip_by_value_13/MinimumConst_65*
T0*&
_output_shapes
:
R
Sqrt_12Sqrtclip_by_value_13*
T0*&
_output_shapes
:
u
mul_43Mul,gradients/Conv2D_3_grad/Conv2DBackpropFilterSqrt_12*
T0*&
_output_shapes
:
M
add_33/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
P
add_33Addadd_31add_33/y*
T0*&
_output_shapes
:
M
Const_67Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_68Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
f
clip_by_value_14/MinimumMinimumadd_33Const_68*
T0*&
_output_shapes
:
p
clip_by_value_14Maximumclip_by_value_14/MinimumConst_67*
T0*&
_output_shapes
:
R
Sqrt_13Sqrtclip_by_value_14*
T0*&
_output_shapes
:
N
div_8Divmul_43Sqrt_13*
T0*&
_output_shapes
:
V
mul_44MulVariable_3/readdiv_8*
T0*&
_output_shapes
:
]
sub_8Subconvolution2d_4_W/readmul_44*
T0*&
_output_shapes
:
≠
	Assign_19Assignconvolution2d_4_Wsub_8*
validate_shape(*$
_class
loc:@convolution2d_4_W*
use_locking(*
T0*&
_output_shapes
:
M
mul_45/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Z
mul_45Mulmul_45/xVariable_26/read*
T0*&
_output_shapes
:
K
	Square_13Squarediv_8*
T0*&
_output_shapes
:
M
mul_46/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
S
mul_46Mulmul_46/x	Square_13*
T0*&
_output_shapes
:
N
add_34Addmul_45mul_46*
T0*&
_output_shapes
:
Ґ
	Assign_20AssignVariable_26add_34*
validate_shape(*
_class
loc:@Variable_26*
use_locking(*
T0*&
_output_shapes
:
M
mul_47/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_47Mulmul_47/xVariable_13/read*
T0*
_output_shapes
:
Z
	Square_14Square gradients/Reshape_3_grad/Reshape*
T0*
_output_shapes
:
M
mul_48/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
G
mul_48Mulmul_48/x	Square_14*
T0*
_output_shapes
:
B
add_35Addmul_47mul_48*
T0*
_output_shapes
:
Ц
	Assign_21AssignVariable_13add_35*
validate_shape(*
_class
loc:@Variable_13*
use_locking(*
T0*
_output_shapes
:
M
add_36/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
N
add_36AddVariable_27/readadd_36/y*
T0*
_output_shapes
:
M
Const_69Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_70Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Z
clip_by_value_15/MinimumMinimumadd_36Const_70*
T0*
_output_shapes
:
d
clip_by_value_15Maximumclip_by_value_15/MinimumConst_69*
T0*
_output_shapes
:
F
Sqrt_14Sqrtclip_by_value_15*
T0*
_output_shapes
:
]
mul_49Mul gradients/Reshape_3_grad/ReshapeSqrt_14*
T0*
_output_shapes
:
M
add_37/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
D
add_37Addadd_35add_37/y*
T0*
_output_shapes
:
M
Const_71Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_72Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Z
clip_by_value_16/MinimumMinimumadd_37Const_72*
T0*
_output_shapes
:
d
clip_by_value_16Maximumclip_by_value_16/MinimumConst_71*
T0*
_output_shapes
:
F
Sqrt_15Sqrtclip_by_value_16*
T0*
_output_shapes
:
B
div_9Divmul_49Sqrt_15*
T0*
_output_shapes
:
J
mul_50MulVariable_3/readdiv_9*
T0*
_output_shapes
:
Q
sub_9Subconvolution2d_4_b/readmul_50*
T0*
_output_shapes
:
°
	Assign_22Assignconvolution2d_4_bsub_9*
validate_shape(*$
_class
loc:@convolution2d_4_b*
use_locking(*
T0*
_output_shapes
:
M
mul_51/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_51Mulmul_51/xVariable_27/read*
T0*
_output_shapes
:
?
	Square_15Squarediv_9*
T0*
_output_shapes
:
M
mul_52/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
G
mul_52Mulmul_52/x	Square_15*
T0*
_output_shapes
:
B
add_38Addmul_51mul_52*
T0*
_output_shapes
:
Ц
	Assign_23AssignVariable_27add_38*
validate_shape(*
_class
loc:@Variable_27*
use_locking(*
T0*
_output_shapes
:
M
mul_53/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Z
mul_53Mulmul_53/xVariable_14/read*
T0*&
_output_shapes
:
r
	Square_16Square,gradients/Conv2D_4_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
M
mul_54/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
S
mul_54Mulmul_54/x	Square_16*
T0*&
_output_shapes
:
N
add_39Addmul_53mul_54*
T0*&
_output_shapes
:
Ґ
	Assign_24AssignVariable_14add_39*
validate_shape(*
_class
loc:@Variable_14*
use_locking(*
T0*&
_output_shapes
:
M
add_40/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
Z
add_40AddVariable_28/readadd_40/y*
T0*&
_output_shapes
:
M
Const_73Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_74Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
f
clip_by_value_17/MinimumMinimumadd_40Const_74*
T0*&
_output_shapes
:
p
clip_by_value_17Maximumclip_by_value_17/MinimumConst_73*
T0*&
_output_shapes
:
R
Sqrt_16Sqrtclip_by_value_17*
T0*&
_output_shapes
:
u
mul_55Mul,gradients/Conv2D_4_grad/Conv2DBackpropFilterSqrt_16*
T0*&
_output_shapes
:
M
add_41/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
P
add_41Addadd_39add_41/y*
T0*&
_output_shapes
:
M
Const_75Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_76Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
f
clip_by_value_18/MinimumMinimumadd_41Const_76*
T0*&
_output_shapes
:
p
clip_by_value_18Maximumclip_by_value_18/MinimumConst_75*
T0*&
_output_shapes
:
R
Sqrt_17Sqrtclip_by_value_18*
T0*&
_output_shapes
:
O
div_10Divmul_55Sqrt_17*
T0*&
_output_shapes
:
W
mul_56MulVariable_3/readdiv_10*
T0*&
_output_shapes
:
^
sub_10Subconvolution2d_5_W/readmul_56*
T0*&
_output_shapes
:
Ѓ
	Assign_25Assignconvolution2d_5_Wsub_10*
validate_shape(*$
_class
loc:@convolution2d_5_W*
use_locking(*
T0*&
_output_shapes
:
M
mul_57/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Z
mul_57Mulmul_57/xVariable_28/read*
T0*&
_output_shapes
:
L
	Square_17Squarediv_10*
T0*&
_output_shapes
:
M
mul_58/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
S
mul_58Mulmul_58/x	Square_17*
T0*&
_output_shapes
:
N
add_42Addmul_57mul_58*
T0*&
_output_shapes
:
Ґ
	Assign_26AssignVariable_28add_42*
validate_shape(*
_class
loc:@Variable_28*
use_locking(*
T0*&
_output_shapes
:
M
mul_59/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_59Mulmul_59/xVariable_15/read*
T0*
_output_shapes
:
Z
	Square_18Square gradients/Reshape_4_grad/Reshape*
T0*
_output_shapes
:
M
mul_60/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
G
mul_60Mulmul_60/x	Square_18*
T0*
_output_shapes
:
B
add_43Addmul_59mul_60*
T0*
_output_shapes
:
Ц
	Assign_27AssignVariable_15add_43*
validate_shape(*
_class
loc:@Variable_15*
use_locking(*
T0*
_output_shapes
:
M
add_44/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
N
add_44AddVariable_29/readadd_44/y*
T0*
_output_shapes
:
M
Const_77Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_78Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Z
clip_by_value_19/MinimumMinimumadd_44Const_78*
T0*
_output_shapes
:
d
clip_by_value_19Maximumclip_by_value_19/MinimumConst_77*
T0*
_output_shapes
:
F
Sqrt_18Sqrtclip_by_value_19*
T0*
_output_shapes
:
]
mul_61Mul gradients/Reshape_4_grad/ReshapeSqrt_18*
T0*
_output_shapes
:
M
add_45/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
D
add_45Addadd_43add_45/y*
T0*
_output_shapes
:
M
Const_79Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_80Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Z
clip_by_value_20/MinimumMinimumadd_45Const_80*
T0*
_output_shapes
:
d
clip_by_value_20Maximumclip_by_value_20/MinimumConst_79*
T0*
_output_shapes
:
F
Sqrt_19Sqrtclip_by_value_20*
T0*
_output_shapes
:
C
div_11Divmul_61Sqrt_19*
T0*
_output_shapes
:
K
mul_62MulVariable_3/readdiv_11*
T0*
_output_shapes
:
R
sub_11Subconvolution2d_5_b/readmul_62*
T0*
_output_shapes
:
Ґ
	Assign_28Assignconvolution2d_5_bsub_11*
validate_shape(*$
_class
loc:@convolution2d_5_b*
use_locking(*
T0*
_output_shapes
:
M
mul_63/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_63Mulmul_63/xVariable_29/read*
T0*
_output_shapes
:
@
	Square_19Squarediv_11*
T0*
_output_shapes
:
M
mul_64/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
G
mul_64Mulmul_64/x	Square_19*
T0*
_output_shapes
:
B
add_46Addmul_63mul_64*
T0*
_output_shapes
:
Ц
	Assign_29AssignVariable_29add_46*
validate_shape(*
_class
loc:@Variable_29*
use_locking(*
T0*
_output_shapes
:
M
mul_65/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Z
mul_65Mulmul_65/xVariable_16/read*
T0*&
_output_shapes
:
r
	Square_20Square,gradients/Conv2D_5_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
M
mul_66/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
S
mul_66Mulmul_66/x	Square_20*
T0*&
_output_shapes
:
N
add_47Addmul_65mul_66*
T0*&
_output_shapes
:
Ґ
	Assign_30AssignVariable_16add_47*
validate_shape(*
_class
loc:@Variable_16*
use_locking(*
T0*&
_output_shapes
:
M
add_48/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
Z
add_48AddVariable_30/readadd_48/y*
T0*&
_output_shapes
:
M
Const_81Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_82Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
f
clip_by_value_21/MinimumMinimumadd_48Const_82*
T0*&
_output_shapes
:
p
clip_by_value_21Maximumclip_by_value_21/MinimumConst_81*
T0*&
_output_shapes
:
R
Sqrt_20Sqrtclip_by_value_21*
T0*&
_output_shapes
:
u
mul_67Mul,gradients/Conv2D_5_grad/Conv2DBackpropFilterSqrt_20*
T0*&
_output_shapes
:
M
add_49/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
P
add_49Addadd_47add_49/y*
T0*&
_output_shapes
:
M
Const_83Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_84Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
f
clip_by_value_22/MinimumMinimumadd_49Const_84*
T0*&
_output_shapes
:
p
clip_by_value_22Maximumclip_by_value_22/MinimumConst_83*
T0*&
_output_shapes
:
R
Sqrt_21Sqrtclip_by_value_22*
T0*&
_output_shapes
:
O
div_12Divmul_67Sqrt_21*
T0*&
_output_shapes
:
W
mul_68MulVariable_3/readdiv_12*
T0*&
_output_shapes
:
^
sub_12Subconvolution2d_6_W/readmul_68*
T0*&
_output_shapes
:
Ѓ
	Assign_31Assignconvolution2d_6_Wsub_12*
validate_shape(*$
_class
loc:@convolution2d_6_W*
use_locking(*
T0*&
_output_shapes
:
M
mul_69/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Z
mul_69Mulmul_69/xVariable_30/read*
T0*&
_output_shapes
:
L
	Square_21Squarediv_12*
T0*&
_output_shapes
:
M
mul_70/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
S
mul_70Mulmul_70/x	Square_21*
T0*&
_output_shapes
:
N
add_50Addmul_69mul_70*
T0*&
_output_shapes
:
Ґ
	Assign_32AssignVariable_30add_50*
validate_shape(*
_class
loc:@Variable_30*
use_locking(*
T0*&
_output_shapes
:
M
mul_71/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_71Mulmul_71/xVariable_17/read*
T0*
_output_shapes
:
Z
	Square_22Square gradients/Reshape_5_grad/Reshape*
T0*
_output_shapes
:
M
mul_72/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
G
mul_72Mulmul_72/x	Square_22*
T0*
_output_shapes
:
B
add_51Addmul_71mul_72*
T0*
_output_shapes
:
Ц
	Assign_33AssignVariable_17add_51*
validate_shape(*
_class
loc:@Variable_17*
use_locking(*
T0*
_output_shapes
:
M
add_52/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
N
add_52AddVariable_31/readadd_52/y*
T0*
_output_shapes
:
M
Const_85Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_86Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Z
clip_by_value_23/MinimumMinimumadd_52Const_86*
T0*
_output_shapes
:
d
clip_by_value_23Maximumclip_by_value_23/MinimumConst_85*
T0*
_output_shapes
:
F
Sqrt_22Sqrtclip_by_value_23*
T0*
_output_shapes
:
]
mul_73Mul gradients/Reshape_5_grad/ReshapeSqrt_22*
T0*
_output_shapes
:
M
add_53/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
D
add_53Addadd_51add_53/y*
T0*
_output_shapes
:
M
Const_87Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_88Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Z
clip_by_value_24/MinimumMinimumadd_53Const_88*
T0*
_output_shapes
:
d
clip_by_value_24Maximumclip_by_value_24/MinimumConst_87*
T0*
_output_shapes
:
F
Sqrt_23Sqrtclip_by_value_24*
T0*
_output_shapes
:
C
div_13Divmul_73Sqrt_23*
T0*
_output_shapes
:
K
mul_74MulVariable_3/readdiv_13*
T0*
_output_shapes
:
R
sub_13Subconvolution2d_6_b/readmul_74*
T0*
_output_shapes
:
Ґ
	Assign_34Assignconvolution2d_6_bsub_13*
validate_shape(*$
_class
loc:@convolution2d_6_b*
use_locking(*
T0*
_output_shapes
:
M
mul_75/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_75Mulmul_75/xVariable_31/read*
T0*
_output_shapes
:
@
	Square_23Squarediv_13*
T0*
_output_shapes
:
M
mul_76/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
G
mul_76Mulmul_76/x	Square_23*
T0*
_output_shapes
:
B
add_54Addmul_75mul_76*
T0*
_output_shapes
:
Ц
	Assign_35AssignVariable_31add_54*
validate_shape(*
_class
loc:@Variable_31*
use_locking(*
T0*
_output_shapes
:
M
mul_77/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Z
mul_77Mulmul_77/xVariable_18/read*
T0*&
_output_shapes
:
r
	Square_24Square,gradients/Conv2D_6_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:
M
mul_78/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
S
mul_78Mulmul_78/x	Square_24*
T0*&
_output_shapes
:
N
add_55Addmul_77mul_78*
T0*&
_output_shapes
:
Ґ
	Assign_36AssignVariable_18add_55*
validate_shape(*
_class
loc:@Variable_18*
use_locking(*
T0*&
_output_shapes
:
M
add_56/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
Z
add_56AddVariable_32/readadd_56/y*
T0*&
_output_shapes
:
M
Const_89Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_90Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
f
clip_by_value_25/MinimumMinimumadd_56Const_90*
T0*&
_output_shapes
:
p
clip_by_value_25Maximumclip_by_value_25/MinimumConst_89*
T0*&
_output_shapes
:
R
Sqrt_24Sqrtclip_by_value_25*
T0*&
_output_shapes
:
u
mul_79Mul,gradients/Conv2D_6_grad/Conv2DBackpropFilterSqrt_24*
T0*&
_output_shapes
:
M
add_57/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
P
add_57Addadd_55add_57/y*
T0*&
_output_shapes
:
M
Const_91Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_92Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
f
clip_by_value_26/MinimumMinimumadd_57Const_92*
T0*&
_output_shapes
:
p
clip_by_value_26Maximumclip_by_value_26/MinimumConst_91*
T0*&
_output_shapes
:
R
Sqrt_25Sqrtclip_by_value_26*
T0*&
_output_shapes
:
O
div_14Divmul_79Sqrt_25*
T0*&
_output_shapes
:
W
mul_80MulVariable_3/readdiv_14*
T0*&
_output_shapes
:
^
sub_14Subconvolution2d_7_W/readmul_80*
T0*&
_output_shapes
:
Ѓ
	Assign_37Assignconvolution2d_7_Wsub_14*
validate_shape(*$
_class
loc:@convolution2d_7_W*
use_locking(*
T0*&
_output_shapes
:
M
mul_81/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
Z
mul_81Mulmul_81/xVariable_32/read*
T0*&
_output_shapes
:
L
	Square_25Squarediv_14*
T0*&
_output_shapes
:
M
mul_82/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
S
mul_82Mulmul_82/x	Square_25*
T0*&
_output_shapes
:
N
add_58Addmul_81mul_82*
T0*&
_output_shapes
:
Ґ
	Assign_38AssignVariable_32add_58*
validate_shape(*
_class
loc:@Variable_32*
use_locking(*
T0*&
_output_shapes
:
M
mul_83/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_83Mulmul_83/xVariable_19/read*
T0*
_output_shapes
:
Z
	Square_26Square gradients/Reshape_6_grad/Reshape*
T0*
_output_shapes
:
M
mul_84/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
G
mul_84Mulmul_84/x	Square_26*
T0*
_output_shapes
:
B
add_59Addmul_83mul_84*
T0*
_output_shapes
:
Ц
	Assign_39AssignVariable_19add_59*
validate_shape(*
_class
loc:@Variable_19*
use_locking(*
T0*
_output_shapes
:
M
add_60/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
N
add_60AddVariable_33/readadd_60/y*
T0*
_output_shapes
:
M
Const_93Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_94Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Z
clip_by_value_27/MinimumMinimumadd_60Const_94*
T0*
_output_shapes
:
d
clip_by_value_27Maximumclip_by_value_27/MinimumConst_93*
T0*
_output_shapes
:
F
Sqrt_26Sqrtclip_by_value_27*
T0*
_output_shapes
:
]
mul_85Mul gradients/Reshape_6_grad/ReshapeSqrt_26*
T0*
_output_shapes
:
M
add_61/yConst*
dtype0*
valueB
 *wћ+2*
_output_shapes
: 
D
add_61Addadd_59add_61/y*
T0*
_output_shapes
:
M
Const_95Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_96Const*
dtype0*
valueB
 *  А*
_output_shapes
: 
Z
clip_by_value_28/MinimumMinimumadd_61Const_96*
T0*
_output_shapes
:
d
clip_by_value_28Maximumclip_by_value_28/MinimumConst_95*
T0*
_output_shapes
:
F
Sqrt_27Sqrtclip_by_value_28*
T0*
_output_shapes
:
C
div_15Divmul_85Sqrt_27*
T0*
_output_shapes
:
K
mul_86MulVariable_3/readdiv_15*
T0*
_output_shapes
:
R
sub_15Subconvolution2d_7_b/readmul_86*
T0*
_output_shapes
:
Ґ
	Assign_40Assignconvolution2d_7_bsub_15*
validate_shape(*$
_class
loc:@convolution2d_7_b*
use_locking(*
T0*
_output_shapes
:
M
mul_87/xConst*
dtype0*
valueB
 *33s?*
_output_shapes
: 
N
mul_87Mulmul_87/xVariable_33/read*
T0*
_output_shapes
:
@
	Square_27Squarediv_15*
T0*
_output_shapes
:
M
mul_88/xConst*
dtype0*
valueB
 *ЌћL=*
_output_shapes
: 
G
mul_88Mulmul_88/x	Square_27*
T0*
_output_shapes
:
B
add_62Addmul_87mul_88*
T0*
_output_shapes
:
Ц
	Assign_41AssignVariable_33add_62*
validate_shape(*
_class
loc:@Variable_33*
use_locking(*
T0*
_output_shapes
:
И
group_deps_1NoOp^mul_4^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41"дoт