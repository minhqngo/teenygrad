import math
from typing import Tuple, Optional, cast

from teenygrad.helpers import argsort, DType
from teenygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from teenygrad.tensor import Function
from teenygrad.lazy import LazyBuffer
from teenygrad.shape.symbolic import sint


class Contiguous(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        return x.contiguous()

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output


class ContiguousBackward(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        return x

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.contiguous()


class Cast(Function):
    def forward(self, x: LazyBuffer, dtype: DType, bitcast: bool = False) -> LazyBuffer:
        self.input_dtype, self.bitcast = x.dtype, bitcast
        return x.cast(dtype, bitcast)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.cast(self.input_dtype, self.bitcast)

# ************* unary ops *************

class Zero(Function):
    def forward(self, x:LazyBuffer) -> LazyBuffer:
        return x.const(0)

    def backward(self, grad:LazyBuffer) -> LazyBuffer:
        return grad.const(0)

class Neg(Function):
    def forward(self, x:LazyBuffer) -> LazyBuffer:
        return x.e(UnaryOps.NEG)

    def backward(self, grad:LazyBuffer) -> LazyBuffer:
        return grad.e(UnaryOps.NEG)

class Sin(Function):
    def forward(self, x:LazyBuffer) -> LazyBuffer:
        self.x = x
        return x.e(UnaryOps.SIN)

    def backward(self, grad:LazyBuffer) -> LazyBuffer:
        half_pi = self.x.const(math.pi / 2)
        cos_x = half_pi.e(BinaryOps.SUB, self.x).e(UnaryOps.SIN)
        return cos_x.e(BinaryOps.MUL, grad)

# NOTE: maximum(x, 0) behaves differently where x=0

class Relu(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.e(BinaryOps.MAX, x.const(0))
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return self.ret.const(0).e(BinaryOps.CMPLT, self.ret).e(BinaryOps.MUL, grad_output)


class Log(Function):
    def forward(self, x:LazyBuffer) -> LazyBuffer:
        self.x = x
        log2_x = x.e(UnaryOps.LOG2)
        ln_2 = x.const(math.log(2))
        return log2_x.e(BinaryOps.MUL, ln_2)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.e(BinaryOps.DIV, self.x)


class Exp(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        inv_ln_2 = x.const(1 / math.log(2))
        x_scaled = x.e(BinaryOps.MUL, inv_ln_2)
        self.ret = x_scaled.e(UnaryOps.EXP2)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return self.ret.e(BinaryOps.MUL, grad_output)


class Sqrt(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.e(BinaryOps.DIV, self.ret.e(BinaryOps.MUL, self.ret.const(2)))


# NOTE: the implicit derivative of sigmoid is not stable
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
# TODO: have the backend automatically find this
class Sigmoid(Function):
    def forward(self, x:LazyBuffer) -> LazyBuffer:
        scale = x.const(-1 / math.log(2))
        scaled = x.e(BinaryOps.MUL, scale)
        exp_val = scaled.e(UnaryOps.EXP2)
        denom = x.const(1).e(BinaryOps.ADD, exp_val)
        self.ret = x.const(1).e(BinaryOps.DIV, denom)
        return self.ret

    def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
        one_minus = self.ret.const(1).e(BinaryOps.SUB, self.ret)
        deriv = self.ret.e(BinaryOps.MUL, one_minus)
        return deriv.e(BinaryOps.MUL, grad_output)

# ************* binary ops *************

class Less(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.CMPLT, y)


class Add(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.ADD, y)

    def backward(self, grad_output: LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        if self.needs_input_grad[0]:
            grad_x = grad_output
        else:
            grad_x = None
        if self.needs_input_grad[1]:
            grad_y = grad_output
        else:
            grad_y = None
        return grad_x, grad_y


class Sub(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.SUB, y)

    def backward(self, grad_output: LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        if self.needs_input_grad[0]:
            grad_x = grad_output
        else:
            grad_x = None
        if self.needs_input_grad[1]:
            grad_y = grad_output.e(UnaryOps.NEG)
        else:
            grad_y = None
        return grad_x, grad_y


class Mul(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        self.x, self.y = x, y
        return x.e(BinaryOps.MUL, y)

    def backward(self, grad_output: LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        if self.needs_input_grad[0]:
            grad_x = self.y.e(BinaryOps.MUL, grad_output)
        else:
            grad_x = None
        if self.needs_input_grad[1]:
            grad_y = self.x.e(BinaryOps.MUL, grad_output)
        else:
            grad_y = None
        return grad_x, grad_y


class Div(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        self.x, self.y = x, y
        return x.e(BinaryOps.DIV, y)

    def backward(self, grad_output: LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
        if self.needs_input_grad[0]:
            grad_x = grad_output.e(BinaryOps.DIV, self.y)
        else:
            grad_x = None

        if self.needs_input_grad[1]:
            neg_grad = grad_output.e(UnaryOps.NEG)
            numerator = neg_grad.e(BinaryOps.MUL, self.x)
            y_squared = self.y.e(BinaryOps.MUL, self.y)
            grad_y = numerator.e(BinaryOps.DIV, y_squared)
        else:
            grad_y = None

        return grad_x, grad_y

# ************* ternary ops *************

class Where(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer, z: LazyBuffer) -> LazyBuffer:
        self.x = x
        return x.e(TernaryOps.WHERE, y, z)

    def backward(self, grad_output: LazyBuffer) -> Tuple[None, Optional[LazyBuffer], Optional[LazyBuffer]]:
        if self.needs_input_grad[1]:
            grad_y = self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0))
        else:
            grad_y = None
        if self.needs_input_grad[2]:
            grad_z = self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output)
        else:
            grad_z = None
        return None, grad_y, grad_z

# ************* reduce ops *************

class Sum(Function):
    def forward(self, x: LazyBuffer, new_shape: Tuple[int, ...]) -> LazyBuffer:
        self.input_shape = x.shape
        return x.r(ReduceOps.SUM, new_shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.expand(self.input_shape)


class Max(Function):
    def forward(self, x: LazyBuffer, new_shape: Tuple[int, ...]) -> LazyBuffer:
        self.x, self.ret = x, x.r(ReduceOps.MAX, new_shape)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        # 1s in locations where the max was chosen (can be two locations)
        expanded = self.ret.expand(self.x.shape)
        is_less = self.x.e(BinaryOps.CMPLT, expanded)
        max_mask = self.x.const(1.0).e(BinaryOps.SUB, is_less)

        count = max_mask.r(ReduceOps.SUM, grad_output.shape)
        div = count.expand(self.x.shape)

        grad_expanded = grad_output.expand(self.x.shape)
        return max_mask.e(BinaryOps.DIV, div).e(BinaryOps.MUL, grad_expanded)

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
    def forward(self, x: LazyBuffer, shape: Tuple[int, ...]) -> LazyBuffer:
        self.input_shape = x.shape
        return x.expand(shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.r(ReduceOps.SUM, self.input_shape)


class Reshape(Function):
    def forward(self, x: LazyBuffer, shape: Tuple[int, ...]) -> LazyBuffer:
        self.input_shape = x.shape
        return x.reshape(shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.reshape(self.input_shape)


class Permute(Function):
    def forward(self, x: LazyBuffer, order: Tuple[int, ...]) -> LazyBuffer:
        self.input_order = order
        return x.permute(order)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.permute(argsort(self.input_order))


class Pad(Function):
    def forward(self, x: LazyBuffer, arg: Tuple[Tuple[int, int], ...]) -> LazyBuffer:
        self.narg = tuple([(p[0], s + p[0]) for s, p in zip(x.shape, arg)])
        return x.pad(arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.shrink(self.narg)


class Shrink(Function):
    def forward(self, x: LazyBuffer, arg: Tuple[Tuple[sint, sint], ...]) -> LazyBuffer:
        self.narg = tuple([(p[0], s - p[1]) for s, p in zip(x.shape, arg)])
        return x.shrink(arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        assert all(isinstance(x[0], int) and isinstance(x[1], int) for x in self.narg), "symbolic shrink does not support backward"
        # need this cast because mypy cannot narrow the type even with assert
        return grad_output.pad(cast(Tuple[Tuple[int, int], ...], self.narg))


class Flip(Function):
    def forward(self, x: LazyBuffer, axis: Tuple[int, ...]) -> LazyBuffer:
        axis_set = set(axis)
        self.arg = tuple([-1 if i in axis_set else 1 for i in range(len(x.shape))])
        return x.stride(self.arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.stride(self.arg)
