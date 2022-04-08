# Forward-mode Automatic Differentiation

using ForwardDiff
using StaticArrays
using BenchmarkTools

#=
 derivative (f, x) where x is a real number
 returns y = df/dx evaluated at x where y is a real number
=#
f1(x) = x^3 + x*sin(x) - x^2 
ForwardDiff.derivative(f1, 2.0) # = 8.077

#=
 gradient (f, x) where x is a vector of real numbers
 f(x) returns a real number
 returns y = ∇fₓ(x) = [δf/δx[1], δf/δx[2], ...] evaluated at x where y has the same size as x
=#
f2(x) = x[1]^3 + x[2]*sin(x[2])
ForwardDiff.gradient(f2, [2.0, 3.0]) # = [12.0, -2.829]

#=
 jacobian (f, x) where x is a vector of real numbers
 f(x) returns a vector of real numbers
 returns J(f) which has a shape of length(f(x)) x length(x)
 t x where y has the same size as x
=#
f3(x) = collect(x.^3 + 2*x) #explicitly makes ForwardDiff know that it is an array
a = @SVector [2.0, 3.0] 
ForwardDiff.jacobian(f3, a)
ForwardDiff.jacobian(x->f3(x), a) # also works. No need to wrap f3 return value with `collect`

#=
jacobian-vector product (also known as the pushforward operation in differential geometry)
jvp(f, x, u) where x and u are vectors of real numbers 
f(x) returns a vector of real numbers 
returns Jₓf(x) * u
=#
function jvp(f, x, u)
    return ForwardDiff.derivative(t->f(x + t*u), 0.0)
end

f(x) = [x[1], x[1]*x[3], x[2]^2]
x = @SVector [3.0, 4.0, 5.0]
u = @SVector [1.0, 2.0, 3.0]
Jvp = ForwardDiff.jacobian(a->f(a), x) * u
Jvp = jvp(f, x, u)