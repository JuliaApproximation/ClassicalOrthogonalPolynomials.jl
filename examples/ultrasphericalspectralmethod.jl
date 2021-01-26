using ContinuumArrays, FillArrays, InfiniteArrays, Plots

T = Chebyshev()
C = Ultraspherical(2)
D = Derivative(axes(T,1))
A = (C\(D^2*T))+100(C\T)

n = 100
c = [T[[-1,1],1:n]; A[1:n-2,1:n]] \ [1;2;zeros(n-2)]
u = T*Vcat(c,Zeros(∞))

xx = range(-1,1;length=1000)
plot(xx,u[xx])

