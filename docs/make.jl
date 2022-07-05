using Documenter, ClassicalOrthogonalPolynomials

makedocs(
    modules = [ClassicalOrthogonalPolynomials],
    sitename="ClassicalOrthogonalPolynomials.jl",
    pages = Any[
        "Home" => "index.md"])

deploydocs(
    repo = "github.com/JuliaApproximation/ClassicalOrthogonalPolynomials.jl.git",
)
