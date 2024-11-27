using Documenter, ClassicalOrthogonalPolynomials

DocMeta.setdocmeta!(ClassicalOrthogonalPolynomials,
    :DocTestSetup,
    :(using ClassicalOrthogonalPolynomials))

makedocs(
    modules = [ClassicalOrthogonalPolynomials],
    sitename="ClassicalOrthogonalPolynomials.jl",
    pages = Any[
        "Home" => "index.md"])

deploydocs(
    repo = "github.com/JuliaApproximation/ClassicalOrthogonalPolynomials.jl.git",
    push_preview = true
)
