function indifference(problem::MySimpleCobbDouglasChoiceProblem, U::Float64, xlim::Array{Float64,1})::Array{Float64,2}

    # initialize -
    α = problem.α
    α₁ = α[1]
    α₂ = α[2]

    # set values for the good and service 1
    X1 = range(xlim[1], stop=xlim[2], step = 0.001) |> collect;
    d = length(X1);

    Y = Array{Float64,2}(undef,d,2);
    for j ∈ 1:d

        tmp = (1/α₂)*(log(U) - α₁*log(X1[j]));

        Y[j,1] = X1[j];
        Y[j,2] = exp(tmp);
    end

    # return -
    return Y;
end

function indifference(problem::MySimpleLinearChoiceProblem, U::Float64, xlim::Array{Float64,2})::Array{Float64,2}

    # initialize -
    α = problem.α
   
    # Use the VLDecisionsPackage to compute the indifference curve -
    model = build(VLLinearUtilityFunction, (
        α = problem.α,
    ));
    tmp = VLDecisionsPackage.indifference(model; utility = U, bounds = xlim, ϵ = 0.01);

    # return array -
    return tmp;
end