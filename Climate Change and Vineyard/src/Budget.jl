function budget(problem::T, xlim::Array{Float64,1})::Array{Float64,2} where {T <: AbstractSimpleChoiceProblem}

    # initialize -
    c = problem.c;
    I = problem.I;

    # set values for the good and service 1
    X1 = range(xlim[1], stop=xlim[2], step = 0.001) |> collect;
    d = length(X1);

    Y = Array{Float64,2}(undef,d,2);
    for j ∈ 1:d

        tmp = (1/c[2])*(I - c[1]*X1[j]);

        Y[j,1] = X1[j];
        Y[j,2] = tmp
    end

    # return -
    return Y;
end