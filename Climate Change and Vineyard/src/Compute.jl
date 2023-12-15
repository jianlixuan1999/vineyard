# encode "short" syntax for evaluting utility functions -

(model::VLCobbDouglasUtilityFunction)(x::Vector{Float64}) = evaluate(model, x); # cobb douglas 
(model::VLLinearUtilityFunction)(x::Vector{Float64}) = evaluate(model, x); # linear
(model::VLLogTransformedCobbDouglasUtilityFunction)(x::Vector{Float64}) = evaluate(model, x); # log transformed cobb douglas


# compute the return matrix -
function log_return_matrix(dataset::Dict{Int64,DataFrame}, 
    firms::Array{Int64,1}; Δt::Float64 = (1.0/252.0))::Array{Float64,2}

    # initialize -
    number_of_firms = length(firms);
    number_of_trading_days = nrow(dataset[1]);
    return_matrix = Array{Float64,2}(undef, number_of_trading_days-1, number_of_firms);

    # main loop -
    for i ∈ eachindex(firms) 

        # get the firm data -
        firm_index = firms[i];
        firm_data = dataset[firm_index];

        # compute the log returns -
        for j ∈ 2:number_of_trading_days
            S₁ = firm_data[j-1, :volume_weighted_average_price];
            S₂ = firm_data[j, :volume_weighted_average_price];
            return_matrix[j-1, i] = (1/Δt)*log(S₂/S₁);
        end
    end

    # return -
    return return_matrix;
end

function fractional_return_matrix(dataset::Dict{Int64,DataFrame}, 
    firms::Array{Int64,1}; scale::Float64 = 100.0)::Array{Float64,2}

    # initialize -
    number_of_firms = length(firms);
    number_of_trading_days = nrow(dataset[1]);
    return_matrix = Array{Float64,2}(undef, number_of_trading_days-1, number_of_firms);

    # main loop -
    for i ∈ eachindex(firms) 

        # get the firm data -
        firm_index = firms[i];
        firm_data = dataset[firm_index];

        # compute the log returns -
        for j ∈ 2:number_of_trading_days
            S₁ = firm_data[j-1, :volume_weighted_average_price];
            S₂ = firm_data[j, :volume_weighted_average_price];
            return_matrix[j-1, i] = ((S₂-S₁)/S₁)*scale;
        end
    end

    # return -
    return return_matrix;
end
