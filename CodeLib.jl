abstract type AbstractLearningModel end
abstract type AbstractWorldModel end

"""
    MyQLearningModel

Holds data for the Q learning problem
"""
mutable struct MyQLearningAgentModel <: AbstractLearningModel

    # data -
    states::Array{Tuple{Int,Int},1}
    actions::Array{Int,1}
    γ::Float64
    α::Float64 
    Q::Array{Float64,2}

    # constructor
    MyQLearningAgentModel() = new();
end

mutable struct MyRectangularGridWorldModel <: AbstractWorldModel

    # data -
    number_of_rows::Int
    number_of_cols::Int
    coordinates::Dict{Int,Tuple{Int,Int}}
    states::Dict{Tuple{Int,Int}, Int}
    moves::Dict{Int,Tuple{Int,Int}}
    rewards::Dict{Tuple{Tuple{Int, Int}, Int}, Float64}

    # constructor -
    MyRectangularGridWorldModel() = new();
end

"""
    build(type::Type{MyQLearningModel},data::NamedTuple) -> MyQLearningModel
"""
function build(modeltype::Type{MyQLearningAgentModel}, data::NamedTuple)::MyQLearningAgentModel

    # initialize -
    model = MyQLearningAgentModel();

    # if we have options, add them to the contract model -
    if (isempty(data) == false)
    
        for key ∈ fieldnames(modeltype)
            
            # convert the field_name_symbol to a string -
            field_name_string = string(key)

            # check the for the key -
            if (haskey(data, key) == false)
                throw(ArgumentError("NamedTuple is missing: $(field_name_string)"))
            end

            # get the value -
            value = data[key]

            # set -
            setproperty!(model, key, value)
        end
    end

    # return -
    return model
end

"""
    build(type::MyRectangularGridWorldModel, nrows::Int, ncols::Int, 
        rewards::Dict{Tuple{Int,Int}, Float64}; defaultreward::Float64 = -1.0) -> MyRectangularGridWorldModel
"""


function build(modeltype::Type{MyRectangularGridWorldModel}, data::NamedTuple)::MyRectangularGridWorldModel

    # initialize and empty model -
    model = MyRectangularGridWorldModel()

    # get the data -
    nrows = data[:nrows]
    ncols = data[:ncols]
    rewards = data[:rewards]
    defaultreward = haskey(data, :defaultreward) == false ? -1.0 : data[:defaultreward]

    # setup storage
    rewards_dict = Dict{Tuple{Tuple{Int, Int}, Int}, Float64}()
    coordinates = Dict{Int,Tuple{Int,Int}}()
    states = Dict{Tuple{Int,Int},Int}()
    moves = Dict{Int,Tuple{Int,Int}}()

    # build all the stuff 
    position_index = 1;
    for i ∈ 0:(nrows - 1)
        for j ∈ 0:(ncols - 1)
            
            # capture this corrdinate 
            coordinate = (i,j)
            coordinates[position_index] = coordinate
            states[coordinate] = position_index
            if i <= j # min temp less than max temp
                # populate rewards for each action at this coordinate
                for action in data[:actions]
                    reward_key = (coordinate, action)
                    rewards_dict[reward_key] = rewards[reward_key]
                end

                # update position_index -
                position_index += 1;
            end
        end
    end

    # setup the moves dictionary -
    moves[1] = (-1,-1)
    moves[2] = (-9,-9)  
    moves[3] = (-3, -3)  
    moves[4] = (-2, -2)  
    moves[5] = (-3, -3)
    moves[6] = (optimal_temp_min-current_temp_min, optimal_temp_max-current_temp_max)

    # add items to the model -
    model.rewards = rewards_dict
    model.coordinates = coordinates
    model.states = states;
    model.moves = moves;
    model.number_of_rows = nrows
    model.number_of_cols = ncols

    # return -
    return model
end


function policy(Q_array::Array{Float64,2})::Array{Int64,1}

    # get the dimension -
    (NR, _) = size(Q_array);

    # initialize some storage -
    π_array = Array{Int64,1}(undef, NR)
    for s ∈ 1:NR
        π_array[s] = argmax(Q_array[s,:]);
    end

    # return -
    return π_array;
end

# PRIVATE METHODS BELOW HERE ================================================================================= #


"""
    _update!(model::MyQLearningModel, data::NamedTuple) -> MyQLearninAgentModel
"""
function _world(model::MyRectangularGridWorldModel, s::Int, a::Int)::Tuple{Int,Float64}

    # initialize -
    s′ = nothing
    r = nothing
    
    # get data from the model -
    coordinates = model.coordinates;
    moves = model.moves
    states = model.states;
    rewards = model.rewards;

    # where are we now?
    current_position = coordinates[s];

    # get the perturbation -
    Δ = moves[a];
    new_position = current_position .+ Δ

    # before we go on, have we "driven off the grid"?
    if (haskey(states, new_position) == true)

        # lookup the new state -
        s′ = states[new_position];
        r = rewards[current_position, a];
    else
        # ok: so we are all the grid. Bounce us back to to the current_position, and charge a huge penalty 
        s′ = states[current_position];
        r = -1000000000000.0
    end

    # return -
    return (s′,r);
end

"""
    _update!(model::MyQLearningModel, data::NamedTuple) -> MyQLearningAgentModel
"""
function _update(model::MyQLearningAgentModel, data::NamedTuple)::MyQLearningAgentModel

    # grab the s,a,reward and next state from the data tuple
    s = data[:s];
    a = data[:a];
    r = data[:r];
    s′ = data[:s′];
    

    # grab parameters from the model -
    γ, Q, α = model.γ, model.Q, model.α

    # use the update rule to update Q -
    Q[s,a] += α*(r+γ*maximum(Q[s′,:]) - Q[s,a])

    # return -
    return model;
end
# PRIVATE METHODS ABOVE HERE ================================================================================= #

# PUBLIC METHODS BELOW HERE ================================================================================== #

# Cool hack: What is going on with these?
(model::MyRectangularGridWorldModel)(s::Int,a::Int) = _world(model, s, a);
(model::MyQLearningAgentModel)(data::NamedTuple) = _update(model, data);

"""
    simulate(model::MyQLearningModel, environment::T, startstate::Int, maxsteps::Int;
        ϵ::Float64 = 0.2) -> MyQLearningModel where T <: AbstractWorldModel
"""
taken_actions = Set{Int}()

function simulate(agent::MyQLearningAgentModel, environment::T, startstate::Tuple{Int,Int}, maxsteps::Int;
    ϵ::Float64 = 0.2)::MyQLearningAgentModel where T <: AbstractWorldModel

    # initialize -   
    s = environment.states[startstate]
    actions = agent.actions;
    number_of_actions = length(actions);
    
    taken_actions = Set{Int}()

    # simulation loop -
    for _ ∈ 1:maxsteps
        a = nothing;
        # available_actions = setdiff(actions, taken_actions);
        # println("available actions:", available_actions)
        
        if rand() < ϵ
            # exploration
            a = rand(1:number_of_actions)
            # a = isempty(available_actions) ? -1 : rand(available_actions)
            # println("random action chosen:", a)
        else
            # exploitation
            Q = agent.Q;
            
#             # Ensure that s is a valid state index
#             if s in 1:size(Q, 1)
#                 # Filter Q-values for available actions
#                 filtered_Q_values = Q[s, available_actions]

#                 # Find the index of the best action in the filtered list
#                 best_action_index = argmax(filtered_Q_values)

#                 # Map the index back to the original action set
#                 a = available_actions[best_action_index]
#             else
#                 # Handle the case where s is not a valid state index
#                 # set a default action
#                 a = 6
#             end
            a = argmax(Q[s,:]);
            # best_action = -1
            # best_value = -Inf
            # for a in available_actions
            #     if s in 1:size(agent_model.Q, 1) && a in 1:size(agent_model.Q, 2)
            #         if Q[s, a] > best_value
            #             best_value = Q[s, a]
            #             best_action = a
            #         end
            #     else
            #         println("skipping invalid index - state:", s, "action:", a)
            #     end
            # end
            # a = best_action
        end
        
        
#         # if no available action, break the loop
#         if a == -1
#             break
#         end
        
#         # update the set of taken actions
#         push!(taken_actions, a)
            
        # compute the next state and reward
        s′, r = nothing, nothing;
        current_position = environment.coordinates[s];
        new_position = current_position .+ environment.moves[a]
        if (haskey(environment.states, new_position) == true)
            # ask the world, what is my next state and reward from this (s,a)
            (s′,r) = environment(s,a)
        else
            s′ = s;
            r = -1000000000000.0;
        end
        
        
        # update the agent model
        agent = agent((
            s = s, a = a, r = r, s′ = s′
        ));

        # move to the next state
        s = s′;
    end

    # return -
    return agent
end
# PUBLIC METHODS ABOVE HERE ================================================================================== #
