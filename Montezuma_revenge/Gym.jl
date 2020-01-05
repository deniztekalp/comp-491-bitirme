using Pkg

if !haskey(Pkg.installed(),"PyCall")
        Pkg.add("PyCall")
end

using PyCall

const gym = PyNULL()

include("spaces.jl")


struct Spec
    id
    reward_threshold
    nondeterministic
    tags
    max_episode_steps
end

mutable struct GymEnv
    name
    spec
    action_space
    observation_space
    reward_range
    gymenv
end

function GymEnv(id::String)
    copy!(gym, pyimport("gym"))
    gymenv = nothing
    try
        gymenv = gym.make(id)
    catch e
        error("Error received during the initialization of $id\n$e")
    end
    
    spec = Spec(gymenv.spec.id,
                gymenv.spec.reward_threshold,
                gymenv.spec.nondeterministic,
                gymenv.spec.tags,
                gymenv.spec.max_episode_steps,
               )
    action_space = julia_space(gymenv.action_space)
    observation_space = julia_space(gymenv.observation_space)

    env = GymEnv(id, spec, action_space,
                 observation_space, gymenv.reward_range, gymenv)
    return env
end

reset!(env::GymEnv) = env.gymenv.reset()
function render(env::GymEnv; mode="human")
    env.gymenv.render(mode)
end

function step!(env::GymEnv, action)
    ob, reward, done, information = env.gymenv.step(action)
    return ob, reward, done, information
end

close!(env::GymEnv) = env.gymenv.close()

seed!(env::GymEnv, seed=nothing) = env.gymenv.seed(seed)
