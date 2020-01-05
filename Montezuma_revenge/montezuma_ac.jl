using Knet

include("actor_critic_montezuma-v1.jl")

w = main(seed=27, episodes=1000, lr=1e-2, render=false)

Knet.save("w_montezuma_v1.jld2", "model", w)


