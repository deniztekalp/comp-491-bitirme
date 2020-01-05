 using Pkg
 for p in ("Knet", "AutoGrad", "Random", "Statistics", "Images", "Colors")
    if !haskey(Pkg.installed(),p)
        Pkg.add(p)
    end
end



using Knet, AutoGrad, Random, Statistics, Images, Colors

include("Bimodal_Embedding1.jl")
include("Gym.jl")

const F = Float32

mutable struct History
    xsize
    nA::Int
    γ::F
    states  # flattens all states into a vector
    actions::Vector{Int}
    rewards::Vector{F}
end

History(xsize, nA, γ, atype) = History(xsize, nA, γ, 
                                convert(atype, zeros(F,0)), 
                                zeros(Int, 0), 
                                zeros(F,0))

function Base.push!(history, s, a, r)
    history.states = [history.states; s]
    push!(history.actions, a)
    push!(history.rewards, r)
end

function preprocess(I, atype)
    I = I[36:195,:,:]
    I = I[1:2:end, 1:2:end, 1]
    I[I .== 144] .= 0
    I[I .== 109] .= 0
    I[I .!= 0] .= 1
    return convert(atype, vec(I))
end

function discount(rewards, γ)
    R = similar(rewards)
    R[end] = rewards[end]
    for k = length(rewards)-1:-1:1
        R[k] = γ * R[k+1] + rewards[k]
    end
    # return R
    return (R .- mean(R)) ./ (std(R) + F(1e-10)) #speeds up training a lot
end

function findLoc(state, es)
    pink = [0xe4, 0x6f, 0x6f]
    for i = 1:210
        for j = 1:160
            if(pink == state[i,j,:])
                return (i,j)
            end
        end
    end
    return es
end

function initweights(atype, xsize, ysize)
    w = []
    ch1 = 4
    ch2 = 8
    nh = 64
    push!(w, xavier(8, 8, xsize[3], ch1))
    push!(w, zeros(1, 1, ch1, 1))
    push!(w, xavier(4, 4, ch1, ch2))
    push!(w, zeros(1, 1, ch2, 1))
    
    push!(w, xavier(nh, 9*9*ch2))
    push!(w, zeros(nh, 1))

    push!(w, xavier(ysize, nh)) # policy
    push!(w, zeros(ysize, 1))
    push!(w, xavier(1, nh)) # value
    push!(w, zeros(1, 1))
    
    return map(wi->convert(atype,wi), w)
end

function predict(w, x)
    x = reshape(x, 80, 80, 1, :)
    x = relu.(conv4(w[1], x, stride=4, padding=2) .+ w[2])
    x = relu.(conv4(w[3], x, stride=2) .+ w[4])
    x = mat(x)
    x = relu.(w[5]*x .+ w[6])
    prob_act = w[7] * x .+ w[8]
    value = w[9] * x .+ w[10]
    return prob_act, value
end

function sample_action(probs)
    probs = Array(probs)
    cprobs = cumsum(probs, dims=1)
    sampled = cprobs .> rand() 
    actions = mapslices(argmax, sampled, dims=1)
    return actions[1]
end

function loss(w, history)
    xsize, nA = history.xsize, history.nA
    nS = prod(xsize)
    M = length(history.states) ÷ nS
    states = reshape(history.states, nS, M)

    p, V = predict(w, states)
    V = vec(V)   

    R = discount(history.rewards, history.γ)
    R = convert(typeof(value(V)), R)
    A = R .- V   # advantage  

    inds = history.actions + nA*(0:M-1)
    lp = logsoftmax(p, dims=1)[inds] # lp is a vector

    return -mean(lp .* value(A)) + L2Reg(A)
end

function preprocessVisLang(f1)
    f1 = Float32.(f1)
    reshape(Float32.(Gray.(colorview(RGB, permutedims(f1, (3,1,2))))), (210,160,1))
end

L2Reg(x) = mean(x .* x)

function main(;
        lr = 1e-2, # learning rate
        γ = 0.99, # discount rate
        episodes = 10000,  # max episodes played
        render = false, # if true display an episode every infotime ones
        seed = 27,
        infotime = 5000,
        atype = gpu() >= 0 ? KnetArray{F} : Array{F},
    )


    env = GymEnv("MontezumaRevenge-v0")
    seed > 0 && (Random.seed!(seed); seed!(env, seed))
    xsize = (80, 80, 1) # input dimensions of the policy network
    nA = 8 # number of actions.
    amap = Dict(1=>0, 2=>1, 3=>2, 4=>3, 5=>4, 6=>5, 7=>11, 8=>12) 

    #w = initweights(atype, xsize, nA)
    w = Knet.load("w_montezuma_vdiff.jld2", "model")
    opt = [Rmsprop(lr=lr) for _=1:length(w)]
    
    
    wdict = Knet.load("dictionary.jld2", "dict")
    UNK = 1
    w2i(x) = get(wdict, x, UNK)
    b = Knet.load("bimodal1_diff.jld2", "model")
    
    total_no_of_commands = 25
    
    sentence1 = w2i.(split("Climb down the ladder Climb down the ladder"))
    sentence2 = w2i.(split("Climb down the ladder Climb down the ladder"))
    sentence3 = w2i.(split("Go to the right side of the room"))
    sentence4 = w2i.(split("Jump to the rope Jump to the rope"))
    sentence5 = w2i.(split("Climb down the ladder Climb down the ladder"))
    sentence6 = w2i.(split("Climb down the ladder Climb down the ladder"))
    sentence6 = w2i.(split("Go to the center of the room center"))
    sentence7 = w2i.(split("Go to the center of the room center"))
    sentence8 = w2i.(split("Go to the center of the room center"))
    sentence9 = w2i.(split("Go to the left side of the room "))
    sentence10 = w2i.(split("Go to the left side of the room "))
    sentence11 = w2i.(split("Climb up the ladder Climb up the ladder"))
    sentence12 = w2i.(split("Climb up the ladder Climb up the ladder"))
    sentence13 = w2i.(split("Get the key Get the key Get key"))
    sentence14 = w2i.(split("Climb down the ladder Climb down the ladder"))
    sentence15 = w2i.(split("Climb down the ladder Climb down the ladder"))
    sentence16 = w2i.(split("Go to the center of the room center"))
    sentence17 = w2i.(split("Go to the center of the room center"))
    sentence18 = w2i.(split("Go to the right side of the room"))
    sentence19 = w2i.(split("Go to the right side of the room"))
    sentence20 = w2i.(split("Climb up the ladder Climb up the ladder"))
    sentence21 = w2i.(split("Climb up the ladder Climb up the ladder"))
    sentence22 = w2i.(split("Jump to the rope Jump to the rope"))
    sentence23 = w2i.(split("Jump to the rope Jump to the rope"))
    sentence24 = w2i.(split("Go to the left side of the room"))
    sentence25 = w2i.(split("Go to the left side of the room"))
    
    dont_do_sentence = w2i.(split("Jump in place Jump in place Jump place"))
    dont_do_sentence2 = w2i.(split("Fall down Fall down Fall down Fall down"))
    
    
    sentences = []
    push!(sentences, sentence1, sentence2, sentence3, sentence4, sentence5, sentence6, sentence7, sentence8, sentence9, sentence10, sentence24, sentence25, sentence11, sentence12, sentence13, sentence14, sentence15, sentence16, sentence17, sentence18, sentence19, sentence20, sentence21, sentence22, sentence23)
    avgreward = 0
    synthetic_reward = 100
    stay_alive_penalty = -1
    death_penalty = 0
    
    
    for episode=1:episodes
        
        lives = 6
        state = reset!(env)
        loc = findLoc(state, (79,79))
            
        MA20loc = (79,79)
        MA20hist = [(loc[1], loc[2])]
        MA20hist = repeat(MA20hist, 20)
        episode_reward = 0
        history = History(xsize, nA, γ, atype)
        vislang_prev = convert(atype, zeros(210, 160, 1))
        prev_s = convert(atype, zeros(prod(xsize)))
        curr_command = 1
        sentence = sentences[curr_command]
        areCommandsAvailable = true
        fell_down_count = 0
        time_step_count = 0
        fall_down_penalty = -3
       
        while time_step_count < 400
            sentence = sentences[curr_command]
            vislang_curr = convert(atype, preprocessVisLang(state))
            
            vislang_history = vislang_curr - vislang_prev
            vislang_prev = vislang_curr
            
            
            isComplete = b((vislang_history, sentence))
            
            
            
            curr_s = preprocess(state, atype)
            s = curr_s .- prev_s
            prev_s = curr_s

            p, V = predict(w, s)
            p = softmax(p, dims=1)
            action = sample_action(p)
             
            state, reward, done, info = step!(env, amap[action])
            
            if(reward>0)
                reward += 10000
                done = true
            end
            
            curr_lives = info["ale.lives"]
            if(lives>curr_lives)
                lives = curr_lives
                curr_command = 1
            end
               
            if (isComplete == 1 && areCommandsAvailable)
                curr_command += 1
                reward += synthetic_reward
                if (curr_command == total_no_of_commands+1)
                    curr_command = total_no_of_commands
                    areCommandsAvailable = false
                end
            end
            
            
            reward += stay_alive_penalty
            push!(history, s, action, reward)
            episode_reward += reward
            if(done)
                println("episode: ", episode, " reward: ", episode_reward)
            end
            if(episode%10==0)
                Knet.save("w_montezuma_hailmary.jld2", "model", w)
            end
            done && break
        end
    
        episode % infotime == 0 && close!(env)

        dw = grad(loss)(w, history)
        update!(w, dw, opt)    
    end
    
    return w
end
