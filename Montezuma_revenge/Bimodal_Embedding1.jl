mutable struct Conv 
    w 
    b 
    f_activation
    p_drop
    f_pool
end
(c::Conv)(x) = c.f_activation.(c.f_pool(conv4(c.w, dropout(x,c.p_drop)) .+ c.b))


Conv(w1::Int,w2::Int,cx::Int,cy::Int;f=relu, pdrop=0, f_pool=pool) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop, f_pool)

struct Dense; w; b; f; p; end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int,o::Int,f=relu;pdrop=0) = Dense(param(o,i), param0(o), f, pdrop)

struct Projection; w; end
(d::Projection)(x) = d.w * x 
Projection(i::Int,o::Int) = Projection(param(o,i))

struct Embed; w; end
Embed(vocabsize::Int,embedsize::Int) = Embed(param(embedsize,vocabsize))
(e::Embed)(x) = e.w[:,x]

mutable struct frame_head
    conv1
    conv2
    conv3
    conv4
    fc
    output
end

function frame_head(w1,c1,w2,c2,w3,c3,w4,c4,hidden, outdims)
    conv1 = Conv(w1, w1, 1, c1)
    conv2 = Conv(w2, w2, c1, c2)
    conv3 = Conv(w3, w3, c2, c3)
    conv4 = Conv(w4, w4, c3, c4; f_pool = identity)
    fc = Dense(35840, hidden)
    output = Projection(hidden, outdims)
    frame_head(conv1, conv2, conv3, conv4, fc, output)
end
    

function (f::frame_head)(x)
    f.output(f.fc(f.conv4(f.conv3(f.conv2(f.conv1(x))))))
end

mutable struct sentence_head
    embed
    encoder
    fc
    output
end


function sentence_head(vocabsize::Int, embeddingsize::Int, hiddensize::Int)
    embed = Embed(vocabsize, embeddingsize)
    encoder = RNN(embeddingsize, hiddensize, rnnType = :lstm, h = 0)
    fc = Dense(8*hiddensize, 40)
    output = Projection(40,10)
    sentence_head(embed, encoder, fc, output)
end

function (s::sentence_head)(x)
    src_embed_tensor = s.embed(x)
    s.encoder.h = 0
    s.encoder.c = 0
    y_enc = s.encoder(src_embed_tensor)
    h, ty = size(y_enc)
    y_enc = reshape(y_enc, (h*ty, 1))
    s.output(s.fc(y_enc))
end

mutable struct bimodalEncoder
    fh
    sh
end


function bimodalEncoder(w1,c1,w2,c2,w3,c3,w4,c4,hidden, outdims, vocabsize, embeddingsize, hiddensize)
    fh = frame_head(w1,c1,w2,c2,w3,c3,w4,c4,hidden, outdims)
    sh = sentence_head(vocabsize, embeddingsize, hiddensize)
    bimodalEncoder(fh, sh)
end

function (b::bimodalEncoder)(image, label, y_truth)
    trans_nll(cosine_similarity(b, image_batch, seqbatch(labels)), y_truth)
end


function (b::bimodalEncoder)(x)
    #1 true, 2 false
    images, sent = x    
    images = reshape(images, (210,160,1,1))
    sent_rep = b.sh(sent)
    image_rep = b.fh(images)
    cos_sim = sum(sent_rep .* image_rep)/sqrt(sum(sent_rep.^2) + sum(image_rep.^2))
    #println("cos sim: ", cos_sim, "sent rep: ", sum(sent_rep), "image rep:", sum(image_rep))
    if(cos_sim>0)
        return 1
    end
    return 2

end
