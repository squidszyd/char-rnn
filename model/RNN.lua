local RNN = {}

function RNN.rnn(input_size, rnn_size, n, dropout)
  
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x NOTE-zyd: Creates a module that returns whatever is input to it as output. 
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]

  end

  --[[
  NOTE-zyd:
    What does the two () in "nn.Identity()()" mean?
    Ans:
    nngraph overloads the call operator(i.e. the () operator, which is used for function calls)
    on all the nn.Module objects.When the call operator is invoked, it returns a node wrapping
    the nn.Module. The call operator takes the parents of the node as arguments, which specify 
    which modules will feed into this one during a forward pass.
    Therefore, nn.Identity() is a nn.Module while nn.Identity()() is a graph node.
    For detailed info, go to https://github.com/torch/nngraph/
  ]]--

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    
    local prev_h = inputs[L+1]
    if L == 1 then 
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return RNN
