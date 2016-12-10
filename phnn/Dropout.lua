local phDropout, Parent = torch.class('nn.Dropout', 'nn.Module')

function Dropout:__init(p)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end
   self.noise = torch.Tensor()
end


function Dropout:updateOutput(input)
   return input
end

function Dropout:updateGradInput(input, gradOutput)
   return input
end

function Dropout:setp(p)
   self.p = p
end
