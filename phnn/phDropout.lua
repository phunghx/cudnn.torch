local phDropout, Parent = torch.class('phnn.phDropout', 'nn.Module')
function phDropout:__init(p)
   Parent.__init(self)
   self.p = p or 0.5
   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end
   self.noise = torch.Tensor()
end

function phDropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.noise:resizeAs(input)
   self.noise:bernoulli(1-self.p)
   self.output:cmul(self.noise)
   return self.output
end

function phDropout:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
   return self.gradInput
end
