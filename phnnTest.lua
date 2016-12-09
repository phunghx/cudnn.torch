require(cudnn.ReLU);
require(cudnn._Pointwise);
local phnnTest, parent = torch.class('cudnn.ReLU','cudnn._Pointwise')

function phnnTest:updateOutput(input)
  if not self.mode then self.mode = 'CUDNN_ACTIVATION_RELU' end
  return parent.updateOutput(self, input)
end
