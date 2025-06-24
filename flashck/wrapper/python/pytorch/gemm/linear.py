"""
Linear API
"""

from typing import Callable, Dict, Optional, Tuple, Union

class Linear:
    
    def __init__(self, in_features:int,  out_features: int, bias: bool = True, params_dtype: Optional[torch.dtype] = None,specialization: Optional[str] = None)-> None:
        super().__init__()
        
        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.specialization = specialization
        self.use_bias = bias

        # Contiguous buffers for params
        weight_tensor = torch.empty(
            self.out_features,
            self.in_features,
            device=device,
            dtype=params_dtype,
        )
        bias_tensor = None
        if self.use_bias:
            bias_tensor = torch.empty(
                self.out_features,
                device=device,
                dtype=params_dtype,
            )
    
    @no_torch_dynamo()
    def forward(self,
        *args: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
    """
        assert len(args) >= 1

        if len(args) == 2:
            if self.use_bias:
                inputs = [x, self.weight.tensor(), self.bias.tensor(), args[1]]
            else:
                inputs = [x, self.weight.tensor(), args[1]]
            output = self.op(*inputs)
            return output
        
        output = (
            self.op(x, self.weight.tensor(), bias=self.bias.tensor())
            if self.use_bias
            else self.op(x, self.weight.tensor())
        )
        return output
        

        
