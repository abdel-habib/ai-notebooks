import torch

def trace_module_shapes(module, input_tensor):
    '''
    Traces the shapes of inputs and outputs for each layer in a PyTorch module.
    
    Args:
        module (torch.nn.Module): The PyTorch module to trace.
        input_tensor (torch.Tensor): A sample input tensor to pass through the module.

    Returns:
        None
    '''
    print("=" * 90)
    print(f"{'Layer':<20} {'Trainable':<20} {'Input Shape':<30} {'Output Shape'}")
    print("-" * 90)

    def forward_hook(mod, input, output):
        '''
        A hook function to capture the input and output shapes of a layer. It is called inside the forward pass.

        Args:
            mod (torch.nn.Module): The layer/module being traced.
            input (tuple): The input to the layer, which is a tuple of tensors.
            output (torch.Tensor): The output from the layer.

        Returns:
            None
        '''
        class_name = mod.__class__.__name__
        input_shape = tuple(input[0].shape)
        output_shape = tuple(output.shape)
        is_trainable =  'True' if any(p.requires_grad for p in module.parameters()) else 'False'

        print(f"{class_name:<20} {is_trainable:<20} {str(input_shape):<30} {str(output_shape)}")

    hooks = []
    is_leaf = lambda m: len(list(m.children())) == 0

    # Special case: single layer
    if is_leaf(module):
        hooks.append(module.register_forward_hook(forward_hook))
    else:
        for layer in module.modules():
            if layer is not module and is_leaf(layer):
                hooks.append(layer.register_forward_hook(forward_hook))

    # Forward pass
    with torch.no_grad():
        module(input_tensor)

    # Add bottom table row
    print("=" * 90)

    # Remove hooks
    for hook in hooks:
        hook.remove()
