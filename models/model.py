
from torch import nn as nn

class Model(nn.Module):
    def __init__(self, module_list):
        super(Model, self).__init__()
        self.model = nn.ModuleList(module_list)

    def __iter__(self):
        ''' Returns the Iterator object '''
        return iter(self.model)

    def __len__(self):
        return len(self.model)

    def __getitem__(self, index):
        return self.model[index]

    def __setitem__(self, idx, value):
        self.model[idx] = value

    def register_backward_hooks(self):
        for module_pt in self.model:
            if hasattr(module_pt,'register_masking_hooks'):
                module_pt.register_masking_hooks()
    
    def unregister_backward_hooks(self):
        for module_pt in self.model:
            if hasattr(module_pt,'unregister_masking_hooks'):
                module_pt.unregister_masking_hooks()    

    def forward(self, x):
        for module_pt in self.model:
            x = module_pt(x)
        return x

