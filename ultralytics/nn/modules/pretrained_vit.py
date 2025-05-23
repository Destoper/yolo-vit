import torch
import torch.nn as nn
from torch.hub import load
from torchvision import transforms
import math
from huggingface_hub import login
import timm

# from aim.v2.utils import load_pretrained

dino_backbones = {
    "small": {"name": "dinov2_vits14_reg", "embedding_size": 384, "patch_size": 14},
    "base": {"name": "dinov2_vitb14_reg", "embedding_size": 768, "patch_size": 14},
    "large": {"name": "dinov2_vitl14_reg", "embedding_size": 1024, "patch_size": 14},
    "giant": {"name": "dinov2_vitg14_reg", "embedding_size": 1536, "patch_size": 14},
}


class DinoV2Patches(nn.Module):
    def __init__(self, in_chanels=3, out_channels=768, size="base"):
        from torchvision import transforms

        super(DinoV2Patches, self).__init__()
        self.size = size
        self.backbone = load("facebookresearch/dinov2", dino_backbones[self.size]["name"], pretrained=True)
        self.backbone.eval()
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self.out_channels = out_channels
        self.inet_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def transform(self, x):
        # x should have shape (B, C, H, W)
        b, c, h, w = x.shape

        # Compute how many pixels to drop to make height/width multiples of 14
        h_new = h - (h % 14)
        w_new = w - (w % 14)

        dh = h - h_new  # total pixels to drop in height
        dw = w - w_new  # total pixels to drop in width

        # Decide how many pixels to drop from each side.
        # For simplicity, drop half from each edge if dh/dw are even.
        # If dh or dw are odd, we'll drop one extra pixel from the "bottom/right".
        dh_top = dh // 2
        dh_bottom = dh - dh_top
        dw_left = dw // 2
        dw_right = dw - dw_left

        # Crop the tensor from edges
        # Make sure slicing indices do not go negative
        x_cropped = x[:, :, dh_top : h - dh_bottom, dw_left : w - dw_right]

        # normalize with imagenet mean and std
        x_cropped = self.inet_norm(x_cropped)

        return x_cropped

    def forward(self, x):
        with torch.no_grad():
            x = self.transform(x)
            batch_size = x.shape[0]
            mask_dim = (x.shape[2] / 14, x.shape[3] / 14)
            with torch.no_grad():
                x = self.backbone.forward_features(x)
                x = x["x_norm_patchtokens"]
                x = x.permute(0, 2, 1)
                x = x.reshape(batch_size, self.out_channels, int(mask_dim[0]), int(mask_dim[1]))
            return x

class UNIPatches(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024):
        super(UNIPatches, self).__init__()
        
        # Load UNI model (requires authentication)
        try:
            # Attempt to load with cached credentials
            self.backbone = timm.create_model("hf-hub:MahmoodLab/uni", 
                                            pretrained=True, 
                                            init_values=1e-5, 
                                            dynamic_img_size=True)
        except:
            print("Please authenticate with your Hugging Face token")
            login()  # Interactive login
            self.backbone = timm.create_model("hf-hub:MahmoodLab/uni", 
                                            pretrained=True, 
                                            init_values=1e-5, 
                                            dynamic_img_size=True)
            
        self.backbone.eval()
        
        # ImageNet normalization parameters
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self.out_channels = out_channels
        self.inet_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def transform(self, x):
        # Ensure dimensions work well with the model - UNI uses 16×16 patches
        b, c, h, w = x.shape
        
        # Make dimensions multiples of 16 for UNI's patch size
        h_new = h - (h % 16)
        w_new = w - (w % 16)
        
        dh = h - h_new
        dw = w - w_new
        
        # Center crop
        dh_top = dh // 2
        dh_bottom = dh - dh_top
        dw_left = dw // 2
        dw_right = dw - dw_left
        
        # Crop the tensor from edges
        x_cropped = x[:, :, dh_top : h - dh_bottom, dw_left : w - dw_right]
        
        # Normalize with ImageNet mean and std - THIS IS NECESSARY!
        x_normalized = self.inet_norm(x_cropped)
        
        return x_normalized

    def forward(self, x):
        with torch.no_grad():
            x = self.transform(x)
            
            # Extract features from UNI
            # For UNI, we need to extract patch features based on its specific output format
            features = self.backbone.forward_features(x)
            print(f"Features shape: {[f.shape for f in features]}")
            # Extract patch tokens and reshape to spatial format
            # Note: The exact code depends on UNI's output structure
            if hasattr(self.backbone, "get_intermediate_layers"):
                # If using timm's newer interfaces
                patch_tokens = features[:, 1:, :]  # Skip CLS token
                
                # Calculate spatial dimensions (should result in 14×14)
                h = w = int(patch_tokens.shape[1] ** 0.5)
                
                # Reshape to [batch, channels, height, width]
                x = patch_tokens.permute(0, 2, 1)
                x = x.reshape(x.shape[0], self.out_channels, h, w)
            else:
                # If UNI already outputs features in the expected format
                x = features  # May need adjustment based on actual output
                
            return x

class UNIMultiLayerPatches(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024, layers=[23], fusion_mode='list'):
        super().__init__()
        
        # Load standard UNI model
        self.backbone = timm.create_model(
            "hf-hub:MahmoodLab/uni", 
            pretrained=True, 
            init_values=1e-5, 
            dynamic_img_size=True
        )
        
        self.backbone.eval()
        self.layers = layers
        self.fusion_mode = fusion_mode
        self.out_channels = out_channels
        self.out_channels_per_block = out_channels
        if fusion_mode == 'concat':
            self.out_channels_per_block = out_channels // len(layers)
            
        self.patch_size = 16  # UNI uses 16×16 patches
        
        # ImageNet normalization
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self.inet_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def transform(self, x):
        b, c, h, w = x.shape
        
        # Make dimensions multiples of patch size
        h_new = h - (h % self.patch_size)
        w_new = w - (w % self.patch_size)
        
        # Center crop
        dh_top = (h - h_new) // 2
        dh_bottom = h - h_new - dh_top
        dw_left = (w - w_new) // 2
        dw_right = w - w_new - dw_left
        
        x_cropped = x[:, :, dh_top : h - dh_bottom, dw_left : w - dw_right]
        x_normalized = self.inet_norm(x_cropped)
        
        return x_normalized
    
    def forward(self, x):
        with torch.no_grad():
            x = self.transform(x)
            batch_size = x.shape[0]
            
            # Calculate expected features dimension
            h_patches = x.shape[2] // self.patch_size
            w_patches = x.shape[3] // self.patch_size
            
            # Get intermediate layer outputs using UNI's native functionality
            features = self.backbone.get_intermediate_layers(x, self.layers)
            
            processed_features = []
            for idx, feat in enumerate(features):
                # Skip the class token (first token) and only use patch tokens
                patch_tokens = feat
                
                # Calculate spatial dimensions from actual tokens
                num_tokens = patch_tokens.shape[1]
                
                # Safety check - verify we have expected number of tokens
                expected_tokens = h_patches * w_patches
                if num_tokens != expected_tokens:
                    print(f"Warning: Token count mismatch. Expected {expected_tokens}, got {num_tokens}")
                    # Try to adapt by finding closest square
                    side = int(math.sqrt(num_tokens))
                    if side * side != num_tokens:
                        # Pad to next perfect square if needed
                        new_size = (side + 1) * (side + 1)
                        padding = new_size - num_tokens
                        patch_tokens = torch.cat([
                            patch_tokens, 
                            torch.zeros(batch_size, padding, self.out_channels_per_block, device=patch_tokens.device)
                        ], dim=1)
                        h_patches = w_patches = side + 1
                    else:
                        h_patches = w_patches = side
                
                # Reshape to [batch, channels, height, width]
                spatial_feat = patch_tokens.permute(0, 2, 1).reshape(
                    batch_size, self.out_channels_per_block, h_patches, w_patches
                )
                processed_features.append(spatial_feat)
            
            # Apply fusion mode
            if self.fusion_mode == 'list':
                return processed_features
            elif self.fusion_mode == 'concat':
                return torch.cat(processed_features, dim=1)
            elif self.fusion_mode == 'add':
                return sum(processed_features)
            else:
                return processed_features[0]  # Default to first feature

class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Select and return a particular index from input.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]

class EnhancedUNIPatches(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024):
        super().__init__()
        
        # Load UNI model with correct configuration
        self.backbone = timm.create_model("hf-hub:MahmoodLab/uni", 
                                        pretrained=True, 
                                        init_values=1e-5,
                                        dynamic_img_size=True)
        
        self.backbone.eval()
        self.out_channels = out_channels
        
        
        # ImageNet normalization
        self.transform = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )
        
        # Add spatial feature alignment
        self.feature_align = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        with torch.no_grad():
            # Handle dimensions - UNI uses 16×16 patches
            b, c, h, w = x.shape
            
            # Preserve aspect ratio but ensure divisibility by 16
            h_new = (h // 16) * 16
            w_new = (w // 16) * 16
            
            # Center crop
            h_offset = (h - h_new) // 2
            w_offset = (w - w_new) // 2
            x = x[:, :, h_offset:h_offset+h_new, w_offset:w_offset+w_new]
            
            # Normalize
            x = torch.stack([self.transform(img) for img in x])
            
            # Extract features
            features = self.backbone.forward_features(x)
            
            # Properly reshape to spatial format - match expected shape
            patch_tokens = features[:, 1:, :]  # Skip CLS token
            h_out = h_new // 16
            w_out = w_new // 16
            
            # Reshape to [batch, channels, height, width]
            spatial_feats = patch_tokens.permute(0, 2, 1).reshape(b, self.out_channels, h_out, w_out)
            
        # Apply learnable feature alignment (not frozen)
        return self.feature_align(spatial_feats)