import torch

def split_image_into_patches(input_image, patch_size, model, padding=0):

    image_dim, image_height, image_width = input_image.shape
    mask = torch.zeros(image_dim, image_height, image_width)  
    
    i = 0

    while i < image_height:
        j = 0

        while j < image_width:

            if i + patch_size > image_height:
                start_i = image_height - patch_size 
                save_i = i + patch_size - image_height 
                delta_i = image_height - i - 2*padding
            else:
                start_i = i
                save_i = 0
                delta_i = patch_size - 2*padding

            end_i = start_i + patch_size

            if j + patch_size > image_width:
                start_j = image_width - patch_size
                save_j = j + patch_size - image_width 
                delta_j = image_width - j - 2*padding
            else:
                start_j = j
                save_j = 0
                delta_j = patch_size - 2*padding

            end_j = start_j + patch_size

            patch = input_image[:, start_i:end_i, start_j:end_j]
            

            processed_patch = model(patch.unsqueeze(0)).squeeze(0)
 
            mask[:, i + padding: i + padding + delta_i, j + padding : j + padding + delta_j] = processed_patch[:, save_i : patch_size - 2*padding, save_j : patch_size - 2*padding]

            j += patch_size - 2* padding

        i += patch_size - 2*padding

    mask = mask[:, padding : image_height - padding, padding : image_width - padding ]

    return mask