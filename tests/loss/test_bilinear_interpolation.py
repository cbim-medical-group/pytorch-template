import torch.nn.functional
dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor

def bilinear_interpolate_torch_gridsample(image, samples_x, samples_y):
                                                # input image is: W x H x C
    image = image.permute(2,0,1)                # change to:      C x W x H
    image = image.unsqueeze(0)                  # change to:  1 x C x W x H
    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)
    samples = torch.cat([samples_x, samples_y],3)
    samples[:,:,:,0] = (samples[:,:,:,0]/(W-1)) # normalize to between  0 and 1
    samples[:,:,:,1] = (samples[:,:,:,1]/(H-1)) # normalize to between  0 and 1
    samples = samples*2-1                       # normalize to between -1 and 1
    print(f"image:{image.shape}, sample:{samples.shape}")
    return torch.nn.functional.grid_sample(image, samples)

# Correctness test
W, H, C = 5, 5, 1
test_image = torch.ones(W,H,C).type(dtype)
test_image[3,3,:] = 4
test_image[3,4,:] = 3

test_samples_x = torch.FloatTensor([[3.2]]).type(dtype)
test_samples_y = torch.FloatTensor([[3.4]]).type(dtype)

print(bilinear_interpolate_torch_gridsample(test_image, test_samples_x, test_samples_y))
