import numpy as np
import pyexr

eps = 0.00316

def preprocess_diffuse(diffuse, albedo):
    return diffuse / (albedo + eps)

def postprocess_diffuse(diffuse, albedo):
    return diffuse * (albedo + eps)

def preprocess_specular(specular):
    assert(np.sum(specular < 0) == 0)
    return np.log(specular + 1)

def postprocess_specular(specular):
    return np.exp(specular) - 1

def preprocess_diff_var(variance, albedo):
    return variance / (albedo + eps)**2

def preprocess_spec_var(variance, specular):
    return variance / (specular+1e-5)**2

def gradients(data):

    h, w, c = data.shape
    dX = data[:, 1:, :] - data[:, :w - 1, :]
    dY = data[1:, :, :] - data[:h - 1, :, :]
     # padding with zeros
    dX = np.concatenate((np.zeros([h,1,c], dtype=np.float32),dX), axis=1)
    dY = np.concatenate((np.zeros([1,w,c], dtype=np.float32),dY), axis=0)
  
    return np.concatenate((dX, dY), axis=2)

def remove_channels(data, channels):
    for c in channels:
        if c in data: del data[c]

# returns network input data from noisy .exr file
def preprocess_input(filename, gt, debug=False):
  
    file = pyexr.open(filename)
    data = file.get_all()

    # just in case
    for k, v in data.items():
        data[k] = np.nan_to_num(v)

    file_gt = pyexr.open(gt)
    gt_data = file_gt.get_all()
  
    # just in case
    for k, v in gt_data.items():
        gt_data[k] = np.nan_to_num(v)
    
    # clip specular data so we don't have negative values in logarithm
    data['Specular'] = np.clip(data['Specular'], 0, np.max(data['Specular']))
    gt_data['Specular'] = np.clip(gt_data['Specular'], 0, np.max(gt_data['Specular'])) 
        
    # save albedo
    data['origAlbedo'] = data['Albedo'].copy()
        
    # save reference data (diffuse and specular)
    diff_ref = preprocess_diffuse(gt_data['Diffuse'], gt_data['Albedo'])
    spec_ref = preprocess_specular(gt_data['Specular'])
    diff_sample = preprocess_diffuse(data['Diffuse'], data['Albedo'])
    
    data['Reference'] = np.concatenate((diff_ref[:,:,:3].copy(), spec_ref[:,:,:3].copy()), axis=2)
    data['Sample'] = np.concatenate((diff_sample, data['Specular']), axis=2)
    
    # save final input and reference for error calculation
    # apply albedo and add specular component to get final color
    data['finalGt'] = gt_data['default']#postprocess_diffuse(data['Reference'][:,:,:3], data['albedo']) + data['Reference'][:,:,3:]
    data['finalInput'] = data['default']#postprocess_diffuse(data['diffuse'][:,:,:3], data['albedo']) + data['specular'][:,:,3:]
        
    # preprocess diffuse
    data['Diffuse'] = preprocess_diffuse(data['Diffuse'], data['Albedo'])

    # preprocess diffuse variance
    # data['diffuseVariance'] = preprocess_diff_var(data['diffuseVariance'], data['albedo'])

    # preprocess specular
    data['Specular'] = preprocess_specular(data['Specular'])

    # preprocess specular variance
    # data['specularVariance'] = preprocess_spec_var(data['specularVariance'], data['specular'])

    # just in case
    data['Depth'] = np.clip(data['Depth'], 0, np.max(data['Depth']))

    # normalize depth
    max_depth = np.max(data['Depth'])
    if (max_depth != 0):
        data['Depth'] /= max_depth
        # also have to transform the variance
        # data['depthVariance'] /= max_depth * max_depth

    # Calculate gradients of features (not including variances)
    data['gradNormal'] = gradients(data['Normal'][:, :, :3].copy())
    data['gradDepth'] = gradients(data['Depth'][:, :, :1].copy())
    data['gradAlbedo'] = gradients(data['Albedo'][:, :, :3].copy())
    data['gradSpecular'] = gradients(data['Specular'][:, :, :3].copy())
    data['gradDiffuse'] = gradients(data['Diffuse'][:, :, :3].copy())
    data['gradIrrad'] = gradients(data['default'][:, :, :3].copy())

    # # append variances and gradients to data tensors
    # data['diffuse'] = np.concatenate((data['diffuse'], data['diffuseVariance'], data['gradDiffuse']), axis=2)
    # data['specular'] = np.concatenate((data['specular'], data['specularVariance'], data['gradSpecular']), axis=2)
    # data['normal'] = np.concatenate((data['normalVariance'], data['gradNormal']), axis=2)
    # data['depth'] = np.concatenate((data['depthVariance'], data['gradDepth']), axis=2)

    X_diffuse = np.concatenate((data['Diffuse'],
                            data['Normal'],
                            data['Depth'],
                            data['Position'],
                            data['gradAlbedo']), axis=2)

    X_specular = np.concatenate((data['Specular'],
                            data['Normal'],
                            data['Depth'],
                            data['Position'],
                            data['gradAlbedo']), axis=2)
    
    assert not np.isnan(X_diffuse).any()
    assert not np.isnan(X_specular).any()

    if debug:
        print("Diffuse Component shape:", X_diffuse.shape)
        print(X_diffuse.dtype, X_specular.dtype)
        
    data['X_diffuse'] = X_diffuse
    data['X_specular'] = X_specular
    
    remove_channels(data, ('diffuseA', 'specularA', 'normalA', 'albedoA', 'depthA',
                            'visibilityA', 'colorA', 'gradNormal', 'gradDepth', 'gradAlbedo',
                            'gradSpecular', 'gradDiffuse', 'gradIrrad', 'Albedo', 'Color', 
                            'Depth', 'Diffuse', 'Specular', 'diffuseVariance', 'specularVariance',
                            'depthVariance', 'visibilityVariance', 'colorVariance',
                            'normalVariance', 'visibility', 'Position'))
    
    return data