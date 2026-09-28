"""Small seeded on-the-fly perturbations shared by every domain training arm."""
from io import BytesIO
import hashlib
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def rng_for(source_sha, sample_id, seed):
    data=f'{source_sha}:{sample_id}:{seed}:photometric-v1'.encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(data).digest()[:8],'big'))


def augment(image, source_sha, sample_id, seed=20260918):
    """Do not erase unknown defining regions; preserve the complete source canvas.

    This conservative initial policy has no label occlusion or clipped glare.
    Moderate noise/blur/compression can still lower readability and require QA;
    original content preservation is not a guarantee of legibility at every scale.
    """
    rng=rng_for(source_sha,sample_id,seed)
    parameters=dict(brightness=float(rng.uniform(.85,1.15)),
        contrast=float(rng.uniform(.9,1.1)),saturation=float(rng.uniform(.9,1.1)),
        white_balance=rng.uniform(.97,1.03,3).tolist(),noise_sigma=float(rng.uniform(0,1.8)),
        blur_radius=float(rng.uniform(0,.45)),jpeg_quality=int(rng.integers(82,99)),
        angle=float(rng.uniform(-6,6)),translation_x=float(rng.uniform(-.02,.02)),
        translation_y=float(rng.uniform(-.02,.02)),scale=float(rng.uniform(.9,1.)))
    image=image.convert('RGB')
    image=ImageEnhance.Brightness(image).enhance(parameters['brightness'])
    image=ImageEnhance.Contrast(image).enhance(parameters['contrast'])
    image=ImageEnhance.Color(image).enhance(parameters['saturation'])
    a=np.asarray(image,dtype=np.float32)*np.array(parameters['white_balance'],dtype=np.float32)
    a+=rng.normal(0,parameters['noise_sigma'],a.shape).astype(np.float32)
    image=Image.fromarray(np.clip(a,0,255).astype('uint8')).filter(ImageFilter.GaussianBlur(parameters['blur_radius']))
    b=BytesIO();image.save(b,format='JPEG',quality=parameters['jpeg_quality']);b.seek(0)
    image=Image.open(b).convert('RGB')
    image=image.rotate(parameters['angle'],resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(24,24,24))
    # Padding-only jitter: do not cut away a vintage/grape at the edge.
    size=max(image.size);canvas_size=int(np.ceil(size*1.12/parameters['scale']))
    canvas=Image.new('RGB',(canvas_size,canvas_size),(24,24,24))
    x=(canvas_size-image.width)//2+round(parameters['translation_x']*size)
    y=(canvas_size-image.height)//2+round(parameters['translation_y']*size)
    if x<0 or y<0 or x+image.width>canvas_size or y+image.height>canvas_size:
        raise ValueError('Jitter would truncate source content')
    canvas.paste(image,(x,y))
    parameters.update(label_occlusion=False,destructive_crop=False,motion_blur=False,
                      strong_glare=False,protected_regions='whole label, conservative initial recipe')
    return canvas,parameters
