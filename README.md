# StainNormalization
implemetns [SCAN algorithm](https://doi.org/10.1016/j.cmpb.2020.105506) for stain normalization in pure pytorch. This is used for stain normalization of histopathology scans. 

## Brief method summary
Briefly this works by solving a self consistency problem, where the stain vectors of hematoxylin and eosin are estimated as the median color(in optical density space) of the stroma and nuclei pixels. The Nuclei and stroma pixels are extracted by a simple heuristic. THe initial stain vectors are estimated by PCA/SVD of the optical densities of the whole image (ignoring white space).

![Image demonstrating the effects of Stain normalization. The reference image does not change, as expected, while the other images become visibly more similar to the reference image after stain normalization. The upper row shows the raw images, the lower row contains the stain normalized versions.](data/example.png)

## Usage
```python
from scan import ScanProcessor
from torchvision.io import read_image

# fitting to the reference image
processor = ScanProcessor()
processor.to(device)
ref_input = read_image("data/reference.jpg").unsqueeze(0).to(device)/255.0
processor.fit(ref_input)

# performing stain normalization 
# Model outputs the masks for stroma, nuclei and the eosin and hematoxyling stain intensities

image = read_image("data/target.jpg").unsqueeze(0).to(device)/255.0
normalized_image = processor(image)[0] # the output image is in float32 values in [0,1]
```