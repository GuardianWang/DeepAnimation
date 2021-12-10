# DeepAnimation
CS2470 Fall 2021 final project.

## Download data and model weights

```bash
bash download.sh
bash download_weights.sh
```

## Circle sampler

Given a point 
<img src="https://render.githubusercontent.com/render/math?math=p" alt="">
on the circle, we can define a circle by specifying
<img src="https://render.githubusercontent.com/render/math?math=v_2" alt="">,
which is the unit vector from 
<img src="https://render.githubusercontent.com/render/math?math=p" alt="">
to circle center
<img src="https://render.githubusercontent.com/render/math?math=o" alt="">,
and 
<img src="https://render.githubusercontent.com/render/math?math=v_1" alt="">,
which is the unit tangent vector that defines the sampling direction.
Then we sample angles from 
<img src="https://render.githubusercontent.com/render/math?math=[0, 2\pi]" alt="">.
For a given
<img src="https://render.githubusercontent.com/render/math?math=\theta" alt="">,
the corresponding point on the circle is defined as
<img src="https://render.githubusercontent.com/render/math?math=p" alt=""> +
<img src="https://render.githubusercontent.com/render/math?math=(r - r \cos \theta)v_2" alt="">+
<img src="https://render.githubusercontent.com/render/math?math=r \sin \theta v_1" alt="">.


![circle_sampling](doc/circle_sampling.png)

This is the visualization of samples surrounding an MNIST image.
We use VAE trained on MNIST to generate  this gif.

![MNIST](doc/gif001.gif)

## Data from other domain

We tried this image with Chinese character on the VAE model trained 
on MNIST dataset.

![wang](doc/unseen.png)

The result is as follows, which means it doesn't generate well for 
different domains.

![MNIST](doc/unseen.gif)

We also notice that the model tends to generate 3 and 6.
This is possibly because we are using the same sampling vectors
as the previous example.

## Environment Variable

On Linux:
```bash
export PYTHONPATH='${PYTHONPATH}:/path/to/DeepAnimation'
```
On Windows:
```bash
set PYTHONPATH=%PYTHONPATH%;\path\to\DeepAnimation\
```
On Colab:
```python
import os
os.environ['PYTHONPATH'] += ':/path/to/DeepAnimation'
```
## Results

### t-SNE visualization
Here we plot the t-SNE visualization of frames sampled fron all 1416 gifs,
each with 10 frames. 
![tsne](doc/tsne-vis.png)

### image query

We used the ahash algorithm as well as the latent vector to do deep hashing.
Therefore we can query hand drawings and fetch well-designed images
in our dataset.
With the hand drawing, the top-10 results are:

![draw](doc/zoom_draw.png)
![result](doc/zoom_query.png)

Note that the fifth result is a perfect match!

More results:
![more](doc/more_query.png)


### play with your drawings

1. `cd` into `vaegan/experiments` 
2. Modify `query.py` `test_nearest_dataset()`. `frame_dir`
is image directory and `fmt` is image file name.
3. run `python query.py`
