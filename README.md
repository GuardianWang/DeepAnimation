# DeepAnimation
CS2470 Fall 2021 final project.


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