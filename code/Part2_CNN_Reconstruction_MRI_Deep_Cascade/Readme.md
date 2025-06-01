# A Deep cascade CNN for undersampled MRI reconstruction   
## Installation and Running    
Make sure the required **dependencies** are met:
<ul>
<li>numpy</li>
<li>torch</li>
<li>torchvision</li>
<li>matplotlib</li>
<li>scikit-image</li>
<li>tqdm</li> 
</ul>
To install the dependencies directly, you can use the commands below directly.
    
    pip install -r requirements.txt

Also, the original dataset is in the format of **.nii**, and our codes only support the format os **.npy**. However, it is a pity that the npz files are too big to upload. So, we provide the code to do the conversion locally.
The conversion can be done with the following commands with local path:

    python -u Nii2Npz.py


Then you can run both **ReconBaseline.py** and **ReconWithImprove.py** to train models with the following commands with local path:

    python -u ReconBaseline.py

or

    python -u ReconWithImprove.py

In order to train the models locally, you should at first change the codes line that indicates the model saved path and the dataset path.

## Results
After running the codes, you can get:
<ul>
<li>The visualization of both fully-sampled images and the undersampled image
<li>The visualization of training progress
<li>The visualization of the output reconstruction result
<li>The result of avg_PSNR and avg_SSIM
</ul>
