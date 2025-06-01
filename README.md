# From Undersampling K-space to Individual Anatomy Modeling  
**Comprehensive Head Digital Twins via Super-Resolution Reconstruction and Segmentation**

## ✨ Overview
This project presents an end-to-end workflow that bridges theoretical MRI knowledge with practical implementation in accelerated imaging and high-fidelity anatomical modeling. The core objective is to develop a pipeline that starts from undersampled MRI k-space data, reconstructs high-resolution images using deep learning, and generates individualized head digital twins for clinical applications such as digital dentistry and neuromuscular analysis.

## 🧠 Project Goals
- Simulate **MRI undersampling** and explore different k-space downsampling patterns.
- Reconstruct super-resolution images using **CNN-based MRI reconstruction networks**.
- Construct high-resolution **Head Digital Twins (HDTs)** by segmenting craniofacial and CNS structures with **nnUNet**.
- Enable personalized anatomical modeling for diagnosis, surgical planning, and functional research.

## 🧰 Major Components
```
MRI_ACCELERATE_RECO/
├── digital_twins_results/                   # Inferred segmentation results (.nii.gz)
├── Part1_kspace_Downsample_experiment/     # Downsampling masks and FFT/iFFT experiments
├── Part2_CNN_Reconstruction_MRI_Deep/      # CNN super-resolution reconstruction
│   ├── models/ utils/
│   ├── ReconBaseline.py
│   └── ReconWithImprove.py
├── Part3_Digital_twins_by_nnUNet/          # Head digital twins construction via nnUNet
│   ├── nnunetv2/ documentation/
│   └── setup.py, LICENSE, readme.md
├── Paper_From_Undersampling_K-space...pdf  # Full technical report
└── README.md
```

## 📌 Algorithms Used
- **Undersampling Strategies**: Cartesian, Spiral, Radial, Compressed Sensing, etc.
- **Super-Resolution Reconstruction**: Deep cascaded CNN with k-space Data Consistency layers (PSNR ↑, SSIM ↑)
- **Segmentation**: Universal segmentation with nnUNet for 52 brain regions and 17 craniofacial anatomical region modeling.
- **Evaluation Metrics**: PSNR, SSIM (reconstruction); Dice coefficient (segmentation)

## 📊 Key Results
- CNN reconstruction achieves **PSNR: 28.81**, **SSIM: 0.89**
- HDT segmentation achieves average **Dice: 0.91** across tissues
- Demonstrated clinically meaningful structures including brain hemispheres, mandible, TMJ discs, spinal cord, etc.

## 📬 Contact
**Wei Jia**  
School of Biomedical Engineering, ShanghaiTech University  
📧 Email: [jiawei1@shanghaitech.edu.cn](mailto:jiawei1@shanghaitech.edu.cn)
