# Brain-Tumor-Segmentation
The proliferation of abnormal cells within the brain is called brain tumor. Brain tumors are
classified into two types: malignant (cancerous) and benign (noncancerous). Primary brain
tumors are those that develop in the brain; secondary (metastatic) brain tumors are those that
develop after cancer has progressed to other regions of the body and reached the brain.

The process of separating a brain tumor from normal brain tissues is known as segmentation.
In clinical practice, it is useful for diagnosis and therapy planning. However, due to the irregular
shape and perplexing limits of tumors, it remains a difficult process. Tumor cells act as a heat
source because their temperature is higher than that of normal brain cells.

The most prevalent kind of human brain tumors are Gliomas. Because of their changing
structure and characteristics in multi-modal magnetic resonance imaging, proper segmentation
is a difficult medical image analysis problem (MRI). Segmentation of such brain tumors by
hand has many drawbacks:
• It necessitates a high level of medical knowledge.
• It is very time-consuming.
• It is Prone to human error
• Lacks consistency and reproducibility.
• Ultimately result in the wrong diagnosis and course of treatment.

In a variety of applications, including microscope image segmentation, image classification
and among many others, convolutional neural networks (CNNs) have significantly advanced,
enabling models to compete with or outperform humans. U-Net is the most popular CNN
architecture used for medical image segmentation, and it serves as the basis for cutting-edge
models of brain tumor segmentation. A timely diagnosis, made feasible by precisely segmenting the brain tumor, can lower the fatality rate. In this study, we present an optimized nnU-Net architecture with deep supervision for automated brain tumor segmentation with a greater accuracy rate compared to existing methods, for an effective diagnosis.
