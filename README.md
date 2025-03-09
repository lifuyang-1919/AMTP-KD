# AMTP-KD: Adaptive Multi-Teacher Pruned Knowledge Distillation  

High-performance LiDAR-based 3D object detectors often face significant computational overhead, necessitating their compression into lightweight detectors. Knowledge distillation (KD) is an effective approach for sparse 3D object detection compression.  

We propose **Adaptive Multi-Teacher Pruned Knowledge Distillation (AMTP-KD)**, which:  
- Generates **student-friendly teacher assistants** through structured and unstructured pruning.  
- Introduces a **sample-level teacher hm-loss-based fusion strategy**, incorporating a **hard selection strategy** and a **soft multi-stage Softmax-T strategy** for adaptive weighting.  
- Proposes **multi-teacher pivotal region masks** to enhance knowledge transfer.  

Extensive experiments on **Waymo & KITTI datasets** demonstrate that our method achieves a **4× reduction in FLOPs and parameters** without noticeable accuracy loss in LiDAR-based 3D object detection.  


