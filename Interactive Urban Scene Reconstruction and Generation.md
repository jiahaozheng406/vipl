# Interactive Urban Scene Reconstruction and Generation



### Background

目前 Real2Sim 大致可以分为三种路线：

1. 基于几何、图形学等从 RGB / Depth 提取 Mesh （Vid2Sim）
   
   - 还原度高
   
   - 提取 Mesh 质量显著低于资产库中的 Asset

2. 通过 CLIP / DINO 等计算相似度在资产库中筛选最匹配的 Asset （UrbanVerse）
   
   - Asset 质量高
   
   - 多样性和还原度受到资产库的制约

3. 将 Image / Video 等输入作为 Prompt 生成场景 （DIMR）
   
   - 生成结果可能无法严格还原真实场景
   
   - 某种意义上，可以看作 1 和 2 之间的 Tradeoff
     
     

### Motivation

可以发现几个路线都存在一定的问题，可以尝试对以上三种思路进行结合（参考 DIMR），既可以通过筛选匹配度高的 Asset 以保证高质量，也可以通过生成得到更多样、还原度更高的 Asset。同时，生成的 Asset 经过人工检查和优化可以加入资产库中，从而不断丰富资产库。总体而言，该工作可以产出一个持续进化的 Real2Sim Pipeline，同时得到一个相比 UrbanVerse-100K 更大的资产库。



创新点：

- 基于 Retrieval 与 Generation 相结合的 Real2Sim Pipeline

- 持续进化的资产库与生成模型



### Related Papers

- （Baseline）URBANVERSE: SCALING URBAN SIMULATION BY  WATCHING CITY-TOUR VIDEOS （ICLR 2026）

- Point Scene Understanding via Disentangled  Instance Mesh Reconstruction （ECCV 2022）

- PhysX-3D: Physical-Grounded 3D Asset Generation （NIPS 2025）
  
  

### Method

- 仿照 UrbanVerse 提取 sky、ground、object 的信息作为 prompt

- 将 prompt 输入 Encoder 得到 latent feature，计算相似度，如果相似度高于阈值直接筛选高匹配度 Asset

- 基于 UrbanVerse-100K 训练生成 Asset 生成模型

- 相似度低于阈值时进行 Asset 的生成

- 对生成的 Asset 进行人工核查与优化，加入到 Asset 库中

- 随着 Asset 库的扩大，定期微调生成模型



### Experiments

1. 场景恢复程度（还原度、准确性、多样性等）

2. Sim2Real 表现

3. 持续学习的效果



### Schedule

Goal：ICRA 2027 / CVPR 2027
