
# 年度工作总结
本年度总结围绕前沿算法跟踪、模型微调训练、算法优化部署及相关文档编写四个方面，对本人年度工作的主要内容与成果进行了梳理与总结。

## 1. 前沿算法跟踪
本章节将从相关数据统计以及算法归纳总结两个方面，介绍本人在前沿算法跟踪所做的工作。
### 1.1 相关数据统计
目前已统计文生视频11个、图生视频8个、动作驱动4个、轨迹控制3个SOTA算法，其主观效果与客观指标的详细内容已整理至下方链接。

[文生视频](https://alidocs.dingtalk.com/i/nodes/R1zknDm0WR3eEBv5iDqK134jVBQEx5rG?doc_type=wiki_doc&rnd=0.241896100495117)
[图生视频](https://alidocs.dingtalk.com/i/nodes/YndMj49yWjPva4OQHQ5RvwgQJ3pmz5aA?doc_type=wiki_doc&rnd=0.5075524582403479)
[动作控制](https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8azvMBnEh51yQQBpWgN7R35y?doc_type=wiki_doc&rnd=0.5013674858600519)
[轨迹控制I2V](https://alidocs.dingtalk.com/i/nodes/G1DKw2zgV2RXg5D2IzyXXkpLVB5r9YAn?doc_type=wiki_doc)
[轨迹控制T2V](https://alidocs.dingtalk.com/i/nodes/6LeBq413JAzG0bRQhyaw31lA8DOnGvpb?doc_type=wiki_doc&rnd=0.18352237255750992)
### 1.2 算法归纳总结
<div align="center">
	<img src=".\img\视频生成模型发展.png" alt="视频生成算法发展" width="600">
</div>

基于Diffusion模型的视频生成技术的发展可以分为三个主要阶段：伪2+1D Unet时期、2+1D Dit时期，以及3D Dit时期。时间复杂度从简单到复杂，逐步实现了视频生成算法的优化和升级。

在 伪2+1D Unet时期，核心算法以 [AnimateDiff](https://github.com/guoyww/AnimateDiff) 和 [SVD](https://github.com/Stability-AI/generative-models) 为代表，这个时期的算法通常是基于预训练的图像生成模型引入额外的运动模块进行视频生成。这些算法主要依赖 2D VAE 进行初步特征提取。位置编码方面采用相对位置编码和可学习位置编码，文本编码器则以 [CLIP](https://github.com/openai/CLIP) 为主。这一时期技术特点是注重静态图像的时间轴扩展，但对动态复杂性的捕捉能力有限。

进入 2+1D Dit时期，主要以 [Open-Sora 1.2](https://github.com/hpcaitech/Open-Sora) 和 [Open-Sora-Plan 1.2.0](https://github.com/PKU-YuanGroup/Open-Sora-Plan) 为代表，模型增加了对时空注意力机制（Spatial & Temporal Attention）的支持，不再使用预训练的图像生成模型，而是用视频进行模型的从头训练。引入了3D Causal VAE来处理更复杂的时空特征。同时，位置编码扩展为3D旋转位置编码，文本编码器升级为 T5，进一步提升了模型对视频生成的动态与语义理解能力。

在 3D Dit时期，技术达到了更高的复杂度和精细度，以 [CogvideoX](https://github.com/THUDM/CogVideo)、[Mochi](https://github.com/genmoai/mochi) 、 [OpenSora-Plan 1.3.0](https://github.com/PKU-YuanGroup/Open-Sora-Plan) 和[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)为代表。模型主要采用全3D注意力机制（Full 3D Attention）技术，时间复杂度增加的同时，生成视频的质量也有了质的飞跃。VAE使用更先进的3D Causal VAE 和 [WF-VAE](https://github.com/PKU-YuanGroup/WF-VAE)，位置编码则采用3D旋转位置编码以增强时空特征的表达。文本编码器仍为 T5或 T5+CLIP。

对于条件控制视频生成，伪2+1D Unet时期会使用与Unet相匹配的ControlNet作为条件适配器，而到了Dit时期取消了Unet后，则采用的是条件编码器来进行额外条件控制的适配器。

整个技术发展体现了从2+1D到3D，从单纯时序到时空一体化的演变，同时模型的精度、生成质量和动态复杂性也得到了显著提升。

## 2. 模型微调训练

<div align="center">

<table>
  <tr>
    <th>类型</th>
    <th>数据集</th>
    <th>输出视频</th>
  </tr>
  <tr>
    <td>
      画风
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/7fbda3e0-c797-426e-b12c-14856bc02921" width="200" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/6a7f055b-33a8-4c8e-ad8b-a8b361c08260" width="600" controls></video>
    </td>
  </tr>
  <tr>
    <td>
      人物
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/b08a80f4-6258-4702-a491-7da8869a4bf5" width="200" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/25d21c00-5d3e-4dc1-aaed-3c97a4e8a1b5" width="600" controls></video>
    </td>
  </tr>
</table>

</div>

模型微调训练主要是以Cogvideox+LoRA的技术路线实现，以满足人物、画风定制化。




## 3. 算法优化部署
本章节将从算法优化以及算法部署两个方面，介绍本人在算法优化部署所做的工作。
### 3.1 算法优化
#### 3.1.1 SVD算法优化
<div align="center">

<table>
  <tr>
    <th>算法</th>
    <th>SVD</th>
    <th>SVD</th>
	<th>SVD+exVideo</th>
  </tr>
  <tr>
    <th>帧率</th>
    <th>25</th>
    <th>64</th>
	<th>64</th>
  </tr>
  <tr>
    <td>
      效果
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/3b80519d-aed6-4129-9600-070d19bcbf6b" width="200" controls></video>
    </td>
    <td>
	<video src="https://github.com/user-attachments/assets/6fff63c5-2062-4f6e-b141-ec78d93d8ac0" width="200" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/a8db2d63-0801-4030-916f-e940484667fd" width="200" controls></video>
    </td>
  </tr>
</table>

</div>
SVD算法在生成大于25帧的视频会发生明显的抖动现象，在结合exVideo算法后，就能避免这种现象。

#### 3.1.2 Mimicmotion算法流程优化
<div align="center">

<table>
  <tr>
    <th>输入图像</th>
	<th>输入视频</th>
    <th>Mimicmotion</th>
    <th>+Facefusion</th>
	<th>+SGM-VFI</th>
	<th>(+插值法校正)</th>
  </tr>
  <tr>
    <td>
      <img src=".\img\fatman.jpg" width="100">
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/e7f5f981-51c7-4d31-aee0-183bc0f04a5d" width="80" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/6f6485ff-6149-49ff-8537-1b7b95989393" width="80" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/ad031616-f284-4fb4-9bbd-62b2db68fa84" width="80" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/a666a8f0-8260-4e25-8716-3df4e4b23d4b" width="80" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/cd6ab76e-ab20-4e05-8f5a-774c06b3748f" width="80" controls></video>
    </td>
  </tr>
</table>

</div>
Mimicmotion算法在处理输入图像主题与输入视频主体体型差异较大时，存在以下问题：无法有效还原输入图像中的主体体型，人脸难以保持与输入图像一致，且生成视频帧数减少一半，导致视频流畅性不足。为解决这些问题，我们引入了 FaceFusion 技术，使生成视频能够更好地保持输入图像中的人脸特征。同时，采用 SGM-VFI算法 对视频进行帧插值，大幅提升视频流畅性。此外，通过插值法校正体型关键点，使输出视频能够更准确地保持输入图像的主体体型。然而，由于插值法的引入，在人物距离镜头过近的场景中，可能会导致主体结构出现崩坏现象。
<div align="center">

<table>
  <tr>
    <th>w/o 插值矫正</th>
	<th>w 插值校正</th>
    
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/064a4b50-c183-40f4-bf61-787dda88fb3e" width="200" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/6c1a45e6-ca77-4bbb-994a-42ac4e349005" width="200" controls></video>
    </td>
  </tr>
</table>

</div>

#### 3.1.3 Cogvideo文生视频算法流程优化
<div align="center">

<table>
  <tr>
    <th>Cogvideox</th>
    <th>+llm扩词</th>
    <th>+SGM-VFI</th>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/c23b7858-9e89-4436-8179-3712019a8812" width="200" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/1deb3900-e6c6-4a7f-b6d2-ab549c006bac" width="200" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/dbe87ea7-a9a5-4fa8-8a05-69c87c0d15f6" width="200" controls></video>
    </td>
  </tr>
</table>

</div>
通过运用大语言模型对提示词进行扩展，可以使视频生成模型的生成内容更加生动与丰富。同时，结合 SGM-VFI 算法对生成视频进行帧插值处理，提升了视频的流畅性和观感。

#### 3.1.4 Cogvideo图生视频算法流程优化
<div align="center">

<table>
  <tr>
    <th>输入图像</th>
    <th>调整前</th>
    <th>调整后</th>
  </tr>
  <tr>
    <td>
      <img src=".\img\图片1.png" width="200">
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/e73e5972-1948-4677-a3e7-f3b4a704a814" width="300" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/5534fd2c-91b1-4ac6-8f59-6e616a5745d6" width="200" controls></video>
    </td>
  </tr>
</table>

</div>
调整前，CogvideoX 的图生视频模型无法还原输入图像的原始分辨率，容易导致视频生成过程中出现变形问题。为了解决这一问题，我们在生成视频前对输入图像进行黑边填充，生成后再切割掉黑边。这种方法能够有效还原输入图像的分辨率，避免形变的发生。

#### 3.1.5 Mochi算法提速

<div align="center">

<table>
  <tr>
    <th>算法</th>
    <th>Mochi</th>
    <th>Mochi(diffuser)</th>
	<th>Mochi+xDiT(ray)</th>
	<th><a href="https://github.com/zishen-ucap/Mochi-Diffusers-xDit" target="_blank">Mochi (diffusers) + xDiT (xfuser)</a></th>
  </tr>
  <tr>
    <td>
      效果
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/fc10ca10-4edb-4b0e-8234-d729bd1e88ce" width="200" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/24a80411-1aed-4853-87d9-27219aa29ee0" width="200" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/00b86301-42b0-4676-aa76-d04913a10381" width="200" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/79579b21-fbde-4694-a8dc-292bc6dcc8d0" width="200" controls></video>
    </td>
  </tr>
  <tr>
    <th>显存（G）</th>
    <th>25*4</th>
    <th>24</th>
	<th>38*4</th>
	<th>45*4</th>
  </tr>
  <tr>
    <th>推理时间（s）</th>
    <th>308</th>
    <th>827</th>
	<th>279</th>
	<th>286</th>
  </tr>

</table>

</div>
使用xDiT提高了Mochi算法的推理速度。

### 3.2 算法部署
完成相应视频生成接口开发：

- 文生视频：Cogvideox(T2V 5B) （已上至开悟集市）
- 图生视频：Cogvideox(I2V 5B)
- 动作驱动：Mimicmotion


## 4. 相关文档撰写
### 4.1 专利交底书撰写
完成专利交底书撰写并与知识产权公司完成对接，后续流程由专利公司负责推进：

- 《基于文本的运动视频生成方法、装置、存储介质及设备》
- 《基于文本描述的视频拼接方法、装置、存储介质及设备》

### 4.2 视频类技术介绍文档的撰写
视频类介绍文档：文生视频、图生视频、动作驱动、图像换风格、视频换脸、图像换脸、图片换背景
