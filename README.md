# SpeechRobot


# Architecture
## Backbone
![图片1](https://github.com/whichwhichgone/SpeechRobot/blob/main/backbone.png)

# Results
## wandb
https://wandb.ai/mypostdoc/huggingface?nw=nwuseryijiulanpishu

## exps for mmllms
| Model                  | VQAv2(test) | GQA   | VizWiz | SQA  | TextVQA | POPE  | LibriSpeech(WER)|GQA-speech(VCTK)|GQA-speech(LibriTTS)|
|------                  |------       |------ |------  |------|------   |------ |------           |------          |------
|LLaVa                   | 78.77       | 61.95 | 50.0   | 66.8 | 58.17   | 85.88 | N/A             |
|LLaVa Speech            | 78.67       | 61.86 | 51.14  | 72.15| 58.06   | 85.52 | N/A             |23.72           |
|LLaVa Speech 3548       |             | 61.97 |        |      |         |       | N/A             |20.27           |
|LLaVa Speech connector  |             | 63.21 |        |      |         |       | 2.79%           |44.35           |50.80
|Whisper Large v2        |N/A          |N/A    |N/A     |N/A   |N/A      |N/A    | 2.7%            |N/A             |N/A


# Todo
- [x] 跑通LLaVA-7B基础实验代码，确定模型初步结构
- [x] 添加ASR数据集，设计ASR任务模版
- [x] Librispeech-clean-360上进行基座1阶段SFT
- [x] 多说话人数据集构建，基于VQA数据集来构造新的多说话人数据集
- [x] LLava原数据集、合成数据集、Librispeech-clean-100上进行基座2阶段SFT
- [x] 基座模型简单Eval
- [ ] 根据CALVIN数据集合成对应的音频指令数据
- [ ] 在合成的CALVIN数据集上对基座进行SFT
- [x] 仿真环境进行操控验证
- [ ] 加入区分声纹的的adapter，对该adapter进行FT
- [ ] 设计实验场景，不同声纹对应不同的knowledge，并且根据声纹进行RAG

# Issues
1. 模型训练时预测的action格式，绝对坐标还是相对坐标？针对夹爪夹取的动作也是统一的loss函数么？夹爪夹取的动作两个离散的值？
2. 仿真环境中，当得到action之后是否有进一步的数据处理？
3. SFT持续了多少个epoch？
4. SFT的范式是RT2的这种范式么？当前输入是两幅图的话，这两幅图是如何加入的？训练过程有没有加入本体信息？
