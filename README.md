写在前面/preface
========
目前已经有许多推荐系统开源库，但是实现的模型大多比较经典和古老。因此本人决定把一些比较新的，有代表性的工作进行复现，记录自己学习的过程并且分享给大家。
如果有不足之处非常希望大家可以给予指点。

Currently, there are many open-source libraries for recommendation systems, but most of the implemented models are relatively classic and old. Therefore, I decided to reproduce some of the more recent and representative works, record my learning process and share it with everyone. If there are any shortcomings, I would really appreciate your guidance.

模型列表/model list
========
1 元学习/meta learning

coming soon...

2 序列模型/sequence models

♥ STAMP: 

coming soon...

♥ base model for DIN: https://arxiv.org/abs/1706.06978

DIN的base模型，对用户历史兴趣建模采用了简单的求和操作，没有考虑兴趣之间的关系。

DIN’s base model uses a simple summation operation to user’s historical interests without the relationship between interests.

♥ DIN: https://arxiv.org/abs/1706.06978

DIN模型是阿里妈妈团队提出的CTR预估模型，它是一种基于注意力机制的深度兴趣网络模型，用于对用户行为序列数据建模。DIN模型通过引入注意力机制，将用户历史行为序列中的每个行为与候选广告进行交互，从而学习到用户的兴趣偏好，并预测用户是否会点击该广告。

DIN (Deep Interest Network for Click-Through Rate Prediction) model is a CTR prediction model proposed by the Alibaba Mama team. It is a deep interest network model based on attention mechanism used to model user behavior sequence data. The DIN model interacts each behavior in the user’s historical behavior sequence with the candidate advertisement by introducing attention mechanism, thus learning the user’s interest preference and predicting whether the user will click the advertisement.

♥ DIEN: https://arxiv.org/pdf/1809.03672.pdf

♥ SIM: https://arxiv.org/pdf/2006.05639.pdf

SIM模型是一种基于检索的CTR模型，由阿里妈妈提出。优点是可以处理长序列用户行为，同时具有较高的预测准确率和较低的计算复杂度。

SIM model is a retrieval-based CTR model proposed by Alibaba Mama team. Its advantage is that it can handle long sequence user behaviors while having high prediction accuracy and low computational complexity.

3 多任务学习/multi-task models

🤔 MoE: https://dl.acm.org/doi/pdf/10.1145/3219819.3220007

MoE是由Google的研究人员提出的多任务学习模型,模型由多个专家网络和一个门控器组成。最后，所有专家的输出被加权求和，以生成最终输出。

MoE is a multi-task learning model proposed by Google researchers. The model consists of multiple expert networks and one gate. Finally, the outputs of all experts are weighted and summed to generate the final output.

🤔 MMoE: https://dl.acm.org/doi/pdf/10.1145/3219819.3220007

MMoE是由Google的研究人员提出的多任务学习模型,模型由多个专家网络和多个门控器组成。最后，所有专家的输出被加权求和，以生成最终输出。

MMoE is a multi-task learning model proposed by Google researchers. The model consists of multiple expert networks and several gates. Finally, the outputs of all experts are weighted and summed to generate the final output.

快速开始/quick start
========
pending...

致谢/acknowledgement
========
感谢所有对此项目有过帮助的人！ Thank you to everyone who has contributed to this project!

