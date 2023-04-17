写在前面/preface
========
目前已经有许多推荐系统开源库，但是实现的模型大多比较经典和古老。因此本人决定把一些比较新的，有代表性的工作进行复现，记录自己学习的过程并且分享给大家。
如果有不足之处非常希望大家可以给予指点。

Currently, there are many open-source libraries for recommendation systems, but most of the implemented models are relatively classic and old. Therefore, I decided to reproduce some of the more recent and representative works, record my learning process and share it with everyone. If there are any shortcomings, I would really appreciate your guidance.

模型列表/model list
========
1 冷启动/cold start

coming soon...

2 多任务学习/multi-task models

🤔 ESMM: https://arxiv.org/pdf/1804.07931.pdf

ESMM模型是一种多任务学习的方法，用于预测点击后的转化率。它同时学习两个任务：点击率和点击后转化率，并利用它们的乘积关系来隐式地学习转化率，解决了样本选择偏差和数据稀疏问题。

ESMM model is a multi-task learning method for predicting post-click conversion rate. It simultaneously learns two tasks: click-through rate and post-click conversion rate, and uses their product relationship to implicitly learn conversion rate, solving the problems of sample selection bias and data sparsity.

🤔 MoE: https://dl.acm.org/doi/pdf/10.1145/3219819.3220007

MoE是由Google的研究人员提出的多任务学习模型,模型由多个专家网络和一个门控器组成。最后，所有专家的输出被加权求和，以生成最终输出。

MoE is a multi-task learning model proposed by Google researchers. The model consists of multiple expert networks and one gate. Finally, the outputs of all experts are weighted and summed to generate the final output.

🤔 MMoE: https://dl.acm.org/doi/pdf/10.1145/3219819.3220007

MMoE是由Google的研究人员提出的多任务学习模型,模型由多个专家网络和多个门控器组成。最后，所有专家的输出被加权求和，以生成最终输出。

MMoE is a multi-task learning model proposed by Google researchers. The model consists of multiple expert networks and several gates. Finally, the outputs of all experts are weighted and summed to generate the final output.

🤔 CGC: https://dl.acm.org/doi/pdf/10.1145/3383313.3412236

CGC是腾讯提出的多任务学习模块，旨在解决跷跷板问题（负迁移问题）。通过为不同任务引入独立的专家网络解耦学习目标。

CGC is a multi-task learning module proposed by Tencent, aiming to solve the seesaw problem (negative transfer problem). It decouples the learning objectives by introducing independent expert networks for different tasks. 

🤔 PLE: https://dl.acm.org/doi/pdf/10.1145/3383313.3412236

PLE是腾讯提出的多任务学习模型，旨在解决跷跷板问题（负迁移问题）。通过为不同任务引入独立的专家网络解耦学习目标。它可以被看做是堆叠了多层CGC模块渐进式分层抽取学习模型。

PLE is a multi-task learning model proposed by Tencent, aiming to solve the seesaw problem (negative transfer problem). It decouples the learning objectives by introducing independent expert networks for different tasks. Moreover, it can be considered as a model stacking multiple CGC modules to progressively extract features.

🤔 Kuaishou-EBR: https://arxiv.org/pdf/2302.02657.pdf

快手在WWW2023最新提出的算法。文章从多任务学习的角度提出了embedding-based搜索召回的优化方案。该方法利用分而治之的思想提高EBR召回结果的多样性，新颖性等多个目标。

The latest algorithm proposed by Kuaishou at WWW2023. The paper proposes an optimization scheme for embedding-based retrieval recall from the perspective of multi-task learning. The method uses the divide-and-conquer idea to improve the diversity, novelty and other objectives of EBR recall results.

🤔 AITM: https://arxiv.org/pdf/2105.08489.pdf

AITM是美团发表在KDD2021的多任务学习算法。文章提出多个任务目标之间有先后的转化关系（曝光-点击-加购-付款），该模型使用自适应信息传递模块模拟多步转化过程中的顺序依赖关系，可以根据不同转化阶段自适应地学习要传递的信息和传递的程度。

AITM is a multi-task learning algorithm published by Meituan at KDD2021. The paper proposes that there is a sequential transformation relationship between multiple task objectives (exposure-click-add to cart-payment), and the model uses an adaptive information transformation module to simulate the sequential dependency relationship in the multi-step transformation process, which can adaptively learn the information and degree of transmission according to different stages.

3 序列模型/sequence models

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

♥ TiCoSeRec: https://arxiv.org/pdf/2212.08262.pdf

TiCoSeRec是基于CoSeRec算法的，由阿里巴巴和东北大学提出。文章提出了五种不同的数据增强算法，提升序列模型推荐效果。因此，本仓库只实现数据增强算法而不给出具体推荐算法实现。

TiCoSeRec, based on CoSeRec, is proposed by Alibaba and Northeast University. It presents five data argumentation algorithm to improve the performance of sequence recommender. Hence, here I just give the code of data argumentation instead of recommender.

文件结构/document structure
========
MTL: 多任务学习文件夹/multi-task

sequence：序列推荐文件夹/sequential recommender

cold：冷启动文件夹/cold start

快速开始/quick start
========
pending...

致谢/acknowledgement
========
感谢所有对此项目有过帮助的人！ Thank you to everyone who has contributed to this project!

