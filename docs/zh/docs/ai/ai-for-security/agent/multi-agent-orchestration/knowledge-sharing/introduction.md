# 简介

知识共享（Knowledge Sharing）是多智能体系统（Multi-Agent System）中的一种协作机制，用于支持不同 Agent 之间交换、传播和利用任务过程中产生的信息。

在单 Agent 系统中，Agent 通常依赖自身的 Memory 保存和检索历史信息。而在多智能体系统中，不同 Agent 可能分别负责不同子任务，因此需要一种机制使 Agent 能够共享彼此发现的重要信息，从而减少重复工作，提高整体任务完成效率。

例如，在一个安全分析场景中：

- Agent A 负责总体编排与目标规划
- Sub Agent a 负责日志分析，发现某个可疑 IP；
- Sub Agent b 负责威胁情报查询，发现该 IP 与某个恶意活动相关；
- Sub Agent c 负责生成事件报告。

如果没有知识共享机制，每个 Agent 只能基于自身的信息与上下文进行决策，从而可能会导致由于信息缺失而工作效率下降，且可能重复其他 Agent 已经进行过的动作，导致整体工作效率下降。而通过知识共享，Agent 可以将中间发现同步给其他 Agent，使整个系统形成协同分析能力。

特别地，在 CTF 领域的 Agent 设计中，部分开发者尤其喜欢使用 Concurrency Agent Swarm + Knowledge Sharing 的机制，这是因为在 Agent Swarm 中的多个 Agent 都朝着同一目标（解出同一道 CTF 题目）前进，不同 Agent 间可能发现互补线索，从而更快地推进整体的解题速度。

> 这种方式也适合低成本地将现有 Agent Application 进行包装，因为 knowledge sharing 的一种实现范式便是 Agent 主动将自身发现发送出去，对于开发者而言只需要构建一个简易的消息管道便能完成知识分发。 **相当一部分现有所谓“高效”的 CTF Agent 使用的都是这样的架构** ，因为这种方式可以通过很低的成本利用起 Codex、Claude Code 等 SOTA Agent 的性能，虽然细看会发现项目本身架构其实只是很简陋的 Wrapper，但架不住 Codex、Claude Code 本身是非常强的（笑）。

在后续章节中我们将以 [Flagent](https://github.com/a3infra/flagent) 项目的知识共享机制作为示例进行讲解，作为一个实验性质的项目，其支持包括主动共享与被动共享在内的多种不同的知识共享机制。

> 待施工。
