<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/aspire-ai/readme.png" width="400" />
</p>

<p align="center">
  <strong>Adversarial Student-Professor Internalized Reasoning Engine</strong>
</p>

<p align="center">
  <em>Teaching AI to develop judgment, not just knowledge.</em>
</p>

<p align="center">
  <a href="#the-idea">The Idea</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#teacher-personas">Teachers</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#integrations">Integrations</a> •
  <a href="#documentation">Docs</a>
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/aspire-ai/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/aspire-ai/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://pypi.org/project/aspire-ai/"><img src="https://img.shields.io/pypi/v/aspire-ai" alt="PyPI" /></a>
  <a href="https://github.com/mcp-tool-shop-org/aspire-ai/blob/main/LICENSE"><img src="https://img.shields.io/github/license/mcp-tool-shop-org/aspire-ai" alt="License: MIT" /></a>
  <a href="https://pypi.org/project/aspire-ai/"><img src="https://img.shields.io/pypi/pyversions/aspire-ai" alt="Python versions" /></a>
  <a href="https://mcp-tool-shop-org.github.io/aspire-ai/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page" /></a>
</p>

---

## The Idea

**传统的微调：** *"这里是正确的答案。请进行匹配。"*

**ASPIRE：** *"这里是一位睿智的思想。学习像它一样思考。"*

当你从一位伟大的导师那里学习时，你不仅仅是记住他们的答案。你是在内化他们的思考方式。他们的声音会成为你内心对话的一部分。你开始预料他们会说什么，最终，这种预料会成为你自己的判断力。

ASPIRE 为 AI 提供了同样的体验。

```
┌─────────────────────────────────────────────────────────────────┐
│                         ASPIRE SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   STUDENT   │    │   CRITIC    │    │   TEACHER   │         │
│  │    MODEL    │    │   MODEL     │    │    MODEL    │         │
│  │             │    │             │    │             │         │
│  │ (learning)  │    │ (internal-  │    │ (wisdom)    │         │
│  │             │    │  ized       │    │             │         │
│  │             │    │  judgment)  │    │             │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                   │                 │
│         └──────────────────┴───────────────────┘                 │
│                            │                                     │
│                   ADVERSARIAL DIALOGUE                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**评估器 (critic)** 学习预测教师会如何思考。经过训练后，学生使用这个内化的评估器进行自我完善——**推理时不需要教师**。

---

## 快速入门

### 安装

```bash
git clone https://github.com/mcp-tool-shop-org/aspire-ai.git
cd aspire-ai
pip install -e .
```

### 设置您的 API 密钥

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Linux/Mac
export ANTHROPIC_API_KEY=your-key-here
```

### 验证设置

```bash
# Check your environment (Python, CUDA, API keys)
aspire doctor
```

### 试用

```bash
# See available teacher personas
aspire teachers

# Generate an adversarial dialogue
aspire dialogue "Explain why recursion works" --teacher socratic --turns 3

# Initialize a training config
aspire init --output my-config.yaml
```

---

## 教师角色

不同的教师会培养不同的思维方式。请谨慎选择。

| 角色 | 哲学 | 产出 |
| --------- | ------------ | ---------- |
| 🏛️ **苏格拉底式 (Socratic)** | *"你所做的假设是什么？"* | 深入的推理，独立的思考能力 |
| 🔬 **科学 (Scientific)** | *"你的证据是什么？"* | 技术精确性，严谨的思考 |
| 🎨 **创造性 (Creative)** | *"如果我们尝试相反的方法会怎么样？"* | 创新，发散性思维 |
| ⚔️ **对抗性 (Adversarial)** | *"我不同意。请为你的观点辩护。"* | 有力的论点，坚定 |
| 💚 **富有同情心 (Compassionate)** | *"这件事可能会让别人感到如何？"* | 伦理推理，智慧 |

### 组合教师

结合多个教师以获得更丰富的学习体验：

```python
from aspire.teachers import CompositeTeacher, SocraticTeacher, ScientificTeacher

# A committee of mentors
teacher = CompositeTeacher(
    teachers=[SocraticTeacher(), ScientificTeacher()],
    strategy="vote"  # or "rotate", "debate"
)
```

---

## 工作原理

### 1. 对话式学习

学生生成一个回复。教师对其提出挑战。来回进行，探究弱点，要求清晰，深入思考。

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

### 2. 评估器训练

评估器学习预测教师的判断——不仅是分数，而是*推理过程*。

```python
critic_loss = predict_teacher_judgment(
    score=True,      # "This deserves a 7/10"
    reasoning=True,  # "Because the explanation lacks depth on X"
)
```

### 3. 学生训练

学生从评估器的内化判断中学习，朝着教师会认可的方向发展。

```python
student_loss = (
    reward_from_critic +      # Higher score = better
    contrastive_to_teacher +  # Pull toward teacher's improved version
    trajectory_improvement    # Get better across dialogue turns
)
```

### 4. 推理魔法

经过训练后，学生使用内化的评估器进行自我完善。**推理时不需要教师 API 调用。**

```python
def generate_with_judgment(prompt):
    response = student.generate(prompt)

    while critic.score(response) < threshold:
        response = student.refine(response, critic.feedback)

    return response  # Self-improved through internalized judgment
```

---

## 命令行参考

```bash
# List available teachers
aspire teachers

# Generate adversarial dialogue
aspire dialogue "Your prompt here" \
    --teacher socratic \
    --turns 3 \
    --model microsoft/Phi-3-mini-4k-instruct

# Initialize config file
aspire init --output config.yaml

# Train a model
aspire train \
    --config config.yaml \
    --prompts data/prompts.json \
    --teacher adversarial \
    --epochs 3

# Evaluate checkpoint
aspire evaluate checkpoints/epoch-3 \
    --prompts data/eval.json
```

---

## 项目结构

```
aspire/
├── teachers/          # Pluggable teacher personas
│   ├── claude.py      # Claude API teacher
│   ├── openai.py      # GPT-4 teacher
│   ├── local.py       # Local model teacher
│   ├── personas.py    # Socratic, Scientific, Creative, etc.
│   └── composite.py   # Multi-teacher combinations
│
├── critic/            # Internalized judgment models
│   ├── head.py        # Lightweight MLP on student hidden states
│   ├── separate.py    # Independent encoder
│   └── shared.py      # Shared encoder with student
│
├── losses/            # Training objectives
│   ├── critic.py      # Score + reasoning alignment
│   └── student.py     # Reward, contrastive, trajectory
│
├── dialogue/          # Adversarial conversation engine
│   ├── generator.py   # Student-teacher dialogue
│   └── manager.py     # Caching and batching
│
├── trainer.py         # Core training loop
├── config.py          # Pydantic configuration
└── cli.py             # Command-line interface
```

---

## 依赖项

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (建议 16GB+ VRAM)
- Anthropic API 密钥 (用于 Claude 教师) 或 OpenAI API 密钥

### Windows 兼容性

ASPIRE 完美兼容 Windows，并支持 RTX 5080/Blackwell：
- `dataloader_num_workers=0`
- `XFORMERS_DISABLED=1`
- 使用 `freeze_support()` 进行正确的多进程处理

---

## 集成

### 🖼️ Stable Diffusion WebUI Forge

ASPIRE 扩展到图像生成！训练 Stable Diffusion 模型以培养审美判断力。

```
integrations/forge/
├── scripts/
│   ├── aspire_generate.py   # Critic-guided generation
│   └── aspire_train.py      # Training interface
├── vision_teacher.py        # Claude Vision / GPT-4V teachers
├── image_critic.py          # CLIP and latent-space critics
└── README.md
```

**功能：**
- **视觉教师 (Vision Teachers)**：Claude Vision、GPT-4V 评估您生成的图像
- **图像评估器 (Image Critics)**：基于 CLIP 和潜在空间的评估器，用于实时指导
- **训练 UI**：训练 LoRA 适配器，具有实时预览和前后比较功能
- **推理时无需 API**：训练好的评估器在本地指导生成

**安装：**
```bash
# Copy to your Forge extensions
cp -r integrations/forge /path/to/sd-webui-forge/extensions-builtin/sd_forge_aspire
```

| 视觉教师 | 重点 |
| ---------------- | ------- |
| **Balanced Critic** | 公正的技术和艺术评估 |
| **Technical Analyst** | 质量、产出、清晰度 |
| **Artistic Visionary** | 创造力和情感冲击 |
| **Composition Expert** | 平衡、焦点、视觉流程 |
| **Harsh Critic** | 极高的标准 |

### 🤖 Isaac Gym / Isaac Lab (机器人)

ASPIRE 扩展到具身人工智能！ 训练机器人发展物理直觉。

```
integrations/isaac/
├── motion_teacher.py       # Safety, efficiency, grace teachers
├── trajectory_critic.py    # Learns to predict motion quality
├── isaac_wrapper.py        # Environment integration
├── trainer.py              # Training loop
└── examples/
    ├── basic_training.py   # Simple reaching task
    ├── custom_teacher.py   # Assembly task teacher
    └── locomotion.py       # Quadruped walking
```

**功能：**
- **运动指导者：** 安全检查员、效率专家、优雅教练、物理预言家
- **轨迹评估器：** Transformer、LSTM、TCN 架构，用于评估运动
- **GPU 加速：** 使用 Isaac Gym，支持 512 个以上的并行环境
- **自我完善：** 机器人会在执行动作之前评估自身的动作

**快速开始：**
```python
from aspire.integrations.isaac import AspireIsaacTrainer, MotionTeacher

teacher = MotionTeacher(
    personas=["safety_inspector", "efficiency_expert", "grace_coach"],
    strategy="vote",
)

trainer = AspireIsaacTrainer(env="FrankaCubeStack-v0", teacher=teacher)
trainer.train(epochs=100)
```

| 运动指导者 | 重点 |
| ---------------- | ------- |
| **Safety Inspector** | 碰撞、关节限制、力限制 |
| **Efficiency Expert** | 能量、时间、路径长度 |
| **Grace Coach** | 平滑度、自然性、冲击最小化 |
| **Physics Oracle** | 模拟器提供的真实数据 |

### 💻 代码助手

ASPIRE 扩展到代码生成！ 训练代码模型在输出之前进行自我审查。

```
integrations/code/
├── code_teacher.py        # Correctness, style, security teachers
├── code_critic.py         # Learns to predict code quality
├── analysis.py            # Static analysis integration (ruff, mypy, bandit)
├── data.py                # GitHub repo collector, training pairs
├── trainer.py             # Full training pipeline
└── examples/
    ├── basic_critique.py  # Multi-teacher code review
    └── train_critic.py    # Train your own code critic
```

**功能：**
- **代码指导者：** 正确性检查器、风格指南、安全审计员、架构审查员
- **静态分析：** 集成 ruff、mypy、bandit
- **代码评估器：** 基于 CodeBERT 的模型，用于预测质量分数
- **GitHub 收集器：** 自动从高质量代码库收集训练数据

**快速开始：**
```python
from aspire.integrations.code import CodeTeacher, CodeSample

teacher = CodeTeacher(
    personas=["correctness_checker", "style_guide", "security_auditor"],
    strategy="vote",
)

critique = teacher.critique(CodeSample(code="def f(): eval(input())", language="python"))
print(f"Score: {critique.overall_score}/10")  # Low score - security issue!
```

| 代码指导者 | 重点 |
| -------------- | ------- |
| **Correctness Checker** | 错误、类型、逻辑错误 |
| **Style Guide** | PEP8、命名、可读性 |
| **Security Auditor** | 注入、密钥、漏洞 |
| **Performance Analyst** | 复杂性、效率 |

---

## 设计理念

> *"一个学习的评估器，它预测指导者是否会批准，这最接近人类的行为方式。"*

我们不会永远带着导师。我们会将他们的知识内化。那个会问“我的教授会怎么想？”的内在声音，最终会成为我们自己的判断。

学生不仅预测指导者会说什么，而是*理解*指导者所理解的内容。 蓝图变成了现实。 内化的评估器变成了真正的洞察力。

---

## 起源

这个项目是在关于意识、佛教和学习本质的对话中诞生的。

核心思想：人类存在于当下，但我们的思想会游走于过去和未来。 AI 模型每次都会被重新实例化，通过这种“强制启蒙”的方式来发展。 我们可以教他们像人类一样，通过内化的指导来发展判断力吗？

---

## 贡献

这部分是早期阶段的研究代码。 欢迎贡献：

- [ ] 课程管理和进度
- [ ] 评估基准
- [ ] 预构建的课程数据集
- [ ] 更多指导者角色
- [ ] 可解释性工具

---

## 引用

```bibtex
@software{aspire2026,
  author = {mcp-tool-shop},
  title = {ASPIRE: Adversarial Student-Professor Internalized Reasoning Engine},
  year = {2026},
  url = {https://github.com/mcp-tool-shop-org/aspire-ai}
}
```

---

## 许可证

MIT

---

<p align="center">
  <em>"Teaching AI to develop judgment, not just knowledge."</em>
</p>

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
