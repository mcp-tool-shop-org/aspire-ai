# ASPIRE Test Coverage Improvement Plan

## Instructions for Claude

**Goal:** Improve test coverage from 44% to 80%+

**Process:**
1. Pick 20 unchecked items from this list
2. Write the tests
3. Run `pytest tests/ --cov=aspire --cov=integrations -q` to verify
4. Mark completed items with `[x]`
5. Commit changes
6. Repeat until coverage target is met

**Current Coverage:** 44% (411 tests passing)

---

## Priority 1: Core ASPIRE Package (High Impact)

### CLI Tests (`aspire/cli.py` - 0% coverage)
- [ ] Test `aspire --version` outputs version
- [ ] Test `aspire -V` outputs version
- [ ] Test `aspire` with no args shows help
- [ ] Test `aspire doctors` checks Python version
- [ ] Test `aspire doctor` checks PyTorch
- [ ] Test `aspire doctor` checks transformers
- [ ] Test `aspire doctor` with ANTHROPIC_API_KEY set
- [ ] Test `aspire doctor` with ANTHROPIC_API_KEY missing
- [ ] Test `aspire doctor` with OPENAI_API_KEY set
- [ ] Test `aspire doctor` disk space check
- [ ] Test `aspire init` creates config file
- [ ] Test `aspire init --output custom.yaml` uses custom path
- [ ] Test `aspire teachers` lists all teachers
- [ ] Test `aspire train` with demo prompts (mock trainer)
- [ ] Test `aspire train --config` loads config file
- [ ] Test `aspire evaluate` with mock checkpoint

### Trainer Tests (`aspire/trainer.py` - 17% coverage)
- [ ] Test AspireTrainer initialization with default config
- [ ] Test AspireTrainer._init_student with LoRA
- [ ] Test AspireTrainer._init_student without LoRA
- [ ] Test AspireTrainer._init_student with 4-bit quantization
- [ ] Test AspireTrainer._init_student with 8-bit quantization
- [ ] Test AspireTrainer._init_critic with "head" architecture
- [ ] Test AspireTrainer._init_critic with "separate" architecture
- [ ] Test AspireTrainer._init_critic with "shared_encoder" architecture
- [ ] Test AspireTrainer._init_teacher with Claude
- [ ] Test AspireTrainer._init_teacher with OpenAI
- [ ] Test AspireTrainer._init_loss creates AspireLoss
- [ ] Test AspireTrainer._init_optimizers creates optimizers
- [ ] Test AspireTrainer._save_checkpoint creates files
- [ ] Test AspireTrainer.load_checkpoint restores state
- [ ] Test AspireDataset.__len__ returns correct length
- [ ] Test AspireDataset.__getitem__ returns tokenized data

### Teacher Tests - Claude (`aspire/teachers/claude.py` - 20% coverage)
- [ ] Test ClaudeTeacher initialization with API key
- [ ] Test ClaudeTeacher raises ClaudeTeacherError without key
- [ ] Test ClaudeTeacher.challenge returns TeacherChallenge (mock API)
- [ ] Test ClaudeTeacher.challenge with different challenge types (mock API)
- [ ] Test ClaudeTeacher.challenge builds history context
- [ ] Test ClaudeTeacher.evaluate returns TeacherEvaluation (mock API)
- [ ] Test ClaudeTeacher.evaluate with generate_improved=True (mock API)
- [ ] Test ClaudeTeacher.evaluate with generate_improved=False (mock API)
- [ ] Test ClaudeTeacher._get_challenge_description for all types
- [ ] Test ClaudeTeacher handles JSON parse errors gracefully

### Teacher Tests - OpenAI (`aspire/teachers/openai.py` - 15% coverage)
- [ ] Test OpenAITeacher initialization with API key
- [ ] Test OpenAITeacher raises OpenAITeacherError without key
- [ ] Test OpenAITeacher.challenge returns TeacherChallenge (mock API)
- [ ] Test OpenAITeacher.challenge with different challenge types (mock API)
- [ ] Test OpenAITeacher.evaluate returns TeacherEvaluation (mock API)
- [ ] Test OpenAITeacher.evaluate with generate_improved=True (mock API)
- [ ] Test OpenAITeacher handles JSON parse errors gracefully

### Teacher Tests - Local (`aspire/teachers/local.py` - 11% coverage)
- [ ] Test LocalTeacher initialization with model path
- [ ] Test LocalTeacher.challenge returns TeacherChallenge (mock model)
- [ ] Test LocalTeacher.evaluate returns TeacherEvaluation (mock model)
- [ ] Test LocalTeacher._generate_text produces output (mock model)
- [ ] Test LocalTeacher with custom tokenizer

### Student Loss Tests (`aspire/losses/student.py` - 84% coverage)
- [ ] Test RewardLoss with edge case scores (0, 10)
- [ ] Test ContrastiveLoss with identical embeddings
- [ ] Test TrajectoryLoss with declining scores
- [ ] Test CoherenceLoss with uniform logits
- [ ] Test KLDivergenceLoss with temperature scaling
- [ ] Test StudentLoss weight configuration

---

## Priority 2: Integration Code (Medium Impact)

### Code Data (`integrations/code/data.py` - 22% coverage)
- [ ] Test CodeSample creation with all fields
- [ ] Test CodeSample.from_file reads file correctly
- [ ] Test CodeSample.from_string creates sample
- [ ] Test GitHubRepoCollector initialization
- [ ] Test GitHubRepoCollector.clone_repo (mock git)
- [ ] Test GitHubRepoCollector.collect_samples finds Python files
- [ ] Test GitHubRepoCollector.collect_samples finds JS files
- [ ] Test TrainingPairGenerator creates pairs
- [ ] Test TrainingPairGenerator with quality filtering
- [ ] Test CodeDataset.__len__ and __getitem__

### Code Trainer (`integrations/code/trainer.py` - 11% coverage)
- [ ] Test CodeCriticTrainer initialization
- [ ] Test CodeCriticTrainer.train_step computes loss
- [ ] Test CodeCriticTrainer.evaluate returns metrics
- [ ] Test CodeCriticTrainer.save_checkpoint
- [ ] Test CodeCriticTrainer.load_checkpoint

### Code Teacher (`integrations/code/code_teacher.py` - 63% coverage)
- [ ] Test CorrectnessChecker with valid async code
- [ ] Test StyleGuide with long lines
- [ ] Test SecurityAuditor with pickle.loads
- [ ] Test ArchitectureReviewer with deeply nested code
- [ ] Test PerformanceAnalyst with inefficient patterns
- [ ] Test CodeTeacher with "rotate" strategy
- [ ] Test CodeTeacher with "debate" strategy
- [ ] Test CodeTeacher.get_improvement_suggestions

---

## Priority 3: Optional Integrations (Lower Priority)

### Forge Integration (`integrations/forge/` - 0% coverage)
- [ ] Test VisionTeacher initialization (mock API)
- [ ] Test VisionTeacher.critique_image (mock API)
- [ ] Test ImageCritic.score_image
- [ ] Test ImageCritic with different architectures

### Isaac Integration (`integrations/isaac/` - 0% coverage)
- [ ] Test MotionTeacher initialization
- [ ] Test MotionTeacher.evaluate_trajectory
- [ ] Test TrajectoryCritic.score_motion
- [ ] Test IsaacWrapper environment setup (mock Isaac)

---

## Test Utilities to Create

### Fixtures Needed (add to `tests/conftest.py`)
- [ ] `mock_anthropic_client` - Mock Anthropic API responses
- [ ] `mock_openai_client` - Mock OpenAI API responses
- [ ] `mock_student_model` - Mock HuggingFace model
- [ ] `mock_tokenizer` - Mock tokenizer
- [ ] `temp_config_file` - Temporary YAML config
- [ ] `temp_checkpoint_dir` - Temporary checkpoint directory

### Test File Structure
```
tests/
├── conftest.py              # Shared fixtures
├── test_cli.py              # NEW - CLI tests
├── test_trainer_unit.py     # NEW - Trainer unit tests (mocked)
├── test_teachers_claude.py  # NEW - Claude teacher tests (mocked)
├── test_teachers_openai.py  # NEW - OpenAI teacher tests (mocked)
├── test_teachers_local.py   # NEW - Local teacher tests (mocked)
├── test_code_data.py        # NEW - Code data tests
├── test_code_trainer.py     # NEW - Code trainer tests
└── ... existing tests ...
```

---

## Progress Tracking

| Date | Tests Added | Coverage | Notes |
|------|-------------|----------|-------|
| 2026-01-22 | 411 | 44% | Initial baseline |
| | | | |
| | | | |

---

## Commands

```bash
# Run all tests with coverage
pytest tests/ --cov=aspire --cov=integrations --cov-report=term -q

# Run specific test file
pytest tests/test_cli.py -v

# Run with HTML report
pytest tests/ --cov=aspire --cov=integrations --cov-report=html

# Skip slow tests
pytest tests/ -m "not slow" --cov=aspire -q
```

---

## Notes

- **Mock external APIs** - Don't make real API calls in tests
- **Mock model loading** - Use lightweight mocks instead of loading real models
- **Use `@pytest.mark.asyncio`** for async tests
- **Keep tests fast** - Target < 30 seconds for full suite
- **Windows compatible** - Use `freeze_support()` where needed
