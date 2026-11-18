# AI-Powered Autonomous Change Impact Analysis & Software Delivery Platform

A closed-loop AI system that automatically analyzes code changes, builds and compares a custom dependency graph, predicts change/deployment risk, explains the prediction, and autonomously repairs and validates failures before deployment.

## Project Overview

Modern software systems contain many interconnected modules and services. A small code change in one component can unintentionally affect other components and cause build failures, test failures, security issues, or deployment failures.

This system provides:
- **Custom Dependency Analysis Engine** - Built from scratch using Python AST (no Neo4j, NetworkX, or GAT)
- **Change Impact Analysis** - Structural and textual change detection with impact propagation
- **Risk Prediction** - ML-based deployment risk prediction using CatBoost with SHAP explainability
- **Autonomous Repair** - LLM-powered code repair with automated validation loop
- **CI/CD Integration** - GitHub Actions pipeline for automated analysis

## Architecture

```
DEVELOPER → GITHUB PR → GITHUB ACTIONS → REPOSITORY ANALYZER → PYTHON AST PARSER
                                                      ↓
                                        CUSTOM DEPENDENCY EXTRACTOR → CUSTOM NAME RESOLVER
                                                      ↓
                                            CUSTOM GRAPH ENGINE
                                                      ↓
                                    GRAPH BEFORE vs GRAPH AFTER → CUSTOM GRAPH COMPARISON
                                                      ↓
                                            CUSTOM CHANGE ANALYZER
                                                      ↓
                                        CUSTOM IMPACT PROPAGATION & SCORING
                                                      ↓
                                            FEATURE GENERATION → CATBOOST → RISK PREDICTION
                                                      ↓
                                                    SHAP EXPLANATION
                                                      ↓
                                                    CODE LLM (Llama 3.3 / Qwen2.5-Coder)
                                                      ↓
                                            AUTONOMOUS REPAIR & VALIDATION LOOP
                                                      ↓
                                                    BUILD → TESTS → SECURITY → DEPENDENCY SCAN
                                                      ↓
                                                    PASS → DEPLOY | FAIL → RETRY (max iterations)
```

## Project Structure

```
AI-Change-Impact-Platform/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── settings.py
├── repository/
│   ├── repository_manager.py
│   ├── scanner.py
│   └── language_detector.py
├── analyzer/
│   ├── ast_parser.py
│   ├── dependency_extractor.py
│   ├── resolver.py
│   ├── change_analyzer.py
│   ├── function_analyzer.py
│   └── class_analyzer.py
├── graph_engine/
│   ├── node.py
│   ├── edge.py
│   ├── graph.py
│   ├── graph_builder.py
│   ├── graph_updater.py
│   └── graph_comparator.py
├── impact_engine/
│   ├── traversal.py
│   ├── propagation.py
│   ├── impact_scoring.py
│   └── affected_components.py
├── dataset/
│   ├── historical_pr_processor.py
│   ├── feature_builder.py
│   ├── label_generator.py
│   └── dataset_builder.py
├── risk_engine/
│   ├── train.py
│   ├── predict.py
│   ├── features.py
│   └── shap_explainer.py
├── repair_engine/
│   ├── llm_client.py
│   ├── code_analyzer.py
│   ├── repair_generator.py
│   └── repair_controller.py
├── validation/
│   ├── build_verifier.py
│   ├── unit_tests.py
│   ├── integration_tests.py
│   ├── security_scanner.py
│   └── dependency_scanner.py
├── cicd/
│   └── pipeline_controller.py
├── api/
│   └── main.py
├── tests/
│   ├── test_ast_parser.py
│   ├── test_dependency_extractor.py
│   ├── test_resolver.py
│   ├── test_graph.py
│   ├── test_graph_comparator.py
│   ├── test_impact_engine.py
│   └── test_change_analyzer.py
└── .github/
    └── workflows/
        └── analysis.yml
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd AI-Change-Impact-Platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

Edit `.env` file with your settings:

```env
# GitHub Configuration
GITHUB_TOKEN=your_github_token
GITHUB_REPO=owner/repo

# LLM Configuration
LLM_PROVIDER=ollama  # or openai, anthropic
LLM_MODEL=qwen2.5-coder
LLM_BASE_URL=http://localhost:11434

# Database (for historical data)
DATABASE_URL=sqlite:///data/history.db

# ML Configuration
CATBOOST_MODEL_PATH=models/catboost_model.cbm
```

## Usage

### As a GitHub Action

The system automatically triggers on Pull Request events via `.github/workflows/analysis.yml`.

### As a CLI Tool

```bash
# Analyze a repository
python -m api.main analyze --repo-path /path/to/repo --pr-number 123

# Train risk model
python -m risk_engine.train --data-path data/training_data.csv

# Run validation pipeline
python -m validation.run --repo-path /path/to/repo --commit-sha abc123
```

## Development Phases

### Phase 1: Repository Analysis ✓
- Repository Manager (clone/access)
- Repository Scanner (find Python files)
- AST Parser (extract structural info)

### Phase 2: Custom Dependency Engine
- Dependency Extractor (AST → relationships)
- Name/Module Resolver (imports → files)
- Dependency Records

### Phase 3: Custom Graph Engine
- Node, Edge, Graph data structures
- Graph Builder
- Graph Updater

### Phase 4: Change Analysis
- Git Diff Analysis
- Change Analyzer
- AST Before/After Comparison
- Graph Before/After Comparison

### Phase 5: Impact Engine
- Changed Components Detection
- Reverse Graph Traversal
- Affected Components
- Impact Propagation
- Impact Scoring

### Phase 6: Dataset Construction
- Historical PR Processing
- Feature Extraction
- Label Generation
- Dataset Building

### Phase 7: Multi-Language Architecture Foundation ✓ (v0.7.0)
- Extensible multi-language architecture with Python currently implemented as the first language adapter.
- `LanguageAdapter` Protocol — language-neutral interface for parse/extract/resolve.
- `ParsedSource`, `Symbol`, `Dependency` — language-neutral intermediate representations.
- `LanguageDetector` — extension-based language detection for .py, .java, .js, .ts, .go, .c, .cpp, etc.
- `LanguageRegistry` — central adapter registry with `default_registry` pre-loaded with Python.
- `PythonLanguageAdapter` — wraps Phase 1–3 pipeline (ASTParser + DependencyExtractor + NameResolver).
- **Language status**: Python = detected + fully supported. Java, JavaScript, TypeScript, Go, C, C++ = detected only; adapters planned for future phases.

### Phase 8: ML Risk Engine
- CatBoost Training
- Risk Prediction
- SHAP Explainability

### Phase 9: Autonomous Repair
- LLM Client
- Code Analysis
- Repair Generation
- Repair Controller with Retry Logic

### Phase 10: CI/CD Integration
- GitHub Actions Pipeline
- End-to-End Integration

### Phase 11: Evaluation
- Dependency Extraction Metrics
- Change Detection Accuracy
- Impact Analysis Evaluation
- Risk Model Performance
- Autonomous Repair Success Rate
- System Performance Benchmarks

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_ast_parser.py -v

# Run with coverage
pytest --cov=analyzer --cov=graph_engine tests/
```

## Autonomous repair and validation workflow

The backend exposes a proposal-safe delivery workflow under `/api/v1/workflow/`:

- `POST /analyze` runs the existing change analyzer, graph impact analysis, risk boundary, and SHAP boundary.
- `POST /validate` adds project-detected build, unit, regression, integration, security, and dependency checks.
- `POST /repair` requests a structured repair proposal and regression-test proposal.
- `POST /simulate-deployment` combines the evidence into a deterministic deployment simulation.

`RepairService` depends on the provider-neutral `LLMProvider` contract. A future Qwen2.5-Coder, Llama 3.3, Ollama, or other adapter can implement that contract without changing workflow logic. Unless `LLM_PROVIDER` is configured and an adapter is installed, repair and test generation return `unavailable`; no patch or test content is fabricated. Proposed patches are data only and never modify the original repository.

Validation uses trusted commands detected from project manifests (`pyproject.toml`, `package.json`, `.sln`/`.csproj`, `Cargo.toml`, and `CMakeLists.txt`). Commands run without a shell, through an allowlist, with bounded time and output. Package installation and dependency upgrades are never automatic. Missing scanners are reported as `UNAVAILABLE`, not passed.

Deployment is simulation-only. Security/build/test blocking signals produce `BLOCK`; incomplete or model-unavailable evidence is explicitly qualified and may produce `DELAY`; `ROLLBACK` is reserved for post-deployment regression evidence. CatBoost and SHAP remain unavailable until real artifacts are supplied.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```
@software{ai_change_impact_2026,
  title = {AI-Powered Autonomous Change Impact Analysis & Software Delivery Platform},
  author = {Arshiya},
  year = {2026},
  url = {https://github.com/your-repo/ai-change-impact-platform}
}
```
