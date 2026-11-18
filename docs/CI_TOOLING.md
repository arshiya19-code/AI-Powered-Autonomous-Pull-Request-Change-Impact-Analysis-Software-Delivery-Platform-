# CI tooling and local startup

The approved repair workflow is retained. `CI_EXECUTOR=local` uses backend subprocess validation; `CI_EXECUTOR=jenkins` requires Jenkins and blocks a candidate if orchestration fails. There is no silent executor fallback. Jenkins executes checks; the backend owns analysis/candidate IDs, attempt, aggregation, deterministic re-analysis, selection, approval, deployment simulation, and learning.

## Start the application reliably (PowerShell)

From `C:\Users\Arshiya\AI-Change-Impact-Platform`:

```powershell
# This task created .venv-ci because the existing .venv refers to a missing Python 3.13.
# For a fresh machine with Python 3.12 installed:
py -3.12 -m venv .venv-ci
.\.venv-ci\Scripts\python.exe -m pip install -r requirements.txt
# Copy only when .env does not already exist:
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
.\scripts\start-local.ps1
```

In a second terminal:

```powershell
.\scripts\start-local.ps1 -Frontend
```

The backend loads the project-root `.env` before importing database, GitHub, provider, and CORS settings. Existing process variables take precedence. Restart the backend after editing `.env`; if an old process variable overrides the file, remove that variable from the terminal before restarting. The startup script never displays credentials. Set GitHub App ID, slug, private-key **path**, webhook secret, frontend origins, database URL, and LLM provider/model/base URL/API key in the file. Keep `COOKIE_SECURE=false` for local HTTP only. The example uses the existing accounts database. The private key is not read or copied by this task.

## Jenkins local setup

No Jenkins installation, service, process, Java executable, or configured credentials were detected during this task. TCP probes to localhost:8080 were not reachable. No Jenkins build was run.

Install a supported Java runtime and download the official Jenkins LTS WAR. Java 21 is a supported baseline; consult the current [Jenkins Java policy](https://www.jenkins.io/doc/book/platform-information/support-policy-java/) when choosing the release.

```powershell
winget install --exact --id EclipseAdoptium.Temurin.21.JDK
# Open a new terminal after installation.
New-Item -ItemType Directory -Force .tmp/jenkins | Out-Null
Invoke-WebRequest https://get.jenkins.io/war-stable/latest/jenkins.war -OutFile .tmp/jenkins/jenkins.war
java -jar .tmp/jenkins/jenkins.war --httpPort=8080 --httpListenAddress=127.0.0.1
```

Complete the first-run setup at http://localhost:8080 and install Pipeline and Pipeline: Basic Steps. Create a Pipeline job named `candidate-validation`, with the trusted platform `Jenkinsfile` as its script (or Pipeline script from the platform repository). Do not use the incoming PR's Jenkinsfile. Run the job once to register its two string parameters; with no bundle it intentionally fails preparation and performs no candidate validation.

Configure a dedicated local agent, label `candidate-validation`, with Python/project dependencies, Git, scanner executables and repository-specific test dependencies. Use a disposable agent for untrusted PR code. Set these **agent** environment values:

```text
PLATFORM_ROOT=C:/Users/Arshiya/AI-Change-Impact-Platform
CI_PYTHON=C:/Users/Arshiya/AI-Change-Impact-Platform/.venv-ci/Scripts/python.exe
CI_SHARED_ROOT=C:/Users/Arshiya/AI-Change-Impact-Platform/.ci-bundles
```

Create that shared directory. The backend and agent must see it at the same absolute path (this adapter's supported local/shared-filesystem topology). Grant only those service accounts access. Use Jenkins Credentials Binding for scanner tokens on the dedicated agent, or a separately protected agent environment file; never put credential values in the Jenkinsfile. `PLATFORM_ROOT/.env` is loaded by the worker on a same-machine development installation. The subprocess runner removes service secrets before running repository compilation and tests.

Create an API token for a Jenkins user permitted to read/build this job and read its artifacts. In backend `.env` set:

```dotenv
CI_EXECUTOR=jenkins
JENKINS_URL=http://127.0.0.1:8080
JENKINS_JOB=candidate-validation
JENKINS_USER=<local Jenkins username>
JENKINS_TOKEN=<API token, not login password>
JENKINS_TIMEOUT=1200
CI_SHARED_ROOT=C:/Users/Arshiya/AI-Change-Impact-Platform/.ci-bundles
```

The adapter uses authenticated `buildWithParameters`, follows its exact queue/build identity, and downloads that build's `ci-result.json`. API-token requests do not require a CSRF crumb. [Jenkins Remote Access API](https://www.jenkins.io/doc/book/using/remote-access-api/).

States: READY means API/job configuration is reachable; NOT CONFIGURED means settings are missing; UNREACHABLE means the configured API request failed; RUNNING means queued/building or an external build outlived the backend wait; PASS/FAIL describe ingested validation evidence. A timeout blocks the candidate. It does not assert that the external job stopped. The adapter never sends credentials to queue/build URLs on another host.

Stages, in order:

1. Prepare Candidate
2. Python Compilation Check
3. Existing PyTest Suite
4. AI Regression PyTest Suite
5. AI Targeted Integration PyTest Suite
6. Bandit Python Security
7. SonarQube Analysis
8. Snyk Security/Dependency Scan
9. Dependency Manifest Validation
10. Candidate Re-analysis / Gate

Each execution stage writes its independent check. Test failures are evidence, so later stages can still produce evidence. Jenkins build SUCCESS means the executor completed, **not** that the candidate passed. The final stage hands evidence back to the waiting backend; the existing backend workflow performs deterministic re-analysis and decisions. Missing stages or mismatched analysis ID, candidate ID, attempt, SHAs, or source fingerprint are rejected. Shared bundles omit .env files, PEM/key files, secret-named files and repository caches. Retain bundles only as long as needed for diagnostics; no cleanup deletes originals.

## SonarQube

No SonarScanner or Sonar credentials were detected; localhost:9000 was not reachable. No SonarQube scan was run.

For local evaluation with Docker Desktop running:

```powershell
docker run -d --name impact-sonarqube -p 127.0.0.1:9000:9000 sonarqube:community
```

Use [Sonar's container instructions](https://docs.sonarsource.com/sonarqube-community-build/server-installation/from-docker-image/set-up-and-start-container) for durable deployment. Sign in locally, change the initial password, and create a user token with Execute Analysis, Browse, and Create Projects permissions. The adapter uses a unique project key per candidate to avoid mixing metrics from concurrent analyses. Arrange project retention in SonarQube.

Download and extract the Windows CLI from the official [SonarScanner CLI page](https://docs.sonarsource.com/sonarqube-server/analyzing-source-code/scanners/sonarscanner). Add its `bin` directory to the backend and Jenkins agent PATH and restart both. Verify `sonar-scanner --version`. Set `SONAR_HOST_URL=http://127.0.0.1:9000` and `SONAR_TOKEN` privately in `.env` / agent credentials. [Token configuration](https://docs.sonarsource.com/sonarqube-community-build/user-guide/managing-tokens).

The scanner waits for processing, then the backend reads the exact analysis quality gate via the task ID. Persisted metrics are bugs, vulnerabilities, security hotspots, code smells, and coverage **only if returned**. Existing `coverage.xml` is supplied only if present. An ERROR quality gate blocks validation; scan/auth/network/evidence failures never pass. Missing configuration is displayed as NOT CONFIGURED. The unique candidate project works without assuming commercial PR-analysis support.

## Snyk

No Snyk executable or token was detected. No Snyk scan was run.

```powershell
npm.cmd install -g snyk
snyk --version
snyk auth
```

Interactive authentication is supported. For backend and Jenkins service accounts, prefer a private `SNYK_TOKEN` entry so credentials survive terminal closure. [Snyk authentication](https://docs.snyk.io/snyk-cli/authenticate-to-use-the-cli). Restart services after updating their PATH/configuration.

The adapter executes `snyk test --all-projects --json --ignore-policy` against the original and candidate. Install each repository's dependencies on the agent as required by its ecosystem before scanning. [Snyk test reference](https://docs.snyk.io/snyk-cli/commands/test). Exit 0/1 is parsed; operational failures and malformed/missing reports block. Findings are matched by manifest, vulnerability ID, package, version and dependency path. Critical/High findings newly introduced (including severity escalation) block; existing findings and Medium/Low remain visible. Missing baseline evidence cannot pass candidate Snyk validation. pip-audit remains separate optional evidence and is never labeled Snyk.

## PR #2 and repair evidence

The GitHub analysis worker obtains `pull_request.base.sha` and `pull_request.head.sha` from the API, persists both in analysis metadata, fetches both objects and computes `git diff base_sha head_sha`. Repair reacquisition uses those persisted SHAs, not the mutable synchronized PR row. Thus `demo-risk-change` → `demo-live-ai-repair` is supported without main/master/default-branch substitution.

The API title/body are persisted in analysis metadata. `DeliveryWorkflowService.repair_context` passes `pr_title`, `pr_body`, diff, changed symbols/files, impacted symbols, propagation paths, critical nodes, impact/risk/SHAP, decision, and validation to the provider. The repair instruction explicitly says passing existing tests does not establish correctness. Test generation gets the same PR evidence. Failed generated tests and prior patch/explanation are included on retries. PR #2's body must contain the actual expected `50% = 0.5` and defective `50% = 5000` evidence; the platform does not invent it. Older analyses lacking this metadata must be re-analyzed.

No live Qwen repair for PR #2 was requested or run; focused tests capture the outgoing HTTP payload to verify the actual evidence reaches the provider. No PR #2 was created.

## Verification

```powershell
.\.venv-ci\Scripts\python.exe -m pytest tests/test_ci_tooling.py tests/test_ai_repair_validation_loop.py tests/test_github_integration.py tests/test_live_repair_completion.py tests/test_provider_response_robustness.py -q
npm.cmd --prefix frontend run lint
npm.cmd --prefix frontend run build
.\.venv-ci\Scripts\python.exe -c "from backend.app.services.ci_tools import tool_readiness; import json; print(json.dumps(tool_readiness(), indent=2))"
```

Readiness appears in the existing dashboard health card; individual scan results and severity/quality metrics remain separate. `.env`, `.env.*` except `.env.example`, `*.pem`, `*.key`, secret directories/files, candidate bundles and the local test environment are ignored. This supplied workspace has no `.git` metadata, so ignored patterns can be checked but already-tracked-secret status cannot be established here.

## Results recorded for this implementation

41 focused CI-tooling tests and 40 existing backend regression tests passed across the final runs. Frontend typecheck and production build passed. Python dependency consistency check and startup PowerShell syntax check passed. Only pre-existing deprecation warnings remain. Live Jenkins, SonarQube and Snyk execution remains unverified because these tools are NOT CONFIGURED on this machine. PyTest and Bandit are READY in .venv-ci.
