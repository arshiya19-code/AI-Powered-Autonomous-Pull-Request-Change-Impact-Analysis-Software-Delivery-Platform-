// Configure a trusted agent with PLATFORM_ROOT, CI_PYTHON, CI_SHARED_ROOT and
// scanner credentials. This Jenkinsfile belongs to the platform, never the PR.
// Backend is authoritative for candidate gates, selection, approval and learning.
def executeStage(String name) {
    if (isUnix()) {
        sh "\"\$CI_PYTHON\" -m scripts.jenkins_candidate ${name}"
    } else {
        bat "@\"%CI_PYTHON%\" -m scripts.jenkins_candidate ${name}"
    }
}
pipeline {
    agent { label 'candidate-validation' }
    options { skipDefaultCheckout(true); timeout(time: 30, unit: 'MINUTES'); disableConcurrentBuilds() }
    parameters {
        string(name: 'BUNDLE_ID', defaultValue: '', description: 'Backend-issued source bundle UUID')
        string(name: 'BUNDLE_SHA256', defaultValue: '', description: 'Backend-issued immutable source fingerprint')
    }
    environment { PYTHONPATH = "${env.PLATFORM_ROOT}" }
    stages {
        stage('Prepare Candidate') { steps { deleteDir(); script { executeStage('prepare') } } }
        stage('Python Compilation Check') { steps { script { executeStage('build') } } }
        stage('Existing PyTest Suite') { steps { script { executeStage('unit') } } }
        stage('AI Regression PyTest Suite') { steps { script { executeStage('regression') } } }
        stage('AI Targeted Integration PyTest Suite') { steps { script { executeStage('targeted_integration') } } }
        stage('Bandit Python Security') { steps { script { executeStage('security') } } }
        stage('SonarQube Analysis') { steps { script { executeStage('sonarqube') } } }
        stage('Snyk Security/Dependency Scan') { steps { script { executeStage('snyk') } } }
        stage('Dependency Manifest Validation') { steps { script { executeStage('dependency') } } }
        stage('Candidate Re-analysis / Gate') {
            steps { script { executeStage('handoff') } }
            // Persist completed evidence. The waiting backend ingests it, applies
            // validation gates and runs its existing deterministic re-analysis.
        }
    }
    post { always { archiveArtifacts artifacts: 'ci-result.json', allowEmptyArchive: true, fingerprint: true } }
}
