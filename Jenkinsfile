pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        PYTHONUNBUFFERED = '1'
        VENV_DIR = '.venv-ci'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Create virtual environment') {
            steps {
                bat 'python -m venv "%VENV_DIR%"'
            }
        }

        stage('Install dependencies') {
            steps {
                bat '''
                    "%VENV_DIR%\\Scripts\\python.exe" -m pip install --upgrade --no-cache-dir pip
                    "%VENV_DIR%\\Scripts\\python.exe" -m pip install --no-cache-dir -r requirements-ci.txt
                '''
            }
        }

        stage('Compile Python sources') {
            steps {
                bat '''
                    "%VENV_DIR%\\Scripts\\python.exe" -m compileall -q app.py preprocess.py rl_environment.py dqn_agent.py train.py tests
                '''
            }
        }

        stage('Run tests') {
            steps {
                bat '''
                    "%VENV_DIR%\\Scripts\\python.exe" -m unittest discover -s tests -p "test_ci_source.py" -v
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'test-results/*.xml'
            powershell '''
                if (Test-Path $env:VENV_DIR) {
                    Remove-Item -Recurse -Force $env:VENV_DIR
                }
            '''
        }
        success {
            echo 'Lightweight CI checks passed. Docker build and deployment are intentionally separate.'
        }
    }
}
