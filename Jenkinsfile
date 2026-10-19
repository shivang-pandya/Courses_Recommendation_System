pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        PYTHONUNBUFFERED = '1'
        VENV_DIR = '.venv-ci'
        IMAGE_NAME = 'course-recommender'
        CONTAINER_NAME = 'course-recommender-ci'
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
                    "%VENV_DIR%\\Scripts\\python.exe" -m pip install --no-cache-dir -r requirements.txt
                '''
            }
        }

        stage('Compile Python sources') {
            steps {
                bat '''
                    "%VENV_DIR%\\Scripts\\python.exe" -m compileall -q app.py preprocess.py rl_environment.py dqn_agent.py tests
                '''
            }
        }

        stage('Run tests') {
            steps {
                bat '''
                    "%VENV_DIR%\\Scripts\\python.exe" -m unittest discover -s tests -p "test_*.py" -v
                '''
            }
        }

        stage('Build Docker image') {
            steps {
                bat 'docker build --tag "%IMAGE_NAME%:%BUILD_NUMBER%" .'
            }
        }

        stage('Container smoke test') {
            steps {
                powershell '''
                    docker run --detach --name "$env:CONTAINER_NAME" --publish 5000:5000 "$env:IMAGE_NAME`:$env:BUILD_NUMBER"
                    try {
                        for ($attempt = 1; $attempt -le 30; $attempt++) {
                            try {
                                $response = Invoke-WebRequest -Uri "http://127.0.0.1:5000/health" -UseBasicParsing -TimeoutSec 2
                                if ($response.StatusCode -eq 200) {
                                    exit 0
                                }
                            } catch {
                                Start-Sleep -Seconds 2
                            }
                        }
                        throw "Container health check failed"
                    } finally {
                        docker logs $env:CONTAINER_NAME
                        docker rm --force $env:CONTAINER_NAME
                    }
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'test-results/*.xml'
            powershell '''
                docker rm --force "$env:CONTAINER_NAME" 2>$null
                docker image rm "$env:IMAGE_NAME`:$env:BUILD_NUMBER" 2>$null
                if (Test-Path $env:VENV_DIR) {
                    Remove-Item -Recurse -Force $env:VENV_DIR
                }
            '''
        }
        success {
            echo 'CI and container checks passed. Deployment is intentionally not configured.'
        }
    }
}
