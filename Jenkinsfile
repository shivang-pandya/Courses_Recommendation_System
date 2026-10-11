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
                sh 'python3 -m venv "$VENV_DIR"'
            }
        }

        stage('Install dependencies') {
            steps {
                sh '''
                    . "$VENV_DIR/bin/activate"
                    python -m pip install --upgrade pip
                    python -m pip install -r requirements.txt
                '''
            }
        }

        stage('Compile Python sources') {
            steps {
                sh '''
                    . "$VENV_DIR/bin/activate"
                    python -m compileall -q app.py preprocess.py rl_environment.py dqn_agent.py train.py tests
                '''
            }
        }

        stage('Run tests') {
            steps {
                sh '''
                    . "$VENV_DIR/bin/activate"
                    python -m unittest discover -s tests -p "test_*.py" -v
                '''
            }
        }

        stage('Build Docker image') {
            steps {
                sh 'docker build --tag "$IMAGE_NAME:$BUILD_NUMBER" .'
            }
        }

        stage('Container smoke test') {
            steps {
                sh '''
                    docker run --detach --name "$CONTAINER_NAME" --publish 5000:5000 "$IMAGE_NAME:$BUILD_NUMBER"
                    trap 'docker logs "$CONTAINER_NAME" || true; docker rm --force "$CONTAINER_NAME" || true' EXIT

                    for attempt in $(seq 1 30); do
                        if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2)" 2>/dev/null; then
                            exit 0
                        fi
                        sleep 2
                    done

                    echo 'Container health check failed'
                    exit 1
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'test-results/*.xml'
            sh '''
                docker rm --force "$CONTAINER_NAME" 2>/dev/null || true
                docker image rm "$IMAGE_NAME:$BUILD_NUMBER" 2>/dev/null || true
                rm -rf "$VENV_DIR"
            '''
        }
        success {
            echo 'CI and container checks passed. Deployment is intentionally not configured.'
        }
    }
}
