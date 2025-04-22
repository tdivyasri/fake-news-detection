pipeline {
    agent any

    environment {
        PROJECT_DIR = 'jupyter-docker-project'
    }

    triggers {
        pollSCM('* * * * *')  // Optional: Polls Git repo every minute for changes
    }

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/your-username/your-repo.git'
            }
        }

        stage('Build Docker Image with Compose') {
            steps {
                dir("${env.PROJECT_DIR}") {
                    sh 'docker-compose build'
                }
            }
        }

        stage('Run Docker Container with Compose') {
            steps {
                dir("${env.PROJECT_DIR}") {
                    sh 'docker-compose up -d'
                }
            }
        }
    }
}
