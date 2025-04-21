# Criminal Procedure Assistant


## Getting Started

This repository implements a **Criminal Procedure Assistant**: a web application that lets you query the Massachusetts Rules of Criminal Procedure. It supports:
  - Retrieval-Augmented Generation (RAG) for free-form legal questions
  - A metrics dashboard (requests, latency, token usage)

### Prerequisites
  - Python 3.10+
  - A Pinecone account with an index named `criminalbot`
  - An OpenAI API key

### Installation
```bash
git clone <your-repo-url>
cd <your-repo-directory>
python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
```

### Configuration
Create a `.env` file in the project root:
```ini
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Indexing Documents
```bash
# Loads PDFs from Data/ and upserts embeddings into Pinecone
python store_index.py
```

### Running Locally
```bash
python app.py
```
Open these URLs in your browser:
  - http://localhost:8080/ (chat UI)
  - http://localhost:8080/metrics (metrics dashboard)

### Running with Docker
```bash
docker build -t criminalbot .
docker run --env-file .env -p 8080:8080 criminalbot
```

### Tech Stack
- Python 3.10
- Flask
- LangChain
- Pinecone
- OpenAI GPT-4
- Docker


## AWS CI/CD Deployment with GitHub Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 970547337635.dkr.ecr.ap-south-1.amazonaws.com/criminalbot

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - PINECONE_API_KEY
   - OPENAI_API_KEY

## Access the AWS Deployed App

You can access the AWS deployed app at: http://3.20.132.114:8080
    

