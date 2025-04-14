
all:
	cd src/mxgraph_component/mxgraph_component/frontend; npm install 
	cd src/mxgraph_component/mxgraph_component/frontend; npm run build 
	cd src/mxgraph_component/; python3 setup.py sdist bdist_wheel 
	pip3 install src/mxgraph_component/dist/mxgraph_component-0.0.1-py3-none-any.whl -e .
	echo ""
	echo ""
	echo ""
	echo "Make sure dot is installed on your system"
	echo "If not, install it using 'sudo apt-get install graphviz' or similar"

start-dev:
	cd src/mxgraph_component/mxgraph_component/frontend; npm run start



# docker-image:
# 	docker build --platform linux/amd64 -t flowco-app .

# docker-run:
# 	docker run -p 8501:8501 flowco-app

# aws-auth-docker:
# 	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 925527669208.dkr.ecr.us-east-1.amazonaws.com

# aws-tag-and-push:
# 	docker tag flowco-app:latest 925527669208.dkr.ecr.us-east-1.amazonaws.com/flowco-app:latest
# 	docker push 925527669208.dkr.ecr.us-east-1.amazonaws.com/flowco-app:latest

# docker-deploy: docker-image aws-auth-docker aws-tag-and-push
# 	echo "Boop"

# encode-secrets:
# 	openssl base64 -A -in .streamlit/secrets.toml -out secrets.toml.b64

