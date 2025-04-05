
all:
	pip3 install -e .
	cd mxgraph_component; make
	pip3 uninstall mxgraph_component -y
	pip3 install ./mxgraph_component/dist/mxgraph_component-0.0.1-py3-none-any.whl
	echo ""
	echo "Make sure dot is installed on your system"
	echo "If not, install it using 'sudo apt-get install graphviz' or similar"

start-dev:
	cd mxgraph_component/mxgraph_component/frontend; npm run start



lightsail:
	docker build --platform linux/amd64 -t flowco-app .
	aws lightsail push-container-image --region us-east-1 --service-name flowco-service
 --label flowco-app --image flowco-app:latest

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

