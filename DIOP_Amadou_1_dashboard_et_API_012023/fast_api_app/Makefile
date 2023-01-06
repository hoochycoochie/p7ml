network:
	docker network inspect streamlit_network >/dev/null 2>&1 || \
    docker network create --driver bridge streamlit_network
up:
	docker container rm -f fastapi&& docker image rm -f fastapi_img:latest &&docker build --no-cache   -t fastapi_img . && docker container run -d --name fastapi -p 8000:8000 --network streamlit_network  fastapi_img 


down:
	docker container rm -f fastapi&& docker image rm -f fastapi_img

start:
	docker container run -p 8000:8000 --name fastapi fastapi_img