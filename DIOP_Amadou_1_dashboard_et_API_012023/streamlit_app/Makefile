network:
	docker network inspect streamlit_network >/dev/null 2>&1 || \
    docker network create --driver bridge streamlit_network
up:
	docker container rm -f streamlit&& docker image rm -f streamlit_img:latest &&docker build --no-cache   -t streamlit_img . && docker container run -d --name streamlit -p 8501:8501 --network streamlit_network streamlit_img 


down:
	docker container rm -f streamlit&& docker image rm -f streamlit_img