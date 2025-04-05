
all: build install

.PHONY: all build install uninstall run-dev-server clean

build:
	# change the RELEASE variable in mxgraph_component/frontend/webpack.config.js to true!

	# install dependencies, create build directory, copy assets
	cd mxgraph_component/frontend; npm install;
	cd mxgraph_component/frontend; npm run build

	# install dependencies, create wheel
	python setup.py sdist bdist_wheel

install: uninstall build
	# install wheel
	pip3 install dist/mxgraph_component-0.0.1-py3-none-any.whl

uninstall:
	pip3 uninstall mxgraph_component -y

run-dev-server:
	# change the RELEASE variable in mxgraph_component/frontend/webpack.config.js to false!
	cd mxgraph_component/frontend; npm install;
	cd mxgraph_component/frontend; npm run start


clean: uninstall
	rm -rf dist
	rm -rf build
	rm -rf mxgraph_component.egg-info
	rm -rf mxgraph_component/frontend/node_modules
	rm -rf mxgraph_component/frontend/dist
	rm -rf mxgraph_component/frontend/.cache
	rm -rf mxgraph_component/frontend/package-lock.json
