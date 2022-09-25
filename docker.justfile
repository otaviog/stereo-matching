about:
	@echo "Docker maintenance commands"

basecontainer-build:
	docker build -t otaviog/stereo-matching-base:latest --target base .

devcontainer-build:
	docker build -t otaviog/stereo-matching-devcontainer:latest --target devcontainer .
