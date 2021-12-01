about:
	@echo "Docker maintenance commands"

devcontainer-build:
	docker build -t otaviog/stereo-matching-devcontainer:latest --target devcontainer .
