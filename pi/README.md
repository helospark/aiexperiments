# Pi Docker Setup

This repository contains the necessary files to build and run the Pi AI harness agent in a Docker environment and use current directory as workspace.

The docker helps separate your system from the pi harness, so something going wrong cannot accidentally harm your system.

## Prerequisites

- Docker installed on your system
- Bash shell

## Installation

Run the installation script to build the Docker image and set up the required directory structure:

```bash
./install.sh
```

This will:
1. Build a Docker image tagged as `pi_docker`
2. Create the directory `/home/$USER/pi/pi-agent-home`
3. Copy `models.json` into the created directory
4. Copy the pidocker script to your bin folder

You can edit the `/home/$USER/pi/pi-agent-home/models.json` to connect to the LLM of your choice.

## Usage

After installation, you can use the Pi agent by running:

```bash
pidocker
```

this will attach the current folder as workspace.

## Directory Structure

- **Dockerfile**: Defines the Docker image build process
- **install.sh**: Installs and sets up the environment 
- **pidocker**: Script to run the Pi agent in Docker
- **pi-agent-home/**: Contains configuration files (including models.json)

