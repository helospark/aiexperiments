#!/bin/bash

# Remove existing pi_docker image if it exists
if docker images pi_docker --format "{{.Repository}}:{{.Tag}}" | grep -q "pi_docker"; then
  echo "Removing existing pi_docker image..."
  docker rmi pi_docker
fi

# Build Docker image and tag it as pi_docker
echo "Building Docker image..."
docker build -t pi_docker .

# Create required directories
echo "Creating directory structure..."
mkdir -p /home/$USER/pi/pi-agent-home

# Set proper permissions for the agent home directory to allow access by node user
chown -R $USER:$USER /home/$USER/pi/pi-agent-home
chmod -R 755 /home/$USER/pi/pi-agent-home

# Copy models.json into the created directory
if [ -f ./pi-agent-home/models.json ]; then
  echo "Copying models.json..."
  cp ./pi-agent-home/models.json /home/$USER/pi/pi-agent-home/
else
  echo "Error: models.json not found in ./pi-agent-home/"
  exit 1
fi

# Copy pidocker script to /usr/local/bin for easy access
if [ -f ./pidocker ]; then
  echo "Copying pidocker to /usr/local/bin..."
  sudo cp ./pidocker /usr/local/bin/pidocker
  sudo chmod +x /usr/local/bin/pidocker
  echo "pidocker is now available system-wide as 'pidocker'"
else
  echo "Warning: pidocker script not found in current directory"
fi

echo "Installation complete!"
