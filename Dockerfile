# Use the official PyTorch TorchServe base image
FROM pytorch/torchserve:latest

# Create a directory for the model store
RUN mkdir -p /home/model-server/model-store/

# Copy the MAR file into the model store directory
COPY model_store/model.mar /home/model-server/model-store/

# Create a custom configuration file for TorchServe
COPY config.properties /home/model-server/config.properties

# Set environment variables for TorchServe
ENV MODEL_STORE=/home/model-server/model-store
ENV TS_CONFIG_FILE=/home/model-server/config.properties

# Expose the default ports for TorchServe
EXPOSE 8080 8081

# Define the entry point for the container
CMD ["torchserve", "--start", "--model-store", "/home/model-server/model-store", "--models", "model=model.mar"]
