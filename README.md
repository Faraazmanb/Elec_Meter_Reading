
# **Meter Reading Detection with YOLO**

This project provides a containerized solution for meter reading detection using YOLO models. You can run the model using Docker, with either a pre-trained model or a local image file.

## **Getting Started**

### **Prerequisites**

1. Docker should be installed on your machine. You can download Docker from [here](https://www.docker.com/products/docker-desktop).
2. The Docker image `mafaz03/metermodel` must be pulled to run the application.

### **Pull the Docker Image**

To get started, first pull the image from Docker Hub:

```bash
docker pull mafaz03/metermodel
```

This will download the `metermodel` image containing the application and pre-trained models.

## **Usage**

### **1. Run with Pre-trained Models**

Once the image is pulled, you can run the model using the `last` or `best` pre-trained models.

- **For the 'last' model:**
  
  ```bash
  docker run metermodel python app.py a.jpg last
  ```

- **For the 'best' model:**

  ```bash
  docker run metermodel python app.py a.jpg best
  ```

Replace `a.jpg` with the name of the image file you want to process.

### **2. Run Using Local Image Files**

To run the model with your own local image files, mount the folder containing the images into the Docker container using the `-v` flag.

- **Run with the 'last' model using a local image:**

  ```bash
  docker run -v <path_to_local_folder>:/app/data metermodel python app.py /app/data/<file_name> last
  ```

- **Run with the 'best' model using a local image:**

  ```bash
  docker run -v <path_to_local_folder>:/app/data metermodel python app.py /app/data/<file_name> best
  ```

Replace the following placeholders:
- `<path_to_local_folder>`: The absolute path to the folder containing the image you want to use (e.g., `/Users/youruser/images/`).
- `<file_name>`: The name of the image file (e.g., `image.jpg`).

### **3. Understanding the Models**

- **'last' model**: Refers to the most recently trained model.
- **'best' model**: Refers to the best-performing model based on validation or evaluation metrics.

## **Example Flow**

1. **Pull the image**:
   ```bash
   docker pull mafaz03/metermodel
   ```

2. **Run with a local image**:
   ```bash
   docker run -v /Users/youruser/images:/app/data metermodel python app.py /app/data/sample_image.jpg best
   ```

This will process the `sample_image.jpg` file in the `/Users/youruser/images` directory using the best-performing model.

## **Additional Notes**

- The application uses **YOLO** models to detect meter readings from images.
- You can use either the `last` or `best` model for your predictions depending on which one you believe will perform better.
- The **volume mounting** (`-v` option) is used to allow you to use files from your local system in the Docker container without copying them directly into the container.

## **Troubleshooting**

- **Error: 'No such file or directory'**
  - Ensure that you provide the correct path to the image file and that the file exists in the specified directory.

- **Error: 'Model not found'**
  - Ensure that the image name (e.g., `last`, `best`) is specified correctly.
