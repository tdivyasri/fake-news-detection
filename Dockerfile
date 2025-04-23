# Base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files to the container (including the datasets)
COPY . .

# Install required Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install scikit-learn explicitly to avoid missing dependencies
RUN pip install scikit-learn

# Install scikit-learn-intelex separately
RUN pip install scikit-learn-intelex

# Verify the installation of scikit-learn-intelex
RUN pip show scikit-learn-intelex

# Install joblib
RUN pip install joblib

# Ensure the datasets are copied into the container (if required separately)
COPY True.csv .
COPY Fake.csv .
COPY Final_Prepared_Dataset.csv .
COPY data/IFND.csv /data/IFND.csv

# Expose port 5000 if you're planning to communicate via that port (optional)
EXPOSE 5000

# Run the script
CMD ["python", "Fake_News_Detection.py"]
