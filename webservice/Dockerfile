FROM python:3.10-slim

WORKDIR /app

# Match your model's exact scikit-learn version
RUN pip install --no-cache-dir \
    flask \
    requests \
    joblib \
    "numpy<2.0.0" \
    scikit-learn==1.0.2

# Copy code and model
COPY predict.py .
COPY RandomForestRegressor_model.bin .

CMD ["python", "predict.py"]