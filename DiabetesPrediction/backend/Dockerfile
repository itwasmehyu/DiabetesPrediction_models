FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY app.py diabetes_model.h5 scaler.pkl knn_model.pkl svm_linear_model.pkl svm_rbf_model.pkl logreg_model.pkl .
EXPOSE 5000

CMD ["python", "app.py"]