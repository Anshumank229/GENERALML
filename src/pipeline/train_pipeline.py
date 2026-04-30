import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING COMPLETE TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Data Ingestion
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print("✅ Data Ingestion Complete")
    
    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_path, test_path
    )
    print("✅ Data Transformation Complete")
    
    # Step 3: Model Training
    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"✅ Model Training Complete! Best R² Score: {r2_score:.4f}")
    
    # Verify files
    print("\n📁 Artifacts created:")
    print(f"   preprocessor.pkl: {os.path.exists('artifacts/preprocessor.pkl')}")
    print(f"   model.pkl: {os.path.exists('artifacts/model.pkl')}")