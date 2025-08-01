import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import joblib

warnings.filterwarnings('ignore')

class ProductionHousePriceModel:
    def __init__(self):
        self.rent_model = None
        self.sale_model = None
        self.rent_scaler = None
        self.sale_scaler = None
        self.rent_features = None
        self.sale_features = None
        self.performance_metrics = {}
        
    def load_and_clean_data(self, file_path):
        """Load and clean dataset with validation."""
        print("Loading dataset...")
        df = pd.read_csv(r"zameen-updated.csv")
        
        # Basic cleaning
        numeric_cols = ['price', 'Area Size', 'bedrooms', 'baths', 'latitude', 'longitude']
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = ['property_type', 'city', 'purpose']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove invalid entries
        df = df[df['price'] > 0]
        df = df[df['Area Size'] > 0]
        
        # Remove extreme outliers by purpose
        for purpose in ['For Rent', 'For Sale']:
            purpose_data = df[df['purpose'] == purpose]
            if len(purpose_data) > 0:
                upper_limit = purpose_data['price'].quantile(0.95)
                lower_limit = purpose_data['price'].quantile(0.05)
                df = df[~((df['purpose'] == purpose) & 
                         ((df['price'] > upper_limit) | (df['price'] < lower_limit)))]
        
        print(f"Data cleaned. Final shape: {df.shape}")
        return df
    
    def engineer_features(self, df, purpose_type):
        """Engineer features for specific purpose (rent or sale)."""
        df_purpose = df[df['purpose'] == purpose_type].copy()
        
        if len(df_purpose) == 0:
            raise ValueError(f"No data found for purpose: {purpose_type}")
        
        # Basic features
        df_purpose['bedrooms_per_bathroom'] = np.where(
            df_purpose['baths'] == 0, 
            df_purpose['bedrooms'], 
            df_purpose['bedrooms'] / df_purpose['baths']
        )
        df_purpose['total_rooms'] = df_purpose['bedrooms'] + df_purpose['baths']
        df_purpose['price_per_sqft'] = df_purpose['price'] / df_purpose['Area Size']
        df_purpose['area_bedroom_ratio'] = df_purpose['Area Size'] / (df_purpose['bedrooms'] + 1)
        df_purpose['bed_bath_ratio'] = df_purpose['bedrooms'] / df_purpose['baths'].replace(0, 1)
        
        # Date features
        df_purpose['date_added'] = pd.to_datetime(df_purpose['date_added'], errors='coerce')
        df_purpose['listing_year'] = df_purpose['date_added'].dt.year
        df_purpose['listing_year'] = df_purpose['listing_year'].fillna(df_purpose['listing_year'].median())
        
        # Location-based features
        df_purpose['city_median_price'] = df_purpose.groupby('city')['price'].transform('median')
        df_purpose['city_mean_price'] = df_purpose.groupby('city')['price'].transform('mean')
        
        # Numerical features
        numerical_features = [
            'Area Size', 'bedrooms', 'baths', 'latitude', 'longitude',
            'listing_year', 'bedrooms_per_bathroom', 'total_rooms',
            'area_bedroom_ratio', 'bed_bath_ratio', 'city_median_price', 'city_mean_price'
        ]
        
        # Handle infinite values
        for col in numerical_features:
            df_purpose[col] = df_purpose[col].replace([np.inf, -np.inf], np.nan)
            df_purpose[col] = df_purpose[col].fillna(df_purpose[col].median())
        
        # Categorical features
        categorical_features = ['property_type', 'city']
        for cat in categorical_features:
            top_categories = df_purpose[cat].value_counts().head(8).index
            df_purpose[cat] = df_purpose[cat].where(df_purpose[cat].isin(top_categories), 'Other')
        
        # One-hot encoding
        encoded_features = pd.get_dummies(df_purpose[categorical_features], drop_first=True)
        X = pd.concat([df_purpose[numerical_features], encoded_features], axis=1)
        Y = df_purpose['price']
        
        return X, Y
    
    def train_models(self, df):
        """Train separate models for rent and sale."""
        print("Training separate models for rent and sale...")
        
        # Train rent model
        print("Training rental price model...")
        X_rent, Y_rent = self.engineer_features(df, 'For Rent')
        self.rent_features = X_rent.columns.tolist()
        
        X_rent_train, X_rent_test, Y_rent_train, Y_rent_test = train_test_split(
            X_rent, Y_rent, test_size=0.2, random_state=42
        )
        
        self.rent_scaler = RobustScaler()
        X_rent_train_scaled = self.rent_scaler.fit_transform(X_rent_train)
        X_rent_test_scaled = self.rent_scaler.transform(X_rent_test)
        
        self.rent_model = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.08, max_depth=6,
            min_samples_leaf=10, random_state=42
        )
        self.rent_model.fit(X_rent_train_scaled, Y_rent_train)
        
        rent_pred = self.rent_model.predict(X_rent_test_scaled)
        self.performance_metrics['rent'] = {
            'r2': r2_score(Y_rent_test, rent_pred),
            'rmse': np.sqrt(mean_squared_error(Y_rent_test, rent_pred)),
            'mae': mean_absolute_error(Y_rent_test, rent_pred),
            'mape': np.mean(np.abs((Y_rent_test - rent_pred) / Y_rent_test)) * 100,
            'samples': len(X_rent)
        }
        
        # Train sale model
        print("Training sale price model...")
        X_sale, Y_sale = self.engineer_features(df, 'For Sale')
        self.sale_features = X_sale.columns.tolist()
        
        X_sale_train, X_sale_test, Y_sale_train, Y_sale_test = train_test_split(
            X_sale, Y_sale, test_size=0.2, random_state=42
        )
        
        self.sale_scaler = RobustScaler()
        X_sale_train_scaled = self.sale_scaler.fit_transform(X_sale_train)
        X_sale_test_scaled = self.sale_scaler.transform(X_sale_test)
        
        self.sale_model = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.08, max_depth=8,
            min_samples_leaf=20, random_state=42
        )
        self.sale_model.fit(X_sale_train_scaled, Y_sale_train)
        
        sale_pred = self.sale_model.predict(X_sale_test_scaled)
        self.performance_metrics['sale'] = {
            'r2': r2_score(Y_sale_test, sale_pred),
            'rmse': np.sqrt(mean_squared_error(Y_sale_test, sale_pred)),
            'mae': mean_absolute_error(Y_sale_test, sale_pred),
            'mape': np.mean(np.abs((Y_sale_test - sale_pred) / Y_sale_test)) * 100,
            'samples': len(X_sale)
        }
        
        print("Model training completed successfully")
    
    def display_performance(self):
        """Display model performance."""
        print("\n" + "="*70)
        print(" PRODUCTION-READY HOUSE PRICE PREDICTION MODEL")
        print("="*70)
        
        for model_type in ['rent', 'sale']:
            metrics = self.performance_metrics[model_type]
            model_name = "RENTAL MODEL" if model_type == 'rent' else "SALE MODEL"
            
            print(f"\n{model_name}:")
            print(f"  R² Score:       {metrics['r2']:.4f}")
            print(f"  RMSE:           {metrics['rmse']:,.0f} PKR")
            print(f"  MAE:            {metrics['mae']:,.0f} PKR")
            print(f"  MAPE:           {metrics['mape']:.2f}%")
            print(f"  Training Data:  {metrics['samples']:,} samples")
            
            # Quality assessment
            if metrics['r2'] > 0.80 and metrics['mape'] < 25:
                quality = "EXCELLENT - Production Ready"
            elif metrics['r2'] > 0.70 and metrics['mape'] < 35:
                quality = "GOOD - Ready for Deployment"  
            elif metrics['r2'] > 0.60:
                quality = "FAIR - Acceptable Performance"
            else:
                quality = "NEEDS IMPROVEMENT"
            
            print(f"  Quality:        {quality}")
    
    def predict_price(self, area_size, bedrooms, bathrooms, city, property_type, purpose, df_reference):
        """Make price prediction using appropriate model."""
        
        # Choose model and features based on purpose
        if purpose == 'For Rent':
            model = self.rent_model
            scaler = self.rent_scaler
            features = self.rent_features
        else:
            model = self.sale_model
            scaler = self.sale_scaler
            features = self.sale_features
        
        # Get reference data for the same purpose
        ref_data = df_reference[df_reference['purpose'] == purpose]
        city_data = ref_data[ref_data['city'] == city]
        if len(city_data) == 0:
            city_data = ref_data
        
        # Create feature vector
        prediction_data = {
            'Area Size': area_size,
            'bedrooms': bedrooms,
            'baths': bathrooms,
            'latitude': city_data['latitude'].median(),
            'longitude': city_data['longitude'].median(),
            'listing_year': 2024,
            'bedrooms_per_bathroom': bedrooms / max(bathrooms, 1),
            'total_rooms': bedrooms + bathrooms,
            'area_bedroom_ratio': area_size / (bedrooms + 1),
            'bed_bath_ratio': bedrooms / max(bathrooms, 1),
            'city_median_price': city_data['price'].median(),
            'city_mean_price': city_data['price'].mean()
        }
        
        # Initialize all features to 0
        for feature in features:
            if feature not in prediction_data:
                prediction_data[feature] = 0
        
        # Set categorical flags
        if f"city_{city}" in features:
            prediction_data[f"city_{city}"] = 1
        if f"property_type_{property_type}" in features:
            prediction_data[f"property_type_{property_type}"] = 1
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame([prediction_data])
        pred_df = pred_df.reindex(columns=features, fill_value=0)
        
        # Scale and predict
        pred_scaled = scaler.transform(pred_df)
        prediction = model.predict(pred_scaled)[0]
        
        return max(0, prediction)  # Ensure non-negative
    
    def save_models(self):
        """Save models and scalers."""
        joblib.dump(self.rent_model, 'rent_model.pkl')
        joblib.dump(self.sale_model, 'sale_model.pkl')
        joblib.dump(self.rent_scaler, 'rent_scaler.pkl')
        joblib.dump(self.sale_scaler, 'sale_scaler.pkl')
        
        metadata = {
            'rent_features': self.rent_features,
            'sale_features': self.sale_features,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(metadata, 'model_metadata.pkl')
        
        print("Models saved successfully")

def main():
    print(" === PRODUCTION-READY HOUSE PRICE PREDICTION === ")
    print("Separate models for accurate rent and sale price predictions\n")
    
    # Initialize model
    model = ProductionHousePriceModel()
    
    # Load and process data
    file_path = r'C:\Users\my pc\Downloads\House Price Analysis and Prediction\zameen-updated.csv'
    df = model.load_and_clean_data(file_path)
    
    # Train models
    model.train_models(df)
    
    # Display performance
    model.display_performance()
    
    # Save models
    model.save_models()
    
    # Interactive predictions
    print("\n" + "="*70)
    print(" INTERACTIVE PRICE PREDICTIONS")
    print("="*70)
    
    cities = df['city'].value_counts().head(5).index.tolist()
    prop_types = df['property_type'].value_counts().head(5).index.tolist()
    
    print(f"Available Cities: {', '.join(cities)}")
    print(f"Available Types: {', '.join(prop_types)}")
    
    # Demonstration of rent vs sale
    print("\n DEMONSTRATION: Same Property - Rent vs Sale Prices")
    print("-" * 50)
    
    demo_specs = {'area': 1200, 'beds': 3, 'baths': 2, 'city': 'Karachi', 'type': 'House'}
    
    rent_price = model.predict_price(
        demo_specs['area'], demo_specs['beds'], demo_specs['baths'],
        demo_specs['city'], demo_specs['type'], 'For Rent', df
    )
    
    sale_price = model.predict_price(
        demo_specs['area'], demo_specs['beds'], demo_specs['baths'],
        demo_specs['city'], demo_specs['type'], 'For Sale', df
    )
    
    print(f"\n{demo_specs['area']} sq ft {demo_specs['type']} in {demo_specs['city']}:")
    print(f"  Monthly Rent:  {rent_price:,.0f} PKR")
    print(f"  Sale Price:    {sale_price:,.0f} PKR ({sale_price/1000000:.2f}M)")
    
    if sale_price > 0:
        roi_years = (sale_price / (rent_price * 12)) if rent_price > 0 else 0
        print(f"  ROI Period:    {roi_years:.1f} years")
    
    # Interactive prediction loop
    while True:
        try:
            print("\n" + "-"*50)
            print("Enter property details:")
            
            city = input(f"City ({'/'.join(cities[:3])}): ").strip().title()
            if city not in cities:
                city = cities[0]
                print(f"Using: {city}")
            
            prop_type = input(f"Property Type ({'/'.join(prop_types[:3])}): ").strip().title()
            if prop_type not in prop_types:
                prop_type = prop_types[0]
                print(f"Using: {prop_type}")
            
            purpose = input("Purpose (For Sale/For Rent): ").strip().title()
            if purpose not in ['For Sale', 'For Rent']:
                purpose = 'For Sale'
                print(f"Using: {purpose}")
            
            area_size = float(input("Area Size (sq ft): "))
            bedrooms = int(input("Bedrooms: "))
            bathrooms = int(input("Bathrooms: "))
            
            # Make prediction
            prediction = model.predict_price(
                area_size, bedrooms, bathrooms, city, prop_type, purpose, df
            )
            
            # Display results
            print(f"\n === PREDICTION RESULTS ===")
            purpose_text = "Monthly Rent" if purpose == 'For Rent' else "Sale Price"
            print(f" {purpose_text}: {prediction:,.0f} PKR")
            
            if prediction > 1000000:
                print(f" In Millions: {prediction/1000000:.2f}M PKR")
            
            print(f" Price per sq ft: {prediction/area_size:,.0f} PKR/sq ft")
            
            # Market comparison
            similar_props = df[
                (df['city'] == city) & 
                (df['property_type'] == prop_type) & 
                (df['purpose'] == purpose)
            ]['price']
            
            if len(similar_props) > 10:
                market_median = similar_props.median()
                print(f"\n Market Comparison:")
                print(f" Market Median: {market_median:,.0f} PKR")
                
                deviation = (prediction - market_median) / market_median * 100
                if abs(deviation) < 15:
                    print(f" Status: Within market range ({deviation:+.1f}%)")
                elif deviation > 0:
                    print(f" Status: Above market (+{deviation:.1f}%)")
                else:
                    print(f" Status: Below market ({deviation:.1f}%)")
            
            another = input("\nAnother prediction? (y/n): ").strip().lower()
            if another != 'y':
                break
                
        except ValueError:
            print("Please enter valid numbers.")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n=== COMPLETE ===")
    print("Production-ready models available for deployment!")

if __name__ == '__main__':
    main()
