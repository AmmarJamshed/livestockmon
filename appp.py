#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
import pandas as pd
import numpy as np

app = FastAPI()

# --- Static Data ---
# Dish-Ingredient Mapping
dish_ingredient = pd.DataFrame({
    'Dish': ['Pizza', 'Pizza', 'Pizza', 'Burger', 'Burger', 'Burger', 'Pasta', 'Pasta', 'Pasta'],
    'Ingredient': ['Flour', 'Cheese', 'Tomato Sauce', 'Bun', 'Beef Patty', 'Cheese', 'Flour', 'Tomato Sauce', 'Cheese'],
    'QuantityPerDish': [200, 100, 50, 1, 1, 50, 100, 100, 50]
})

# Current Inventory
inventory = pd.DataFrame({
    'Ingredient': ['Flour', 'Cheese', 'Tomato Sauce', 'Bun', 'Beef Patty'],
    'QuantityAvailable': [5000, 3000, 2000, 100, 100]
})

# List of dishes
dishes = ['Pizza', 'Burger', 'Pasta']

# --- Helper Functions ---

def generate_past_orders():
    """Simulate past 10 hours of order data"""
    np.random.seed(42)
    hours_past = range(1, 11)
    data = []
    for hour in hours_past:
        for dish in dishes:
            orders = np.random.poisson(lam=8)
            data.append([hour, dish, orders])
    return pd.DataFrame(data, columns=['Hour', 'Dish', 'Orders'])

def forecast_next_orders(orders_df):
    """Simple moving average forecast for next 5 hours"""
    forecast_list = []
    for dish in dishes:
        dish_orders = orders_df[orders_df['Dish'] == dish]
        moving_avg = dish_orders['Orders'].rolling(window=3).mean().iloc[-1]
        for h in range(11, 16):  # next 5 hours
            forecast_list.append([h, dish, round(moving_avg)])
    return pd.DataFrame(forecast_list, columns=['Hour', 'Dish', 'ForecastedOrders'])

def predict_shortages(forecast_df):
    """Predict shortages based on forecasted orders"""
    forecast_full = forecast_df.merge(dish_ingredient, on='Dish')
    forecast_full['TotalIngredientNeeded'] = forecast_full['ForecastedOrders'] * forecast_full['QuantityPerDish']
    
    total_ingredient_need = forecast_full.groupby('Ingredient')['TotalIngredientNeeded'].sum().reset_index()
    
    inventory_status = inventory.merge(total_ingredient_need, on='Ingredient', how='left').fillna(0)
    inventory_status['QuantityLeft'] = inventory_status['QuantityAvailable'] - inventory_status['TotalIngredientNeeded']
    
    shortages = inventory_status[inventory_status['QuantityLeft'] < 0]
    shortages = shortages[['Ingredient', 'QuantityLeft']]
    
    return shortages

# --- API Routes ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Restaurant Inventory Predictor API!"}

@app.get("/forecast")
def get_forecast():
    orders_df = generate_past_orders()
    forecast_df = forecast_next_orders(orders_df)
    shortages = predict_shortages(forecast_df)
    return shortages.to_dict(orient="records")

